import os

import time
import multiprocessing as mp
import traceback
from typing import Tuple, Optional, List

import dill
import numpy as np

from pytabkit.bench.scheduling.jobs import JobRunner
from pytabkit.bench.scheduling.resource_manager import ResourceManager, JobInfo
from pytabkit.bench.scheduling.resources import NodeResources, SystemResources
from pytabkit.models.utils import FunctionProcess


def measure_node_resources(node_id: int) -> Tuple[NodeResources, NodeResources]:
    """
    Function that measures available resources.

    :param node_id: Node ID that will be used to identify the node in the returned NodeResources.
    :return: Returns a tuple of NodeResources objects. The first one contains the total available resources,
        and the second one contains the resources that a single process
        (with PyTorch GPU usage) uses without doing anything.
    """
    import torch
    n_gpus = torch.cuda.device_count()

    if n_gpus > 0:
        # init cuda
        # alloc dummy tensors to know how much memory PyTorch uses for its runtime
        dummy_tensors = [torch.ones(1).to(f'cuda:{i}') for i in range(n_gpus)]
        import pynvml
        pynvml.nvmlInit()

        gpu_rams_gb = []
        gpu_rams_fixed_gb = []

        for i in range(n_gpus):
            # adapted torch.cuda.list_gpu_processes(gpu)
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            total = info.total
            # print(f'free     : {info.free}')
            used = info.used
            gpu_rams_gb.append(total / (1024. ** 3))
            gpu_rams_fixed_gb.append(used / (1024. ** 3))
    else:
        gpu_rams_gb = []
        gpu_rams_fixed_gb = []

    import psutil
    import os
    cpu_ram_gb = psutil.virtual_memory().available / (1024. ** 3)
    cpu_ram_fixed_gb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
    n_threads = mp.cpu_count()
    n_physical_cores = n_threads // 2

    node_resources = NodeResources(node_id=node_id, n_threads=n_threads, cpu_ram_gb=cpu_ram_gb,
                                   gpu_usages=np.ones(n_gpus), gpu_rams_gb=np.asarray(gpu_rams_gb),
                                   physical_core_usages=np.ones(n_physical_cores))
    fixed_node_resources = NodeResources(node_id=node_id, n_threads=0.0, cpu_ram_gb=cpu_ram_fixed_gb,
                                         gpu_usages=np.zeros(n_gpus),
                                         gpu_rams_gb=np.asarray(gpu_rams_fixed_gb),
                                         physical_core_usages=np.zeros(n_physical_cores))
    # print('measure_gpu_resources:', gpu_rams_gb, gpu_rams_fixed_gb)
    # return np.asarray(gpu_rams_gb), np.asarray(gpu_rams_fixed_gb)
    return node_resources, fixed_node_resources


def node_runner(feedback_queue, job_queue, node_id: int):
    mp.set_start_method('spawn', force=True)

    # get resources in separate process so CUDA runtime is shut down when the process is terminated
    # this means that this process will not use up CUDA memory all the time
    node_resources, fixed_node_resources = FunctionProcess(measure_node_resources, node_id).start().pop_result()

    feedback_queue.put((node_resources, fixed_node_resources))

    processes = []
    process_rams_gb = []

    # print(f'DEBUG: start loop', flush=True)

    while True:
        # get new jobs from queue
        while not job_queue.empty():
            try:
                job_str = job_queue.get(timeout=0.1)
                # print(f'DEBUG: got job str', flush=True)
            except Exception as e:
                print(traceback.format_exc())
                # might have been queue.Empty or ray.util.queue.Empty exception
                break  # queue is empty
            if job_str is False:  # termination signal
                # cannot use None as termination signal since that is already the timeout signal
                return  # or check if processes are still running?

            job_data = dill.loads(job_str)
            # print(f'DEBUG: got job data', flush=True)
            processes.append(FunctionProcess(JobRunner(*job_data)).start())
            process_rams_gb.append(0.0)

        # check for finished processes
        for i, p in enumerate(processes):
            process_rams_gb[i] = max(process_rams_gb[i], p.get_ram_usage_gb())
            if p.is_done():
                result = p.pop_result()
                result.set_max_cpu_ram_gb(process_rams_gb[i])
                # print(f'Node {node_id}: Before putting result in feedback_queue', flush=True)
                feedback_queue.put(result)
                # print(f'Node {node_id}: After putting result in feedback_queue', flush=True)
                del processes[i]
                del process_rams_gb[i]

        # print(f'.', end='', flush=True)

        time.sleep(0.01)

    # get RAM statistics of all processes and total RAM usage
    # if any process is finished, send time and RAM statistics of that process through the feedback queue
    # maybe have a logging queue?


class NodeManager:
    def start(self):
        raise NotImplementedError()  # start nodes, return queues and node ids?

    def terminate(self):
        raise NotImplementedError()  # terminate nodes?


class RayJobManager(NodeManager):
    def __init__(self, max_n_threads: Optional[int] = None, available_cpu_ram_multiplier: float = 1.0, **ray_kwargs):
        self.ray_kwargs = ray_kwargs
        self.runner_futures = []  # keep node_runner futures for termination
        self.job_queues = []
        self.feedback_queues = []
        self.resource_manager: Optional[ResourceManager] = None
        self.max_n_threads = max_n_threads
        self.available_cpu_ram_multiplier = available_cpu_ram_multiplier

    def start(self) -> None:
        import ray
        # take some ray arguments from os.environ if available
        for (ray_name, environ_name) in [('address', 'ip_head'), ('_redis_password', 'redis_password')]:
            if environ_name in os.environ and ray_name not in self.ray_kwargs:
                self.ray_kwargs[ray_name] = os.environ[environ_name]
        ray.init(**self.ray_kwargs)
        from ray.util import queue
        nodes = ray.nodes()
        print(f'Nodes: {nodes}')
        feedback_queues = [queue.Queue() for i in range(len(nodes))]
        job_queues = [queue.Queue() for i in range(len(nodes))]

        for i, node in enumerate(nodes):
            node_id = f'node:{node["NodeManagerAddress"]}'
            num_gpus = 0 if 'GPU' not in node['Resources'] else round(node['Resources']['GPU'])
            future = ray.remote(num_gpus=num_gpus)(node_runner).options(resources={node_id: 1.0}) \
                .remote(feedback_queue=feedback_queues[i], job_queue=job_queues[i], node_id=i)
            self.runner_futures.append(future)

        print(f'Started {len(job_queues)} nodes', flush=True)
        n_nodes = len(job_queues)
        total_resources: List[Optional[NodeResources]] = [None] * n_nodes
        fixed_resources: List[Optional[NodeResources]] = [None] * n_nodes

        for feedback_queue in feedback_queues:
            nr, fnr = feedback_queue.get()  # should be a NodeResources object
            total_resources[nr.node_id] = nr
            fixed_resources[fnr.node_id] = fnr
            if self.max_n_threads is not None:
                total_resources[nr.node_id].set_n_threads(min(total_resources[nr.node_id].get_n_threads(),
                                                              self.max_n_threads))
            total_resources[nr.node_id].set_cpu_ram_gb(
                self.available_cpu_ram_multiplier * total_resources[nr.node_id].get_cpu_ram_gb())

        print(f'Acquired node resources', flush=True)

        self.resource_manager = ResourceManager(total_resources=SystemResources(total_resources),
                                                fixed_resources=SystemResources(fixed_resources))

        self.job_queues = job_queues
        self.feedback_queues = feedback_queues

    def get_resource_manager(self) -> ResourceManager:
        if self.resource_manager is None:
            raise RuntimeError('called get_resource_manager() before start()')
        return self.resource_manager

    def submit_job(self, job_info: JobInfo) -> None:
        if self.resource_manager is None:
            raise RuntimeError('called submit_job() before start()')
        job = job_info.job
        job_id = job_info.job_id
        assigned_resources = job_info.assigned_resources
        if assigned_resources is None:
            raise RuntimeError('assigned_resources for submitted job must not be None')
        node_id = assigned_resources.node_id
        print(f'Scheduling job {job.get_desc()} on node {node_id}', flush=True)
        job_str = dill.dumps((job, job_id, assigned_resources))
        self.job_queues[node_id].put(job_str)
        self.resource_manager.job_started(job_info)

    def pop_finished_job_infos(self, timeout_s: float = -1.0) -> List[JobInfo]:
        if self.resource_manager is None:
            raise RuntimeError('called pop_results() before start()')
        has_new_result = False
        start_time = time.time()
        job_infos = []

        while not has_new_result:
            if timeout_s > 0.0 and time.time() > start_time + timeout_s:
                # timeout
                return job_infos

            for feedback_queue in self.feedback_queues:
                while not feedback_queue.empty():
                    job_result = feedback_queue.get()
                    job_info = self.resource_manager.job_finished(job_result)
                    job_infos.append(job_info)
                    has_new_result = True

            if not has_new_result:
                time.sleep(0.05)

        return job_infos

    def terminate(self) -> None:
        for jq in self.job_queues:
            jq.put(False)  # termination signal

        import ray
        # maybe wait only a bit and then hard terminate otherwise?
        ray.get(self.runner_futures)
        ray.shutdown()


# class LocalNodeManager(NodeManager):
#     # start node_runner in a thread
#     pass
