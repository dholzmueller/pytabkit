import time
import traceback
import sys
from typing import Optional

from pytabkit.bench.scheduling.resources import NodeResources
from pytabkit.models.alg_interfaces.base import RequiredResources


class JobResult:
    """
    Helper class to store information about a job that has been run.
    """
    def __init__(self, job_id: int, time_s: float,
                 oom_cpu: bool = False, oom_gpu: bool = False, finished_normally: bool = True,
                 exception_msg: Optional[str] = None):
        """
        :param job_id: Job id.
        :param time_s: Time in seconds that the job ran for.
        :param oom_cpu: Whether an out-of-memory error occured on the CPU.
        :param oom_gpu: Whether an out-of-memory error occured on the GPU.
        :param finished_normally: Whether the job ran normally,
            such that its time and RAM values are representative of how it would normally run.
            For example, if the job ran faster because the results were already partially precomputed,
            it should not count towards the time estimation. Of course, if an exception occured,
            we should have finished_normally=False.
        :param exception_msg: Exception message (if there was any).
        """
        self.job_id = job_id
        self.time_s = time_s
        self.oom_cpu = oom_cpu
        self.oom_gpu = oom_gpu
        self.finished_normally = finished_normally
        self.exception_msg = exception_msg
        self.failed = exception_msg is not None
        self.max_cpu_ram_gb = 0.0
        assert exception_msg is None or not finished_normally

    def set_max_cpu_ram_gb(self, value: float) -> None:
        """
        Set the maximum RAM usage of the job.
        :param value: maximum RAM usage in GiB.
        """
        self.max_cpu_ram_gb = value


class AbstractJob:
    """
    Abstract base class for jobs that can be scheduled using schedulers in schedulers.py.
    """
    def get_group(self) -> str:
        """
        :return: Should return a "group name" string. All jobs with the same "group name" will have
            a common time factor that is adjusted on-the-fly during scheduling based on already completed jobs.
        """
        raise NotImplementedError()

    def __call__(self, assigned_resources: NodeResources) -> bool:
        """
        Should perform the main computation of the job.
        Problematic exceptions should not be caught within this method,
        they will be caught and printed in the scheduler.

        :param assigned_resources: Resources that are assigned to this job
            (conforming with the resources requested in get_required_resources()).
        :return: Should return True if the execution finished normally
            such that the timing of this job is representative.
            In cases where pre-computed results were available such that the job is shorter than usual, return False.
        """
        raise NotImplementedError()

    def get_required_resources(self) -> RequiredResources:
        """
        :return: Return the resources requested by this job.
        """
        raise NotImplementedError()

    def get_desc(self) -> str:
        """
        :return: Return a description that can be logged, e.g., when the job is started and when it finishes.
        """
        raise NotImplementedError()


class JobRunner:
    """
    Helper class that runs an AbstractJob, catches exceptions, measures time and RAM usage, and returns its result.
    """
    def __init__(self, job: AbstractJob, job_id: int, assigned_resources: NodeResources):
        """
        :param job: The job to be run.
        :param job_id: An ID that will be returned at the end so that the job can be identified.
        :param assigned_resources: Assigned resources to run the job.
        """
        self.job = job
        self.job_id = job_id
        self.assigned_resources = assigned_resources

    def __call__(self) -> JobResult:
        """
        Runs the job computation.

        :return: Returns a JobResult object that includes information about the job.
        """
        start_time = time.time()
        oom_gpu = False
        oom_cpu = False
        exception_msg = None
        try:
            finished_normally = self.job(self.assigned_resources)
        except Exception as e:
            finished_normally = False
            exception_msg = traceback.format_exc()
            print(exception_msg, file=sys.stderr, flush=True)
            if isinstance(e, MemoryError):
                oom_cpu = True
            elif isinstance(e, RuntimeError) and 'cuda out of memory' in exception_msg.lower():
                oom_gpu = True
            elif isinstance(e, KeyboardInterrupt):
                raise e

        end_time = time.time()

        return JobResult(job_id=self.job_id, time_s=end_time-start_time,
                         oom_cpu=oom_cpu, oom_gpu=oom_gpu, finished_normally=finished_normally,
                         exception_msg=exception_msg)
