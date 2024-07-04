import copy
import enum
import time
from typing import Optional

from pytabkit.bench.scheduling.jobs import AbstractJob, JobResult
from pytabkit.bench.scheduling.resources import NodeResources, SystemResources


class JobStatus(enum.Enum):
    REMAINING = 0
    RUNNING = 1
    SUCCEEDED = 2
    FAILED = 3


class JobInfo:
    def __init__(self, job: AbstractJob, job_id: int, start_time: Optional[float] = None,
                 assigned_resources: Optional[NodeResources] = None, job_result: Optional[JobResult] = None):
        self.job = job
        self.job_id = job_id
        self.start_time = start_time
        self.assigned_resources = assigned_resources
        self.required_resources = job.get_required_resources()
        self.job_result = job_result

    def get_status(self) -> JobStatus:
        if self.start_time is None:
            return JobStatus.REMAINING
        elif self.job_result is None:
            return JobStatus.RUNNING
        elif self.job_result.failed:
            return JobStatus.FAILED
        else:
            return JobStatus.SUCCEEDED

    def set_started(self, assigned_resources: NodeResources):
        self.start_time = time.time()
        self.assigned_resources = assigned_resources

    def set_finished(self, job_result: JobResult):
        self.job_result = job_result

    def is_remaining(self):
        return self.get_status() == JobStatus.REMAINING

    def is_running(self):
        return self.get_status() == JobStatus.RUNNING

    def is_finished(self):
        return self.get_status() in [JobStatus.FAILED, JobStatus.SUCCEEDED]

    def is_failed(self):
        return self.get_status() == JobStatus.FAILED

    def is_succeed(self):
        return self.get_status() == JobStatus.SUCCEEDED


class ResourceManager:
    """
    Keeps track of running jobs and available resources.
    """
    def __init__(self, total_resources: SystemResources, fixed_resources: SystemResources):
        self.total_resources = total_resources
        self.fixed_resources = fixed_resources
        self.running_job_infos = dict()  # map job_id to job_info

    def get_fixed_resources(self):
        return self.fixed_resources

    def get_total_resources(self):
        return self.total_resources

    def get_free_resources(self):
        free_resources = copy.deepcopy(self.total_resources)

        for ji in self.running_job_infos.values():
                ar = ji.assigned_resources
                free_resources.resources[ar.node_id] -= ar

        return free_resources

    def job_started(self, job_info: JobInfo):
        job_info.start_time = time.time()
        if job_info.job_id in self.running_job_infos:
            raise RuntimeError(f'Trying to start job {job_info.job.get_desc()}, which is already running!')
        self.running_job_infos[job_info.job_id] = job_info

    def job_finished(self, job_result: JobResult) -> JobInfo:
        ji = self.running_job_infos[job_result.job_id]
        ji.set_finished(job_result)
        if job_result.exception_msg is not None:
            print(f'Job failed: {ji.job.get_desc()}\nException: {job_result.exception_msg}')
        del self.running_job_infos[job_result.job_id]
        return ji
