# Using the scheduler

`pytabkit` includes a flexible scheduler that can schedule jobs within python using `ray` and `multiprocessing`.
Essentially, it is a much fancier version of `multiprocessing.Pool`.
Custom jobs need to provide an estimate of their required resources. The scheduler will
- run as many jobs in parallel as possible on the current hardware while respecting the RAM and resource constraints
- try to run the slowest jobs first, to avoid waiting for a few slow jobs in the end
- measure free CPU RAM in the beginning, and add the fixed RAM that a CPU process uses to the requested RAM. 
  For processes requesting a GPU, the fixed RAM used by a process using torch CUDA will be added to the requested RAM.
- print info including remaining time estimates after each new started job, failed jobs etc.
  (unless the jobs run so fast that multiple ones are started at once). 
  The time estimates will be based on the time estimates by the jobs, 
  but they will be adapted by a factor learned based on the actual time taken by already finished jobs. 
  Hence, the time estimate is only accurate after a few jobs have finished. 
  It often underestimates the actually needed time to some extent.
  (This is probably also due to selection bias, since the estimated longest jobs are run first.)

The scheduler also works on multi-GPU systems,
and it even works on multi-node systems thanks to `ray`'s multi-node support. 
See [`ray_slurm_launch.py`](https://github.com/dholzmueller/pytabkit/blob/main/scripts/ray_slurm_launch.py) 
and [`ray_slurm_template.sh`](https://github.com/dholzmueller/pytabkit/blob/main/scripts/ray_slurm_template.sh).
To use the scheduler, install `pytabkit[models,bench]`.

Here is some example code:

```python
from pytabkit.models.alg_interfaces.base import RequiredResources
from pytabkit.bench.scheduling.execution import RayJobManager
from pytabkit.bench.scheduling.jobs import AbstractJob
from pytabkit.bench.scheduling.resources import NodeResources
from pytabkit.bench.scheduling.schedulers import SimpleJobScheduler

class CustomJob(AbstractJob):
    def get_group(self):
        # group name, for all jobs with the same group name
        # one joint time multiplier will be fitted in the scheduler
        return 'default'

    def get_desc(self) -> str:
        return 'CustomJob'  # name for displaying

    def __call__(self, assigned_resources: NodeResources) -> bool:
        # the main job, should only use the assigned resources
        print(f'Running job with {assigned_resources.get_n_threads()} threads', flush=True)
        return True  # job finished successfully

    def get_required_resources(self) -> RequiredResources:
        # Return the resources requested by this job (RAM should be upper bounds, time doesn't need to be)
        return RequiredResources(time_s=1.0, n_threads=1, cpu_ram_gb=0.1, n_gpus=0, gpu_ram_gb=0.0, gpu_usage=1.0)


sched = SimpleJobScheduler(RayJobManager(available_gpu_ram_multiplier=0.7))
sched.add_jobs([CustomJob() for _ in range(1000)])
sched.run()
```