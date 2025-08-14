# Code structure

## Algorithm wrappers

To run methods in `tab_bench`, one needs to 
provide them as a subclass of `tab_bench.alg_wrappers.general.AlgWrapper`.
Generally, we use models from the `tab_models` library that implement 
the `AlgInterface` from there, and wrap them lightly as an `AlgInterfaceWrapper`
in `tab_bench/alg_wrappers/interface_wrappers.py`, 
see the numerous classes there for examples. 
As in `tab_models`, we pass parameters to these models via `**kwargs`.
The scikit-learn interfaces in `tab_models` provide in their constructors
a list of the most important hyperparameters.

## Datasets

We represent our datasets using the `DictDataset` class from `tab_models`.
These datasets can be loaded as follows:

```python
from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskDescription

paths = Paths.from_env_variables()
task_desc = TaskDescription('openml-reg', 'fifa')
task_info = task_desc.load_info(paths)  # a TaskInfo object
task = task_info.load_task(paths)
ds = task.ds  # this is the DictDataset object
```

We can convert `ds` to a Pandas DataFrame using `ds.to_df()`. 
It is also possible to load a list of all TaskInfo objects
for an entire task collection:

```python
from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskCollection

paths = Paths.from_env_variables()
task_infos = TaskCollection.from_name('meta-train-class', paths).load_infos(paths)
```

## Scheduling code

We implement general scheduling code in `tab_bench/scheduling`. 
This code can take a list of jobs with certain functionalities 
and run them in parallel in a single-node or multi-node setup, 
respecting the provided resource requirements 
(on RAM usage, number of threads, etc.). It can be used independently as follows:

```python
from typing import List
from pytabkit.bench.scheduling.jobs import AbstractJob
from pytabkit.bench.scheduling.execution import RayJobManager
from pytabkit.bench.scheduling.schedulers import SimpleJobScheduler

jobs: List[AbstractJob] = []  # create a list of jobs here
scheduler = SimpleJobScheduler(RayJobManager())
scheduler.add_jobs(jobs)
scheduler.run()
```

For our tabular benchmarking code, 
the `AbstractJob` objects will be created by the
`tab_bench.run.task_execution.TabBenchJobManager`.
Numerous examples for this can be found in `run_final_experiments.py`.

## Resource estimation

## Evaluation and plotting

