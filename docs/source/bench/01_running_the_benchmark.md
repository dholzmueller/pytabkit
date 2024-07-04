# Running the benchmark

## Configuration of data paths

The paths for storing data and results are configured
through the `tab_bench.data.paths.Paths` class. 
There are several options to configure which folders are used, 
which will be automatically recognized by `Paths.from_env_variables()`:

- **Through environmental variables**: 
The base folder can be configured by setting the environmental variable
`TAB_BENCH_DATA_BASE_FOLDER`. 
Optionally, some sub-folders can be set separately 
(e.g. for moving them to another partition). These are
`TAB_BENCH_DATA_TASKS_FOLDER`, `TAB_BENCH_DATA_RESULTS_FOLDER`,
`TAB_BENCH_DATA_RESULT_SUMMARIES_FOLDER`, `TAB_BENCH_DATA_UCI_DOWNLOAD_FOLDER`.
- **Through a python file**: If `TAB_BENCH_DATA_BASE_FOLDER` is not available, 
the code will try to get the base folder (as a string) from
`scripts.custom_paths.get_base_folder()`.
This can be implemented by copying `scripts/custom_paths.py.default` to `scripts/custom_paths.py`
  (ignored by git) and adjusting the path therein.
- If neither of the two options above is used, 
all data will be stored in `./tab_bench_data`.


## Download datasets

To download all datasets for the meta-train and meta-test benchmarks, run 
(with your desired OpenML cache directory)
```commandline
python3 scripts/download_data.py openml_cache_dir
```
To run methods on the benchmarks, there are two options:

## Run experiments with slurm

Our benchmarking code contains its own scheduling code that will start subprocesses 
for each algorithm-dataset-split combination.
Therefore, it is in principle possible to run all experiments 
through a single slurm job,
though experiments can be divided into smaller pieces by running them separately.

Run the following command (replacing some of the parameters with your own values) on the login node:
```commandline
python3 scripts/ray_slurm_launch.py --exp_name=my_exp_name --num_nodes=num_nodes --queue="queue_name" --time=24:00:00 --mail_user="my@address.edu" --log_folder=log_folder --command="python3 -u run_slurm.py"
```
This will submit a job to the configured queue that will run `run_slurm.py` and create logfiles.
Your experiments then have to be configured in `run_slurm.py`, see below.
Multi-node is supported: `ray` will start instances on each node
and our benchmarking code will schedule the individual experiments on the nodes.

## Run experiments without slurm

Run the file with the corresponding experiments directly. 
For example, many of our experiment configurations 
can be found in `run_final_experiments.py`. 
One possible way to run the experiments detached from the shell with log-files is
````commandline
systemd-run --scope --user python3 -u scripts/run_final_experiments.py > ./out.log 2> ./err.log &
````

## Time measurements

For time measurements, simply run `scripts/run_time_measurements.py` (with or without slurm).
Results can be printed using `scripts/print_runtimes.py` 
(but these are averaged total times, not averaged per 1K samples as in the paper).

## Evaluating the benchmark results

Aggregated algorithm results can be printed using 
````commandline
python3 scripts/run_evaluation.py meta-train-class
````
where `meta-train-class` can be replaced by the name of any other task collection 
(that is stored in the `task_collections` folder in the configured data directory),
or a single dataset such as `openml-class/Higgs`. 
This script also has many more command line options, see the python file.
For example, one can print only those methods with a certain tag 
using the `--tag` option,
print results on individual datasets, for different metrics, etc.
The parameters are the same as the ones of the following method:
```{eval-rst}  
.. autofunction:: scripts.run_evaluation.show_eval
```

## Creating plots and tables

Plots and tables can be created using
````commandline
python3 scripts/create_plots_and_tables.py
````
The plots without missing value datasets require running
```commandline
python3 scripts/check_missing_values.py
```
beforehand.

## Single-task experiments

You can also run a configuration on a single data set,
without saving the results, by adjusting and running `scripts/run_single_task.py`.

## Other utilities

- Use `scripts/analyze_tasks.py` to print some dataset statistics.
- You can rename a method using `python3 scripts/rename_alg.py old_name new_name`.
- We used some code in `scripts/meta_hyperopt.py` to optimize the default parameters for GBDTs.
- The code in `scripts/estimate_resource_params.py` has been used to get more precise estimates 
for RAM usage etc. for running methods on the benchmark.
- `scripts/print_complete_results.py` can be used to check which methods have results available 
on all splits for all tasks in a given collection.