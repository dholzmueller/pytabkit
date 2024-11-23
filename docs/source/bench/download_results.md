# Downloading the benchmark results

The benchmark data (as well as the code)
is archived at [DaRUS](https://doi.org/10.18419/darus-4555).
To download the benchmark data,
- create a folder for the data 
(which is then linked in the environmental variable 
`TAB_BENCH_DATA_BASE_FOLDER` or in `custom_paths.py`)
- in the folder, unpack `main_no_results.tar.gz`, 
this should create the folders `algs`, `result_summaries`, `times`, `plots`,
`task_collections`, and `tasks_only_infos` 
(which should be renamed to `tasks` if no `tasks` folder has been created).
Since `result_summaries` stores the main metrics of the results, 
this is already enough for plotting/evaluating the results. 
- If you want the non-summarized results, 
download and unpack `results_small.tar.gz`, which contains the `results` folder 
(you might need to rename it from `results_no_gz` to `results`).
However, this does not contain the additional files storing the predictions 
and optimal hyperparameters.
- If you want the full results, download and unpack
`results_main.tar.gz` (180 GB!) into the results folder 
(overwriting/replacing the contents of `results_small.tar.gz`)
Moreover, there are additional files containing the results 
of the individual random search steps
for the different methods, 
which could be used for retrospectively optimizing on a different metric etc. 
The file `cv_refit.tar.gz` contains the results of the cross-validation/refitting experiments, 
which are also somewhat large.
- If you need the datasets (in the `tasks` folder), 
you can normally just obtain it by running `scripts/download_data.py`. 
However, there is the option to request access to download `tasks.tar.gz` directly.