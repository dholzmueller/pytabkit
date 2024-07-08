# Downloading the benchmark results

The benchmark data (as well as the code)
is archived at [DaRUS](https://doi.org/10.18419/darus-4255).
To download the benchmark data, please
- create a folder for the data
- in the folder, unpack `main_no_results.tar.gz`, 
this should create the folders `algs`, `result_summaries`, `times` and `plots`.
Since `result_summaries` stores the main metrics of the results, 
this is already enough for plotting/evaluating the results. 
- If you want the non-summarized results, 
download and unpack `results_small.tar.gz`, which contains the `results` folder.
However, this does not contain the additional files storing the predictions 
and optimal hyperparameters.
- If you want the full results, download and unpack
`main_results.tar.gz` (260 GB!) into the results folder 
(overwriting/replacing the contents of `results_small.tar.gz`)
Moreover, there are additional files containing the results 
of the individual random search steps
for the different methods, 
which could be used for retrospectively optimizing on a different metric etc.