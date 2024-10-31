# Overview and Installation of the Benchmarking code

Our benchmarking code contains several features:

- Automatic dataset download
- Running models (parallelized) with automatic scheduling, 
trying to respect RAM constraints
- Evaluation and plotting

## Installation

Our code has been tested with python 3.9 and 3.10.
After cloning/forking the repo, 
the required libraries can be installed as follows:

```commandline
# in the repo folder:
pip3 install -e .[extra,hpo,bench]
```

Note that the version requirements in our `pyproject.toml` 
are somewhat restrictive to avoid problems, they can potentially be relaxed.

To more closely reproduce the installation we used for running the benchmarks, 
we refer to the configuration files in the `original_requirements` folder:
- The pip-only requirements in `requirements_2024_06_25.txt` 
were used to compute many of the older NN results (not TabR).
- The conda requirements in `conda_env_2024_06_25.yml` 
and `conda_env_2024_10_28.yml` were used to compute GBDT-HPO results 
and TabR results as well as a few newer NN results. 
They can be installed as a new conda environment using 
`conda env create -f conda_env_2024_10_28.yml`. 
Note that the older of the two conda environments was very slow 
for TabR on some datasets 
since it uses an older torchmetrics version with slow implementations.


## Using Sphinx Documentation
Go to the repo root dir and run
```commandline
sphinx-apidoc -o docs/source/ pytabkit
sphinx-build -M html docs/source/ docs/build/
```
then open `docs/build/html/index.html`.