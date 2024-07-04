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
pip3 install -e .[full]
```

Note that the version requirements in our `pyproject.toml` 
are somewhat restrictive to avoid problems, they can potentially be relaxed.


## Using Sphinx Documentation
Go to the repo root dir and run
```commandline
sphinx-apidoc -o docs/source/ pytabkit
sphinx-build -M html docs/source/ docs/build/
```
then open `docs/build/html/index.html`.