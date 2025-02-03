# Reproducing results of "Rethinking Early Stopping: Refine, Then Calibrate"

Here, we document how to reproduce results from our paper [Rethinking Early Stopping: Refine, Then Calibrate](https://arxiv.org/abs/2501.19195).
For general instructions on how to set data paths and use slurm, 
we refer to the installation page. 
The following will be the parts specific to this paper.

## Installation

```bash
pip install probmetrics[extra]  # to get smECE
pip install pytabkit[bench,dev]
```

### Original environment

The original conda environment for exact reproduction 
is stored in `original_requirements/conda_env_2025_01_15.yml`.

## Downloading datasets

Download the zipped datasets (`dataset-1218.zip`) of the TALENT benchmark from
[here](https://drive.google.com/drive/folders/1j1zt3zQIo8dO6vkO-K-WE6pSrl71bf0z).
Extract them into a folder. Then, use

```commandline
python3 scripts/download_data.py --import_talent_class_small --talent_folder=<unzipped data folder>
```

where the provided data folder should be the `data` folder inside the unzipped results.

## Running experiments

Experiments can be run using `python3 scripts/run_probclass_experiments.py`,
then plots can be generated using `python3 scripts/create_probclass_plots.py`.
