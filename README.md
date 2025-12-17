[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dholzmueller/pytabkit/blob/main/examples/tutorial_notebook.ipynb)
[![](https://readthedocs.org/projects/pytabkit/badge/?version=latest&style=flat-default)](https://pytabkit.readthedocs.io/en/latest/)
[![test](https://github.com/dholzmueller/pytabkit/actions/workflows/testing.yml/badge.svg)](https://github.com/dholzmueller/pytabkit/actions/workflows/testing.yml)
[![Downloads](https://img.shields.io/pypi/dm/pytabkit)](https://pypistats.org/packages/pytabkit)

# PyTabKit: Tabular ML models and benchmarking (NeurIPS 2024)

 [Paper](https://arxiv.org/abs/2407.04491) | [Documentation](https://pytabkit.readthedocs.io) | [RealMLP-TD-S standalone implementation](https://github.com/dholzmueller/realmlp-td-s_standalone) | [Grinsztajn et al. benchmark code](https://github.com/LeoGrin/tabular-benchmark/tree/better_by_default) | [Data archive](https://doi.org/10.18419/darus-4555) |
|-------------------------------------------|--------------------------------------------------|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|-----------------------------------------------------|

PyTabKit provides **scikit-learn interfaces for modern tabular classification and regression methods**
benchmarked in our [paper](https://arxiv.org/abs/2407.04491), see below.
It also contains the code we used for **benchmarking** these methods
on our benchmarks.

![Meta-test benchmark results](./figures/meta-test_benchmark_results.png)

## When (not) to use pytabkit

- **To get the best possible results**: 
  - Generally we recommend AutoGluon for the best possible results, 
    though it does not include all the models from pytabkit. AutoGluon 1.4
    includes RealMLP (though not in a default configuration) and TabM (in the "extreme" preset for <= 30K samples).
  - To get the best possible results from `pytabkit`, 
    we recommend using 
    `Ensemble_HPO_Classifier(n_cv=8, use_full_caruana_ensembling=True, use_tabarena_spaces=True, n_hpo_steps=50)` 
    with a `val_metric_name` corresponding to your target metric 
    (e.g., `class_error`, `cross_entropy`, `brier`, `1-auc_ovr`), or the corresponding `Regressor`. 
    (This might take very long to fit.)
  - For only a single model, we recommend using 
    `RealMLP_HPO_Classifier(n_cv=8, hpo_space_name='tabarena-new', use_caruana_ensembling=True, n_hyperopt_steps=50)`,
    also with `val_metric_name` as above, or the corresponding `Regressor`.
- **Models**: [TabArena](https://github.com/AutoGluon/tabarena) 
  also includes some newer models like RealMLP and TabM 
  with more general preprocessing (missing numericals, text, etc.),
  as well as very good boosted tree implementations.
  `pytabkit` is currently still easier to use 
  and supports vectorized cross-validation for RealMLP, 
  which can significantly speed up the training.
- **Benchmarking**: While pytabkit can be good for quick benchmarking for development, 
  for method evaluation we recommend [TabArena](https://github.com/AutoGluon/tabarena).

## Installation (new in 1.4.0: optional model dependencies)

```bash
pip install pytabkit[models]
```

- RealMLP (and TabM) can be used without the `[models]` part.
- For xRFM on GPU, faster kernels will be used if you install `kermac[cu12]` or `kermac[cu11]` 
(depending on your CUDA version).
- If you want to use **TabR**, you have to manually install
  [faiss](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md),
  which is only available on **conda**.
- Please install torch separately if you want to control the version (CPU/GPU etc.)
- Use `pytabkit[models,autogluon,extra,hpo,bench,dev]` to install additional dependencies for the other models,
  AutoGluon models, extra preprocessing,
  hyperparameter optimization methods beyond random search (hyperopt/SMAC),
  the benchmarking part, and testing/documentation. For the hpo part,
  you might need to install *swig* (e.g. via pip) if the build of *pyrfr* fails.
  See also the [documentation](https://pytabkit.readthedocs.io).
  To run the data download for the meta-train benchmark, you need one of rar, unrar, or 7-zip
  to be installed on the system.

## Using the ML models

Most of our machine learning models are directly available via scikit-learn interfaces.
For example, you can use RealMLP-TD for classification as follows:

```python
from pytabkit import RealMLP_TD_Classifier

model = RealMLP_TD_Classifier()  # or TabR_S_D_Classifier, CatBoost_TD_Classifier, etc.
model.fit(X_train, y_train)
model.predict(X_test)
```

The code above will automatically select a GPU if available,
try to detect categorical columns in dataframes,
preprocess numerical variables and regression targets (no standardization required),
and use a training-validation split for early stopping.
All of this (and much more) can be configured through the constructor
and the parameters of the fit() method.
For example, it is possible to do bagging
(ensembling of models on 5-fold cross-validation)
simply by passing `n_cv=5` to the constructor.
Here is an example for some of the parameters that can be set explicitly:

```python
from pytabkit import RealMLP_TD_Classifier

model = RealMLP_TD_Classifier(device='cpu', random_state=0, n_cv=1, n_refit=0,
                              n_epochs=256, batch_size=256, hidden_sizes=[256] * 3,
                              val_metric_name='cross_entropy',
                              use_ls=False,  # for metrics like AUC / log-loss
                              lr=0.04, verbosity=2)
model.fit(X_train, y_train, X_val, y_val, cat_col_names=['Education'])
model.predict_proba(X_test)
```

See [this notebook](https://colab.research.google.com/github/dholzmueller/pytabkit/blob/main/examples/tutorial_notebook.ipynb)
for more examples. Missing numerical values are currently *not* allowed and need to be imputed beforehand.

### Available ML models

Our ML models are available in up to three variants, all with best-epoch selection:

- library defaults (D)
- our tuned defaults (TD)
- random search hyperparameter optimization (HPO), 
  sometimes also tree parzen estimator (HPO-TPE) or weighted ensembling (Ensemble)

We provide the following ML models:

- **RealMLP** (TD, HPO, Ensemble): Our new neural net models with tuned defaults (TD),
  random search hyperparameter optimization (HPO), or Ensembling
- **XGB**, **LGBM**, **CatBoost** (D, TD, HPO, HPO-TPE): Interfaces for gradient-boosted
  tree libraries XGBoost, LightGBM, CatBoost
- **MLP**, **ResNet**, **FTT** (D, HPO): Models
  from [Revisiting Deep Learning Models for Tabular Data](https://proceedings.neurips.cc/paper_files/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html)
- **MLP-PLR** (D, HPO): MLP with numerical embeddings
  from [On Embeddings for Numerical Features in Tabular Deep Learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html)
- **TabR** (D, HPO): TabR model
  from [TabR: Tabular Deep Learning Meets Nearest Neighbors](https://openreview.net/forum?id=rhgIgTSSxW)
- **TabM** (D, HPO): TabM model
  from [TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling](https://arxiv.org/abs/2410.24210)
- **XRFM** (D, HPO): xRFM model from [here](https://arxiv.org/abs/2508.10053) ([original repo](https://github.com/dmbeaglehole/xRFM))
- **RealTabR** (D): Our new TabR variant with default parameters
- **Ensemble-TD**: Weighted ensemble of all TD models (RealMLP, XGB, LGBM, CatBoost)

## Post-hoc calibration and refinement stopping

For using post-hoc temperature scaling and refinement stopping from our 
paper [Rethinking Early Stopping: Refine, Then Calibrate](https://arxiv.org/abs/2501.19195),
you can pass the following parameters to the scikit-learn interfaces:
```python
from pytabkit import RealMLP_TD_Classifier
clf = RealMLP_TD_Classifier(
    val_metric_name='ref-ll-ts',  # short for 'refinement_logloss_ts-mix_all'
    calibration_method='ts-mix',  # temperature scaling with laplace smoothing
    use_ls=False  # recommended for cross-entropy loss
)
```
Other calibration methods and validation metrics
from [probmetrics](https://github.com/dholzmueller/probmetrics)
can be used as well.

For reproducing the results from this paper, we refer to the
[documentation](https://pytabkit.readthedocs.io/en/latest/bench/refine_then_calibrate.html).

## Benchmarking code

Our benchmarking code has functionality for

- dataset download
- running methods highly parallel on single-node/multi-node/multi-GPU hardware,
  with automatic scheduling and trying to respect RAM constraints
- analyzing/plotting results

For more details, we refer to the [documentation](https://pytabkit.readthedocs.io).

## Preprocessing code

While many preprocessing methods are implemented in this repository,
a standalone version of our robust scaling + smooth clipping
can be found [here](https://github.com/dholzmueller/realmlp-td-s_standalone/blob/main/preprocessing.py#L65C7-L65C37).

## Citation

If you use this repository for research purposes, please cite our [paper](https://arxiv.org/abs/2407.04491):

```
@inproceedings{holzmuller2024better,
  title={Better by default: {S}trong pre-tuned {MLPs} and boosted trees on tabular data},
  author={Holzm{\"u}ller, David and Grinsztajn, Leo and Steinwart, Ingo},
  booktitle = {Neural {Information} {Processing} {Systems}},
  year={2024}
}
```

## Contributors

- David Holzmüller (main developer)
- Léo Grinsztajn (deep learning baselines, plotting)
- Ingo Steinwart (UCI dataset download)
- Katharina Strecker (PyTorch-Lightning interface)
- Daniel Beaglehole (part of the xRFM implementation)
- Lennart Purucker (some features/fixes)
- Jérôme Dockès (deployment, continuous integration)

## Acknowledgements

Code from other repositories is acknowledged as well as possible in code comments.
Especially, we used code from https://github.com/yandex-research/rtdl
and sub-packages (Apache 2.0 license),
code from https://github.com/catboost/benchmarks/
(Apache 2.0 license),
and https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html
(Apache 2.0 license).

## Releases (see git tags)

- v1.7.2: 
    - Added scikit-learn 1.8 compatibility.
    - Removed debug print in RealMLP.
    - fixed device memory estimation error in the scheduler when `CUDA_VISIBLE_DEVICES` was used.
- v1.7.1:
    - LightGBM now processes the `extra_trees`, `max_cat_to_onehot`, and `min_data_per_group` parameters 
      used in the `'tabarena'` search space, which should improve results.
    - Scikit-learn interfaces for RealMLP (TD, HPO) now support moving the model to a different device 
      (e.g., before saving). This can be achived using, e.g., `model.to('cpu')` (which is in-place).
    - Fixed an xRFM bug in handling binary categorical features.
- v1.7.0:
    - added [xRFM](https://arxiv.org/abs/2508.10053) (D, HPO)
    - added new `'tabarena-new'` search space for RealMLP-HPO, including per-fold ensembling (more expensive)
      and tuning two more categorical hyperparameters
      (with [better results](https://github.com/autogluon/tabarena/pull/195))
    - reduced RealMLP pickle size by not storing the dataset ([#33](https://github.com/dholzmueller/pytabkit/issues/33))
    - fixed gradient clipping for TabM 
      (it did nothing previously, see [#34](https://github.com/dholzmueller/pytabkit/issues/34)).
      To ensure backward compatibility, it is set to None in the HPO search spaces now 
      (it was already None in the default parameters).
    - removed debug print in TabM training loop
- v1.6.1:
    - For `n_ens>1`, changed the default behavior for classification to averaging probabilities instead of logits.
      This can be reverted by setting `ens_av_before_softmax=True`.
    - Implemented time limit for HPO/ensemble methods through `time_limit_s` parameter.
    - Support `torch>=2.6` and Python 3.13.
- v1.6.0:
    - Added support for other training losses in TabM through the `train_metric_name` parameter, 
      for example, (multi)quantile regression via `train_metric_name='multi_pinball(0.05,0.95)'`.
    - RealMLP-TD now adds the `n_ens` hyperparameter, which can be set to values >1 
      to train ensembles per train-validation split (called PackedEnsemble in the TabM paper). 
      This is especially useful when using holdout validation instead of cross-validation ensembles, 
      and to get more reliable validation predictions and scores for tuning/ensembling.
    - fixed RealMLP TabArena search space (`hpo_space_name='tabarena'`) for classification 
      (allow no label smoothing through `use_ls=False` instead of `use_ls="auto"`).
- v1.5.2: fixed more device bugs for HPO and ensembling
- v1.5.1: fixed a device bug in TabM for GPU
- v1.5.0:
    - added `n_repeats` parameter to scikit-learn interfaces for repeated cross-validation
    - HPO sklearn interfaces (the ones using random search)
      can now do weighted ensembling instead by setting `use_caruana_ensembling=True`.
      Removed the `RealMLP_Ensemble_Classifier` and `RealMLP_Ensemble_Regressor` from v1.4.2 
      since they are now redundant through this feature.
    - renamed `space` parameter of GBDT HPO interface 
      to `hpo_space_name` so now it also works with non-TPE versions.
    - Added new [TabArena](https://tabarena.ai) search spaces for boosted trees (not TPE), 
      which should be almost equivalent to the ones from TabArena 
      except for the early stopping logic. 
    - TabM now supports `val_metric_name` for early stopping on different metrics.
    - fixed issues #20 and #21 regarding HPO
    - small updates for the ["Rethinking Early Stopping" paper](https://arxiv.org/abs/2501.19195)
- v1.4.2:
    - fixed handling of custom `val_metric_name` HPO models and `Ensemble_TD_Regressor`.
    - if `tmp_folder` is specified in HPO models, 
      save each model to disk immediately instead of holding all of them in memory.
      This can considerably reduce RAM/VRAM usage.
      In this case, pickled HPO models will still rely on the models stored in the `tmp_folder`.
    - We now provide `RealMLP_Ensemble_Classifier` and `RealMLP_Ensemble_Regressor`,
      which will use weighted ensembling and usually perform better than HPO 
      (but have slower inference time). We recommend using the new `hpo_space_name='tabarena'`
      for best results.
- v1.4.1: 
    - moved dill to optional dependencies
    - updated TabM code to a newer version: 
      added option share_training_batches=False (old version: True), 
      exclude certain parameters from weight decay.
    - added [documentation](https://pytabkit.readthedocs.io/en/latest/bench/using_the_scheduler.html) for using the scheduler with custom jobs.
    - fixed bug in RealMLP refitting.
    - updated process start method for scheduler to speed up benchmarking
- v1.4.0:
    - moved some imports to the new `models` optional dependencies
      to have a more light-weight RealMLP installation
    - Added GPU support for CatBoost with help from 
      [Maximilian Schambach](https://github.com/MaxSchambach) 
      in #16 (not guaranteed to produce exactly the same results).
    - Ensembling now saves models after training if a path is supplied, to reduce memory usage
    - Added more search spaces
    - fixed error in multiquantile output when the passed y was one-dimensional 
      instead of having shape `(n_samples, 1)`
    - Added some examples to the documentation
- v1.3.0: 
    - Added multiquantile regression for RealMLP: 
      see the [documentation](https://pytabkit.readthedocs.io/en/latest/models/quantile_reg.html)
    - More hyperparameters for RealMLP
    - Added [TabICL](github.com/soda-inria/tabicl) wrapper
    - Small fixes
- v1.2.1: avoid error for older skorch versions
- v1.2.0:
    - Included post-hoc calibration and more metrics through 
      [probmetrics](https://github.com/dholzmueller/probmetrics).
    - Added benchmarking code for [Rethinking Early Stopping: Refine, Then Calibrate](https://arxiv.org/abs/2501.19195).
    - Updated format for saving predictions, 
      allow to stop on multiple metrics during the same training 
      in the benchmark.
    - Better categorical handling, 
      avoiding an error for string and object columns,
      not ignoring boolean columns by default but treating them as 
      categorical.
    - Added Ensemble_HPO_Classifier and Ensemble_HPO_Regressor.
- v1.1.3:
  - Fixed a bug where the categorical encoding was incorrect if categories 
    were missing in the training or validation set. The bug affected XGBoost 
    and potentially many other models except RealMLP.
  - Scikit-learn interfaces now accept and auto-detect categorical datatypes
    (category, string, object) in dataframes.
- v1.1.2:
    - Some compatibility improvements for scikit-learn 1.6
      (but disabled 1.6 since skorch is not compatible with it).
    - Improved documentation for Pytorch-Lightning interface.
    - Other small bugfixes and improvements.
- v1.1.1:
    - Added parameters `weight_decay`, `tfms`,
      and `gradient_clipping_norm` to TabM.
      The updated default parameters now apply the RTDL quantile transform.
- v1.1.0:
    - Included TabM
    - Replaced `__` by `_` in parameter names for MLP, MLP-PLR, ResNet, and FTT,
      to comply with scikit-learn interface requirements.
    - Fixed non-determinism in NN baselines
      by initializing the random state of quantile (and KDI)
      preprocessing transforms.
    - n_threads parameter is not ignored by NNs anymore.
    - Changes by [Lennart Purucker](https://github.com/LennartPurucker):
      Add time limit for RealMLP,
      add support for `lightning` (but also still allowing `pytorch-lightning`),
      making skorch a lazy import, removed msgpack\_numpy dependency.
- v1.0.0: Release for the NeurIPS version and arXiv v2+v3.
    - More baselines (MLP-PLR, FT-Transformer, TabR-HPO, RF-HPO),
      also some un-polished internal interfaces for other methods,
      esp. the ones in AutoGluon.
    - Updated benchmarking code (configurations, plots)
      including the new version of the Grinsztajn et al. benchmark
    - Updated fit() parameters in scikit-learn interfaces, etc.
- v0.0.1: First release for arXiv v1.
  Code and data are archived at [DaRUS](https://doi.org/10.18419/darus-4255).

