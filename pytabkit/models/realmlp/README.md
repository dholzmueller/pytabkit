# Standalone RealMLP

**Experimental:** this implementation is half vibe-coded. It has been confirmed experimentally to yield similarly good
results as the original implementation, but it is less battle-tested.

A compact re-implementation of RealMLP in plain PyTorch.
It does not depend on Lightning or on the fitter/factory framework in `pytabkit.models.nn_models`.
It trains all models (train/test splits × train/val splits × ensemble members) as a single vectorized model,
like the original implementation.

| File | Content |
|---|---|
| `model.py` | `RealMLPConfig` (all hyperparameters) and the `RealMLP` module, written out explicitly |
| `train.py` | preprocessing statistics, data loading, training loop, and the `fit()` / `predict()` functions |
| `alg_interface.py` | adapter to pytabkit: `StandaloneRealMLPAlgInterface` and an HPO sampler |

`model.py` and `train.py` only depend on PyTorch and NumPy.
For benchmarking, use `StandaloneRealMLPInterfaceWrapper` from `pytabkit.bench.alg_wrappers.interface_wrappers`.

## Usage

```python
from pytabkit.models.realmlp import RealMLPConfig, fit, predict

# x_num: (n, n_num) float with NaN for missing values; x_cat: (n, n_cat) long with 0 for missing values
# one row of train_idxs / val_idxs per split, all splits are trained in parallel
model, info = fit(RealMLPConfig(n_ens=4), x_num, x_cat, cat_sizes, y,
                  train_idxs=train_idxs[None], val_idxs=val_idxs[None], n_classes=n_classes)
# info: best epoch per split (used for refitting with stop_epochs=...)
logits = predict(model, x_num_test, x_cat_test)[0]  # regression: predictions on the original scale

# through the pytabkit interface (RealMLPConfig fields as keyword arguments)
alg = StandaloneRealMLPAlgInterface(act='mish', lr=0.2)  # regression TD settings (= RealMLPConfig.td_reg())
```

## What is the same as in the original

With the same hyperparameters, the following match the original RealMLP-TD (`NNAlgInterface`):
- architecture and its parameterization
- weight, bias, PBLD and embedding initialization
- preprocessing, including one-hot encoding with category 0 as "missing"
- preprocessing statistics computed separately on the training set of each split
- output standardization and clamping for regression
- schedules, optimizer and per-parameter learning rate and weight decay factors
- dropout and label smoothing
- ensembling
- best-epoch selection and early stopping per split
- refitting at the best mean epoch
- the `fit_params` format

Results are not bitwise identical, though, because random numbers are drawn in a different order.

## Differences to the original

**Defaults.** The `RealMLPConfig()` defaults follow the TabArena default settings:
- no label smoothing (`use_ls=False`)
- best-epoch selection on `cross_entropy` instead of `class_error` (classification)

The paper defaults are available as `RealMLPConfig.td_class()` and `RealMLPConfig.td_reg()`.

**Initialization.** The data-dependent initialization (weight "std" init, he+5 bias init) has two differences:
- It uses the first (shuffled) training batch. The original uses the whole training set, subsampled if RAM is limited.
- It uses a single random seed for all vectorized models. The original seeds each model separately.

**Weight decay factors.** The original multiplies the lr and wd factors of a parameter into its weight decay twice.
In effect, the decay is `wd * lr * (wd_factor * lr_factor)²`.
Here, they enter once: `wd * lr * wd_factor * lr_factor`.
The defaults of `scale_wd_factor` (6), `plr_wd_factor` (0.1) and `act_wd_factor` (0.1) are chosen so that the TD settings
give exactly the same decay.

There is also a new `first_layer_wd_factor`. It defaults to `first_layer_lr_factor`, which again reproduces the original.

**HPO search spaces.** `sample_realmlp_params` supports `'tabarena'` and `'tabarena-new'`
(which adds 8-member ensembles and tuning of the categorical encoding). Compared to the original spaces, both also tune:
- `plr_wd_factor`
- `scale_wd_factor`
- `first_layer_wd_factor`

In the original, these are tied to the corresponding lr factors. A given seed also gives a different configuration
than in the original.

**Missing values.** NaNs in numerical features are supported; the original raises an error.
- Preprocessing statistics ignore NaNs.
- Missing values get a learned embedding per feature for the PBLD part, and a learned replacement value per feature
  for the raw feature value (the PBLD "densenet" part, or the feature itself with `num_emb_type='none'`).
- Both are initialized to zero.

**Batch size.** `batch_size='auto'` uses the number of training samples.
Before v1.8.0, the original used the number of vectorized splits instead, which was a bug.

**Reduced scope.** Only options relevant for RealMLP-TD and the TabArena search spaces are supported.
- The architecture is fixed to: front scale, then (linear, bias, activation, dropout) per hidden layer,
  then (linear, bias).
  Configurable: layer sizes, activation, parametric activation, front scale.
  The original's `block_str` is not supported.
- `num_emb_type`: `'pbld'` or `'none'`.
- NTK parameterization with "std" weight init and "he+5" bias init.
- Validation metrics: `cross_entropy`, `class_error`, `rmse`.
- Refitting always stops at the best *mean* epoch (`use_best_mean_epoch_for_refit=True`).

Not available (among others):
- other preprocessing options (`tfms`), parameterizations and initializations
- other optimizers and train metrics, such as quantile regression
- feature importances
- an sklearn interface
