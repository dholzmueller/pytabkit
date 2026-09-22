# RealMLP-TD on MLX (Apple silicon)

Native MLX port of [pytabkit](https://github.com/dholzmueller/pytabkit)'s
**RealMLP-TD** — the strongest tuned-default model from
*Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data*
(NeurIPS 2024) — plus drop-in sklearn-style estimators.

Same pattern as the TabFM/LimiX MLX ports: faithful math, verifiable parity
against torch (including the real pytabkit layers), head-to-head MPS-vs-MLX
bench.

GBDTs (XGB/LGBM/CatBoost), TabM/TabR/HPO/ensemble/benchmark infra are
intentionally out of scope: tree libraries have no MLX backend and run on CPU
either way; the neural net is what benefits from MLX.

## Layout

| File | Purpose |
|---|---|
| `realmlp_mlx.py` | MLX arch: PBLD numeric embeddings, processed one-hot / learned-emb cats, front scale, 3× NTK-linear + he+5 bias + parametric SELU/Mish + dropout, head |
| `preprocessing.py` | numpy port of TD tfms: codes 0=missing; one-hot (missing→zeros, binary→1 col) then median-center → robust-scale → smooth-clip; big cats → emb-8 |
| `schedules.py` | `coslog4` (lr) / `flat_cos` (dropout, wd) / `constant`, exact ports |
| `train.py` | minibatch Adam (`drop_last`, per-param lr/wd factors, decoupled wd before step), label smoothing, best-epoch selection on class_error/rmse |
| `api.py` | `RealMLP_MLX_Classifier` / `RealMLP_MLX_Regressor` (sklearn API, pandas cat auto-detect, RandomSplitter-exact val split, target standardize+clamp) |
| `rng.py` | seed-compatible RNG backends (torch/numpy) for init + shuffle |
| `convert.py` | npz save/load, torch-mirror weight copy (`to_torch`/`from_torch`) |
| `parity.py` | component + forward + Adam-step parity vs torch mirror |
| `parity_real.py` | EXACT parity vs real pytabkit layers (one-hot, median/robust/clip chain, Metrics, CE/MSE) via stub import (no lightning needed) |
| `exact.py` | exact-replay training (real init dump + torch-global-RNG-synced shuffle/masks) |
| `extract_reference.py` | drive the real fitter chain: dump init values + RNG snapshot + probe |
| `exact_compare.py` / `step_compare.py` | replay-vs-real verification (curves + params) |
| `train_parity.py` | 40-epoch MLX-vs-torch training-curve parity (same init/batches) |
| `real_compare.py` | real pytabkit (lightning/CPU) vs port on identical train/val sets |
| `e2e.py` | fit/predict smoke tests (breast cancer, categoricals, diabetes) |
| `bench_mps_vs_mlx.py` | torch/MPS vs MLX training-step timing |

## Results (M4, this repo)

- `parity_real` vs real pytabkit code: one-hot, one-hot+median+robust+clip,
  class_error, rmse, mse, soft-target CE — all max|diff| = **0.0**
- Forward vs torch mirror: 1.9e-06; schedules exact; Adam step ≤2.4e-07
- 40-epoch training curves (same init/batches): max diff 7e-04 RMSE
- **Exact replay**: same `random_state` with `rng_backend='exact'` replicates
  the reference global RNG stream (init draws in fitter order, one
  randperm/epoch, dropout masks via the same aten op) — 8-epoch runs give
  **identical predictions** (diabetes RMSE 65.87 = 65.87, cancer accuracy
  0.9735 = 0.9735) with no weight dump. Per-epoch val curves match to 4
  decimals, best-restored params to 9e-05. This hunt found 3 real bugs:
  train rows must be in sorted `argwhere` order (he+5 samples positionally),
  eval val is clamped raw RMSE, and the wd step double-applies lr/wd
  factors (replicated).
- vs REAL pytabkit, identical train/val but different RNG streams:
  diabetes 52.8 vs 56.0 (pred corr 0.94), cancer 0.938–0.982;
  at 256ep both reach reference level (0.97–0.99 / ~52–58).
- E2E (256 epochs): breast cancer acc **0.991**, diabetes RMSE **52.1**
- Bench (20k×30 + cats, full 32-epoch fits, median of 2): **MLX 5.0s vs MPS 11.5s → 2.3x**

## Usage

```python
from pytabkit_mlx.api import RealMLP_MLX_Classifier, RealMLP_MLX_Regressor

clf = RealMLP_MLX_Classifier()   # TD defaults: lr=0.04, wd=0.02, selu, drop=0.15
clf.fit(X_train, y_train)        # pandas auto-detects categorical columns
clf.predict_proba(X_test)

reg = RealMLP_MLX_Regressor()    # TD defaults: lr=0.2, wd=0.02, mish, drop=0.15
reg.fit(X_train, y_train)
reg.predict(X_test)
```

## TD defaults ported

- Both: `hidden=[256]*3`, PBLD (`sigma=0.1, h1=16, h2=4, densenet, cos-bias`,
  lr×0.1), front scale (lr×6), NTK weights + `std` init, `he+5` biases
  (lr×0.1, no wd), parametric act (lr×0.1), Adam (0.9, 0.95),
  `coslog4` lr, `flat_cos` wd + dropout, 256 epochs × batch 256 `drop_last`.
- Class: selu, lr 0.04, wd 0.02, dropout 0.15, label smoothing 0.1,
  best epoch on class_error.
- Reg: mish, lr 0.2, wd 0.02, dropout 0.15, MSE on standardized targets,
  predictions clamped to train [min, max], best epoch on rmse.
- Input: codes 0=missing/unknown; cats with cat_size ≤ 9 one-hot-encoded
  (missing column dropped, binary → single [0,+1,-1] column) then
  median/robust/clip-normalized; larger cats → emb-8; val = first
  ceil(20%) of seeded randperm (unstratified, like RandomSplitter).

## Notes

- Float32 throughout (matches torch reference).
- Same seed, same draws: `rng_backend='torch'` (default when torch is
  importable) draws init weights and batch order from a dedicated
  torch.Generator — bit-identical to torch itself (tested 0.0 diff).
  `'numpy'` uses `np.random.default_rng`. `'exact'` replicates the
  reference *global* stream from the seed alone (see above).
  Dropout masks in default mode come from MLX's own RNG.
- Missing numericals must be imputed beforehand, as in pytabkit.
- Single train/val split, no CV ensembling/HPO/calibration.
- RNG streams (init/shuffle/dropout) intentionally differ from torch;
  parity is exact on all deterministic math, statistical on training.
