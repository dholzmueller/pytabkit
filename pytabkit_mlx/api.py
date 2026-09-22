"""Scikit-learn style interfaces for RealMLP-TD on MLX.

Matches pytabkit's RealMLP_TD_Classifier/Regressor behavior:
  - pandas auto-detection of categorical columns (object/category/string/bool)
  - missing numerical values are rejected (impute beforehand, as in pytabkit)
  - single train/validation split (first ceil((1-val_fraction)*n) of a
    seeded randperm, exactly like RandomSplitter; NOT stratified),
    best-epoch selection on class_error / rmse
  - regression targets standardized; predictions clamped to train [min, max]
"""

import math

import numpy as np

import mlx.core as mx
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from . import realmlp_mlx as R
from .preprocessing import TabPreprocessor, fit_target_stats
from .rng import make_rng
from .train import fit_model


def _resolve_seed(random_state):
    """Coherent seed for ALL streams (split, init/shuffle backend, MLX dropout).

    sklearn convention: ``None`` means nondeterministic. Derive one fresh
    entropy seed and use it everywhere, so random_state=None is fully
    nondeterministic (instead of deterministic split/dropout mixed with
    nondeterministic init/shuffle, as before).
    """
    if random_state is None:
        import secrets
        return secrets.randbits(31)
    return random_state


def pytabkit_split(n, seed, first_fraction=0.8):
    """Replicates the reference train/val split with explicit val indices.

    sklearn_base with val_idxs: train = complement of val IN SORTED ORDER
    (torch.argwhere), val = as given. Uses torch's dedicated-generator
    randperm (exact) when torch is available, else a numpy permutation.
    NOTE: the numpy fallback is deterministic but yields a DIFFERENT split
    than torch for the same seed (different RNG algorithms), so results
    depend on whether torch is installed. Splits are NOT stratified.
    """
    try:
        import torch
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(n, generator=g).numpy()
    except ImportError:
        perm = np.random.RandomState(seed).permutation(n)
    k = int(math.ceil(first_fraction * n))
    return np.sort(perm[:k]), perm[k:]


def _detect_columns(X):
    """Split DataFrame columns into (numerical, categorical, dropped).

    Mirrors ToDictDatasetConverter: numerical = number dtypes (plain bool
    excluded), categorical = string/object/category/boolean. Anything else
    (plain bool, datetime, ...) is unused, with a warning like the reference.
    """
    import pandas as pd
    import pandas.api.types as T
    nums, cats, dropped = [], [], []
    for c in X.columns:
        dt = X[c].dtype
        if T.is_bool_dtype(dt):
            cats.append(c)
        elif T.is_numeric_dtype(dt):
            nums.append(c)
        elif (T.is_object_dtype(dt) or T.is_string_dtype(dt)
                or isinstance(dt, pd.CategoricalDtype)):
            cats.append(c)
        else:
            dropped.append(c)
    return nums, cats, dropped


def _resolve_columns(X, cat_features):
    """Returns (num_selector, cat_selector). For DataFrames: names; else indices."""
    import pandas as pd
    if isinstance(X, pd.DataFrame):
        cols = list(X.columns)
        if cat_features == 'auto' or cat_features is None:
            nums, cats, dropped = _detect_columns(X)
            if dropped:
                import warnings
                warnings.warn(f'Columns unused due to their data type: {dropped}')
        else:
            cats = [cols[i] if isinstance(i, int) else i for i in cat_features]
            nums = [c for c in cols if c not in cats]
        return nums, cats
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    idx = [] if not cat_features or cat_features == 'auto' else list(cat_features)
    nums = [i for i in range(X.shape[1]) if i not in idx]
    return nums, idx


def _select(X, nums, cats):
    import pandas as pd
    if isinstance(X, pd.DataFrame):
        Xn = (X[nums].to_numpy(dtype=np.float32, copy=True) if nums
              else np.zeros((len(X), 0), dtype=np.float32))
        cat_cols = [X[c].to_numpy() for c in cats]
    else:
        Xa = np.asarray(X)
        if Xa.ndim == 1:
            Xa = Xa[:, None]
        Xn = (Xa[:, nums].astype(np.float32) if nums
              else np.zeros((Xa.shape[0], 0), np.float32))
        cat_cols = [Xa[:, i] for i in cats]
    return Xn, cat_cols


class _RealMLP_MLX_Base(BaseEstimator):
    def __init__(self, hidden_sizes=(256, 256, 256), n_epochs=256, batch_size=256,
                 lr=None, wd=None, act=None, p_drop=None, use_ls=True, ls_eps=0.1,
                 val_fraction=0.2, random_state=0, verbosity=0,
                 max_one_hot_cat_size=9, embedding_size=8, rng_backend='auto'):
        self.hidden_sizes = hidden_sizes
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.wd = wd
        self.act = act
        self.p_drop = p_drop
        self.use_ls = use_ls
        self.ls_eps = ls_eps
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.verbosity = verbosity
        self.max_one_hot_cat_size = max_one_hot_cat_size
        self.embedding_size = embedding_size
        self.rng_backend = rng_backend

    def _split(self, n, seed):
        return pytabkit_split(n, seed, 1.0 - self.val_fraction)

    def _build(self, rng, prep, tr, n_out, act_name):
        Xn, Xo, Xc = tr
        P = R.init_params(Xn.shape[1], Xo.shape[1], prep.emb_cat_sizes_, n_out,
                          list(self.hidden_sizes), act_name, rng)
        P['_n_out'] = n_out
        # model-input matrix for data-dependent init (subsampled for RAM).
        # No draw when using all rows (the reference draws nothing here).
        m = min(Xn.shape[0], 8192)
        sel = rng.permutation(Xn.shape[0])[:m] if m < Xn.shape[0] \
            else np.arange(Xn.shape[0])
        S_parts = [np.array(R.pbld_forward(
            P, mx.array(Xn[sel])).tolist(), dtype=np.float64).reshape(m, -1)]
        mx.eval(S_parts)
        if Xo.shape[1] > 0:
            S_parts.append(Xo[sel].astype(np.float64))
        for i in range(P['n_emb_tables']):
            S_parts.append(np.array(
                P[f'emb_table_{i}'][mx.array(Xc[sel][:, i])].tolist(),
                dtype=np.float64).reshape(m, -1))
        R.init_data_dependent(P, np.concatenate(S_parts, axis=1), rng)
        return P

    def _forward(self, X):
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            if set(X.columns) != set(self.feature_names_in_):
                raise ValueError(
                    f'Different columns during fit() and predict(): '
                    f'{self.feature_names_in_} and {list(X.columns)}')
        Xn_all, cat_cols = _select(X, self.num_cols_, self.cat_cols_)
        Xn, Xo, Xc = self.prep_.transform(Xn_all, cat_cols)
        out = R.forward(self.model_, mx.array(Xn), mx.array(Xo), mx.array(Xc))
        mx.eval(out)
        return np.array(out.tolist(), dtype=np.float32)


class RealMLP_MLX_Classifier(_RealMLP_MLX_Base, ClassifierMixin):
    """RealMLP-TD classifier on MLX. Defaults: lr=0.04, wd=0.02,
    act='selu', p_drop=0.15, label smoothing 0.1, val on class_error."""

    def fit(self, X, y, cat_features='auto'):
        seed = _resolve_seed(self.random_state)
        rng = make_rng(seed, self.rng_backend)
        if self.rng_backend == 'exact':
            import torch
            from .rng import combine_seeds, sub_split_seed
            torch.manual_seed(combine_seeds(sub_split_seed(seed), 0))
        y = np.asarray(y)
        classes, y_idx = np.unique(y, return_inverse=True)
        self.classes_ = classes
        y_idx = y_idx.astype(np.int64)
        self.num_cols_, self.cat_cols_ = _resolve_columns(X, cat_features)
        import pandas as pd
        self.feature_names_in_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        self.n_features_in_ = len(self.num_cols_) + len(self.cat_cols_)
        Xn_all, cat_cols = _select(X, self.num_cols_, self.cat_cols_)
        if Xn_all.shape[1] > 0 and not np.isfinite(Xn_all).all():
            raise ValueError('Missing numerical values must be imputed beforehand (as in pytabkit)')
        idx_tr, idx_va = self._split(len(y), seed)
        prep = TabPreprocessor(self.max_one_hot_cat_size, self.embedding_size)
        prep.fit(Xn_all[idx_tr], [c[idx_tr] for c in cat_cols])
        self.prep_ = prep
        tr = prep.transform(Xn_all[idx_tr], [c[idx_tr] for c in cat_cols])
        va = prep.transform(Xn_all[idx_va], [c[idx_va] for c in cat_cols])
        mx.random.seed(seed)
        P = self._build(rng, prep, tr, len(classes), self.act or 'selu')
        train = {'Xn': tr[0], 'Xo': tr[1], 'Xc': tr[2], 'y': y_idx[idx_tr]}
        val = {'Xn': va[0], 'Xo': va[1], 'Xc': va[2], 'y': y_idx[idx_va]}
        cfg = dict(lr=self.lr if self.lr is not None else 4e-2,
                   wd=self.wd if self.wd is not None else 2e-2,
                   n_epochs=self.n_epochs, batch_size=self.batch_size,
                   lr_sched='coslog4', wd_sched='flat_cos',
                   p_drop=self.p_drop if self.p_drop is not None else 0.15,
                   p_drop_sched='flat_cos',
                   ls_eps=self.ls_eps if self.use_ls else 0.0,
                   task='class', n_classes=len(classes),
                   exact_masks=(self.rng_backend == 'exact'))
        fit_model(P, train, val, cfg, rng, self.verbosity)
        self.model_ = P
        return self

    def predict_proba(self, X):
        out = self._forward(X)
        e = np.exp(out - out.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self._forward(X), axis=1)]


class RealMLP_MLX_Regressor(_RealMLP_MLX_Base, RegressorMixin):
    """RealMLP-TD regressor on MLX. Defaults: lr=0.2, wd=0.02,
    act='mish', p_drop=0.15, val on rmse, standardized + clamped target."""

    def fit(self, X, y, cat_features='auto'):
        seed = _resolve_seed(self.random_state)
        rng = make_rng(seed, self.rng_backend)
        if self.rng_backend == 'exact':
            import torch
            from .rng import combine_seeds, sub_split_seed
            torch.manual_seed(combine_seeds(sub_split_seed(seed), 0))
        y = np.asarray(y, dtype=np.float32).ravel()
        self.num_cols_, self.cat_cols_ = _resolve_columns(X, cat_features)
        import pandas as pd
        self.feature_names_in_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        self.n_features_in_ = len(self.num_cols_) + len(self.cat_cols_)
        Xn_all, cat_cols = _select(X, self.num_cols_, self.cat_cols_)
        if Xn_all.shape[1] > 0 and not np.isfinite(Xn_all).all():
            raise ValueError('Missing numerical values must be imputed beforehand (as in pytabkit)')
        idx_tr, idx_va = self._split(len(y), seed)
        self.tgt_ = fit_target_stats(y[idx_tr])
        std = self.tgt_['std'] if self.tgt_['std'] > 0 else 1.0
        prep = TabPreprocessor(self.max_one_hot_cat_size, self.embedding_size)
        prep.fit(Xn_all[idx_tr], [c[idx_tr] for c in cat_cols])
        self.prep_ = prep
        tr = prep.transform(Xn_all[idx_tr], [c[idx_tr] for c in cat_cols])
        va = prep.transform(Xn_all[idx_va], [c[idx_va] for c in cat_cols])
        mx.random.seed(seed)
        P = self._build(rng, prep, tr, 1, self.act or 'mish')
        train = {'Xn': tr[0], 'Xo': tr[1], 'Xc': tr[2],
                 'y': ((y[idx_tr] - self.tgt_['mean']) / std).astype(np.float32)}
        val = {'Xn': va[0], 'Xo': va[1], 'Xc': va[2],
               'y': ((y[idx_va] - self.tgt_['mean']) / std).astype(np.float32),
               'y_raw': y[idx_va].astype(np.float64)}
        cfg = dict(lr=self.lr if self.lr is not None else 0.2,
                   wd=self.wd if self.wd is not None else 2e-2,
                   n_epochs=self.n_epochs, batch_size=self.batch_size,
                   lr_sched='coslog4', wd_sched='flat_cos',
                   p_drop=self.p_drop if self.p_drop is not None else 0.15,
                   p_drop_sched='flat_cos', ls_eps=0.0,
                   task='reg', n_classes=0, y_mean=self.tgt_['mean'],
                   y_std=std, y_min=self.tgt_['min'], y_max=self.tgt_['max'],
                   exact_masks=(self.rng_backend == 'exact'))
        fit_model(P, train, val, cfg, rng, self.verbosity)
        self.model_ = P
        return self

    def predict(self, X):
        out = self._forward(X)[:, 0]
        std = self.tgt_['std'] if self.tgt_['std'] > 0 else 1.0
        out = out * std + self.tgt_['mean']
        return np.clip(out, self.tgt_['min'], self.tgt_['max'])
