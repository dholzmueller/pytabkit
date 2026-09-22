"""Tabular preprocessing matching RealMLP-TD's tfms:
['one_hot', 'median_center', 'robust_scale', 'smooth_clip', 'embedding'].

Pipeline (reference: NNFactory + PreprocessingFactory + categorical.py):
  codes: 0 = missing/unknown, 1..K = seen categories (OrdinalEncoder + 1).
  numericals: median-center -> robust-scale (IQR + fallbacks) -> smooth-clip.
  small cats (cat_size <= max_one_hot_cat_size, default 9):
      SingleOneHot with use_missing_zero=True, use_1d_binary_onehot=True:
        - cat_size == 2: single column, codes -> [-1, +1]
        - cat_size == 3: single column, codes -> [0, +1, -1]
        - else: full one-hot, column 0 (missing) DROPPED -> missing = zeros
      the resulting columns go through median-center -> robust-scale ->
      smooth-clip (statistics fitted on the TRAIN one-hot matrix).
  big cats: integer codes -> learned embedding (dim 8), no preprocessing.

Matches pipeline.py (MedianCenterFactory, RobustScaleFactory), models.py
smooth_clip_func, categorical.py (SingleOneHotLayer, EncodingLayer routing)
and data/conversion.py (OrdinalEncoder + 1 code convention).
"""

import numpy as np


SMOOTH_CLIP_MAX = 3.0


def smooth_clip(x):
    return x / np.sqrt(1.0 + (x / SMOOTH_CLIP_MAX) ** 2)


def fit_num_stats(X):
    """Per-feature median + IQR scale factors. X: (n, f) float array."""
    X = np.asarray(X, dtype=np.float64)
    median = np.median(X, axis=0)
    q75 = np.quantile(X, 0.75, axis=0)
    q25 = np.quantile(X, 0.25, axis=0)
    iqr = q75 - q25
    mx = X.max(axis=0)
    mn = X.min(axis=0)
    iqr = np.where(iqr == 0.0, 0.5 * (mx - mn), iqr)
    factors = 1.0 / (iqr + 1e-30)
    factors[iqr == 0.0] = 0.0
    return {'median': median.astype(np.float32),
            'scale': factors.astype(np.float32)}


def transform_num(X, stats):
    X = np.asarray(X, dtype=np.float32)
    X = (X - stats['median'][None, :]) * stats['scale'][None, :]
    return smooth_clip(X).astype(np.float32)


def _is_missing(v):
    if v is None:
        return True
    try:
        import pandas as pd
        if v is pd.NA or v is pd.NaT:
            return True
    except ImportError:
        pass
    return isinstance(v, float) and np.isnan(v)


def factorize_column(col, sort=True):
    seen = []
    seen_set = set()
    for v in col:
        if _is_missing(v):
            continue
        if v not in seen_set:
            seen_set.add(v)
            seen.append(v)
    if sort:
        try:
            seen = sorted(seen)
        except TypeError:
            seen = sorted(seen, key=repr)
    return seen


class CatFeature:
    """Factorization + one-hot/embedding routing for one categorical column.

    Codes: 0 = missing/unknown, 1..K = seen (matches OrdinalEncoder + 1).
    """

    def __init__(self, categories, max_one_hot_cat_size=9, embedding_size=8):
        self.categories = list(categories)
        self.cat_size = len(categories) + 1  # +1 for missing/unknown (code 0)
        self.use_one_hot = self.cat_size <= max_one_hot_cat_size
        self.embedding_size = embedding_size
        self._index = {c: i + 1 for i, c in enumerate(self.categories)}

    @property
    def out_dim(self):
        """Number of (pre-preprocessing) one-hot output columns."""
        if not self.use_one_hot:
            return 0
        if self.cat_size in (2, 3):
            return 1  # use_1d_binary_onehot
        return self.cat_size - 1  # use_missing_zero: drop column 0

    def encode(self, col):
        return np.array([self._index.get(v, 0) if not _is_missing(v) else 0
                         for v in col], dtype=np.int64)

    def one_hot(self, codes):
        n = codes.shape[0]
        if self.cat_size == 2:
            return np.where(codes[:, None] == 0, -1.0, 1.0).astype(np.float32)
        if self.cat_size == 3:
            return np.array([0.0, 1.0, -1.0], dtype=np.float32)[codes][:, None]
        full = np.zeros((n, self.cat_size), dtype=np.float32)
        full[np.arange(n), codes] = 1.0
        return full[:, 1:]  # use_missing_zero: drop the missing column


class TabPreprocessor:
    """Fit on train, transform train/val/test. Handles num + cat split."""

    def __init__(self, max_one_hot_cat_size=9, embedding_size=8):
        self.max_one_hot_cat_size = max_one_hot_cat_size
        self.embedding_size = embedding_size
        self.num_stats_ = None
        self.oh_stats_ = None  # median/scale/clip stats for one-hot columns
        self.cat_features_ = []
        self.n_num_ = 0
        self.emb_cat_sizes_ = []

    def _raw_one_hot(self, cat_columns):
        parts = []
        for cf, col in zip(self.cat_features_, cat_columns):
            if cf.use_one_hot:
                parts.append(cf.one_hot(cf.encode(col)))
        return parts

    def fit(self, X_num, cat_columns):
        """X_num: (n, f) float32 array (may have f == 0).
        cat_columns: list of 1-D arrays (raw values)."""
        X_num = np.asarray(X_num, dtype=np.float32)
        self.n_num_ = X_num.shape[1] if X_num.ndim == 2 else 0
        if self.n_num_ > 0:
            self.num_stats_ = fit_num_stats(X_num)
        self.cat_features_ = []
        self.emb_cat_sizes_ = []
        for col in cat_columns:
            cf = CatFeature(factorize_column(col),
                            self.max_one_hot_cat_size, self.embedding_size)
            self.cat_features_.append(cf)
            if not cf.use_one_hot:
                self.emb_cat_sizes_.append(cf.cat_size)
        oh = self._raw_one_hot(cat_columns)
        if oh:
            self.oh_stats_ = fit_num_stats(
                np.concatenate(oh, axis=1).astype(np.float32))
        else:
            self.oh_stats_ = None
        return self

    def transform(self, X_num, cat_columns):
        """Returns (X_num_proc float32, X_onehot_proc float32, X_emb_codes int64)."""
        n = self._n_rows(X_num, cat_columns)
        if self.n_num_ > 0:
            Xn = transform_num(np.asarray(X_num, dtype=np.float32), self.num_stats_)
        else:
            Xn = np.zeros((n, 0), dtype=np.float32)
        oh = self._raw_one_hot(cat_columns)
        Xo = transform_num(np.concatenate(oh, axis=1).astype(np.float32),
                           self.oh_stats_) if oh \
            else np.zeros((n, 0), dtype=np.float32)
        code_parts = []
        for cf, col in zip(self.cat_features_, cat_columns):
            if not cf.use_one_hot:
                code_parts.append(cf.encode(col)[:, None])
        Xc = np.concatenate(code_parts, axis=1).astype(np.int64) if code_parts \
            else np.zeros((n, 0), dtype=np.int64)
        return Xn, Xo, Xc

    def _n_rows(self, X_num, cat_columns):
        if self.n_num_ > 0:
            return np.asarray(X_num).shape[0]
        if cat_columns:
            return len(cat_columns[0])
        raise ValueError('No features')

    @property
    def n_onehot_(self):
        return sum(cf.out_dim for cf in self.cat_features_ if cf.use_one_hot)


def fit_target_stats(y):
    y = np.asarray(y, dtype=np.float32)
    return {'mean': float(y.mean()), 'std': float(y.std()),
            'min': float(y.min()), 'max': float(y.max())}
