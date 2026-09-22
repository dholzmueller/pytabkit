"""Exact parity against REAL pytabkit layers (no lightning needed).

The top-level pytabkit/__init__ imports lightning-based modules, so we stub
the bare `pytabkit` package (its models/__init__ is empty) and import the
pure-torch submodules directly. Compares, on identical inputs:
  - SingleOneHotLayer (use_missing_zero + 1d-binary) vs CatFeature.one_hot
  - MedianCenter/RobustScale/smooth-clip on one-hot cols vs TabPreprocessor
  - full categorical input block vs the chained real layers
  - Metrics.apply class_error/rmse + cross_entropy/mse vs train.py

Run: ./.venv/bin/python -m pytabkit_mlx.parity_real
"""
import sys
from pathlib import Path
import types

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- stub the `pytabkit` package to skip its lightning-pulling __init__ ---
_pkg = types.ModuleType('pytabkit')
_pkg.__path__ = [str(Path(__file__).resolve().parents[1] / 'pytabkit')]
sys.modules['pytabkit'] = _pkg

from pytabkit.models.data.data import TensorInfo, DictDataset
from pytabkit.models.nn_models.categorical import SingleOneHotFitter
from pytabkit.models.nn_models.pipeline import MedianCenterFactory, RobustScaleFactory
from pytabkit.models.nn_models.base import FunctionFitter
from pytabkit.models.nn_models.models import smooth_clip_func
from pytabkit.models.training.metrics import Metrics, cross_entropy, mse

from pytabkit_mlx import preprocessing as PP
from pytabkit_mlx.train import _ce_loss, _class_error, _rmse

rng = np.random.default_rng(0)


def check(name, a, b, tol):
    d = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    status = 'OK ' if d <= tol else 'FAIL'
    print(f'[{status}] {name}: max|diff|={d:.3e} (tol {tol:.0e})')
    assert d <= tol, name


def _ds(x_cat, cat_sizes):
    return DictDataset(
        tensors={'x_cat': torch.as_tensor(x_cat, dtype=torch.long)},
        tensor_infos={'x_cat': TensorInfo(cat_sizes=list(cat_sizes))},
        device='cpu', n_samples=x_cat.shape[0])


def test_one_hot_layer():
    # features: binary+missing (3), 4 cats (5), degenerate (2); 12 -> embedding
    sizes = [3, 5, 2]
    for cs in sizes:
        codes = rng.integers(0, cs, size=400).astype(np.int64)
        ds = _ds(codes[:, None], [cs])
        fitter = SingleOneHotFitter(use_missing_zero=True, bin_onoff=(1.0, 0.0),
                                    multi_onoff=(1.0, 0.0), use_1d_binary_onehot=True)
        out_ti = fitter.forward_tensor_infos(ds.tensor_infos)
        layer = fitter._fit(ds)
        ref = layer.forward_tensors({'x_cat': torch.as_tensor(codes[:, None])})['x_cont'].numpy()
        assert ref.shape[1] == out_ti['x_cont'].get_feat_shape()[0]
        n_seen = cs - 1
        cf = PP.CatFeature([f'v{i}' for i in range(n_seen)])
        assert cf.cat_size == cs, (cf.cat_size, cs)
        assert cf.out_dim == ref.shape[1], (cf.out_dim, ref.shape[1], cs)
        mine = cf.one_hot(codes)
        check(f'one_hot cat_size={cs}', mine, ref, 1e-6)
    # big cats route to embeddings, not one-hot
    cf_big = PP.CatFeature([f'v{i}' for i in range(11)])
    assert cf_big.cat_size == 12 and not cf_big.use_one_hot and cf_big.out_dim == 0


def test_one_hot_pipeline():
    # raw values incl. missing/unknown; full TabPreprocessor Xo vs real chain
    cats_small = ['a', 'b']                      # cat_size 3 -> [0,1,-1]
    cats_mid = ['a', 'b', 'c', 'd', 'e', 'f']    # cat_size 7 -> 6 cols
    n = 300
    col1 = rng.choice(cats_small + [None], size=n, p=[0.3, 0.3, 0.4]).tolist()
    col2 = rng.choice(cats_mid + [None], size=n).tolist()
    col3 = rng.choice([f'k{i}' for i in range(20)], size=n).tolist()  # emb path
    prep = PP.TabPreprocessor()
    prep.fit(np.zeros((n, 0), np.float32), [col1, col2, col3])
    _, Xo, Xc = prep.transform(np.zeros((n, 0), np.float32), [col1, col2, col3])
    assert Xc.shape == (n, 1) and (Xc[:, 0] == 0).sum() == 0  # all seen here
    # unseen -> code 0
    _, _, Xc2 = prep.transform(np.zeros((2, 0), np.float32),
                               [['a', 'b'], ['a', 'b'], ['nope', 'k1']])
    assert Xc2[0, 0] == 0, Xc2

    # real chain on codes 1..K / 0
    code1 = prep.cat_features_[0].encode(col1)[:, None]
    code2 = prep.cat_features_[1].encode(col2)[:, None]
    codes = np.concatenate([code1, code2], axis=1)
    ds = _ds(codes, [3, 7])
    f1 = SingleOneHotFitter(True, (1.0, 0.0), (1.0, 0.0), True)
    l1 = f1._fit(DictDataset({'x_cat': torch.as_tensor(code1)}, {'x_cat': TensorInfo(cat_sizes=[3])}, 'cpu', n))
    f2 = SingleOneHotFitter(True, (1.0, 0.0), (1.0, 0.0), True)
    l2 = f2._fit(DictDataset({'x_cat': torch.as_tensor(code2)}, {'x_cat': TensorInfo(cat_sizes=[7])}, 'cpu', n))
    oh = torch.cat([l1.forward_tensors({'x_cat': torch.as_tensor(code1)})['x_cont'],
                    l2.forward_tensors({'x_cat': torch.as_tensor(code2)})['x_cont']], dim=-1)
    ds2 = DictDataset({'x_cont': oh}, {'x_cont': TensorInfo(feat_shape=[oh.shape[1]])}, 'cpu', n)
    med = MedianCenterFactory()._fit(ds2)
    x = med.forward_cont(oh)
    # robust scale fitted on median-centered data (sequential fitting)
    ds3 = DictDataset({'x_cont': x}, {'x_cont': TensorInfo(feat_shape=[x.shape[1]])}, 'cpu', n)
    scl = RobustScaleFactory()._fit(ds3)
    x = scl.forward_cont(x)
    x = smooth_clip_func(x)
    check('one_hot+median+robust+clip', Xo, x.numpy(), 1e-5)


def test_metrics_and_losses():
    r = np.random.default_rng(2)
    logits = r.normal(size=(64, 3)).astype(np.float32)
    y = r.integers(0, 3, 64)
    assert abs(Metrics.apply(torch.as_tensor(logits), torch.as_tensor(y), 'class_error').item()
               - float((logits.argmax(1) != y).mean())) < 1e-9
    check('class_error', _class_error(mx.array(logits), mx.array(y)),
          Metrics.apply(torch.as_tensor(logits), torch.as_tensor(y), 'class_error').item(), 1e-6)
    pred = r.normal(size=(64, 1)).astype(np.float32)
    yt = r.normal(size=(64, 1)).astype(np.float32)
    check('rmse', _rmse(mx.array(pred), mx.array(yt[:, 0])),
          Metrics.apply(torch.as_tensor(pred), torch.as_tensor(yt), 'rmse').item(), 1e-6)
    check('mse-fn', float(mse(torch.as_tensor(pred), torch.as_tensor(yt)).item()),
          float(np.mean((pred - yt) ** 2)), 1e-6)
    # soft-target CE (label smoothing applied to one-hot targets upstream)
    n, k, eps = 64, 3, 0.1
    oh = torch.zeros(n, k)
    oh[torch.arange(n), torch.as_tensor(y)] = 1.0
    y_smooth = (1 - eps) * oh + eps / k
    ref = float(cross_entropy(torch.as_tensor(logits), y_smooth).item())
    out = _ce_loss(mx.array(logits), mx.array(y), eps, k)
    mx.eval(out)
    check('soft-target CE', float(out.item()), ref, 1e-5)


def test_column_routing():
    """My dtype routing must agree with the selectors in conversion.py."""
    import pandas as pd
    from sklearn.compose import make_column_selector
    from pytabkit_mlx.api import _detect_columns
    df = pd.DataFrame({
        'f': rng.normal(size=20),
        'i': np.arange(20),
        's': ['a', 'b'] * 10,
        'o': pd.array(['a', None] * 10, dtype=object),
        'c': pd.Categorical(['x', 'y'] * 10),
        'b': [True, False] * 10,                          # plain bool: dropped
        'B': pd.array([True, False] * 10, dtype='boolean'),
    })
    nums, cats, dropped = _detect_columns(df)
    ref_nums = list(make_column_selector(dtype_include='number')(df))
    ref_cats = list(make_column_selector(
        dtype_include=['string', 'object', 'category', 'boolean'])(df))
    assert nums == ref_nums, (nums, ref_nums)
    assert cats == ref_cats, (cats, ref_cats)
    assert set(nums) | set(cats) | set(dropped) == set(df.columns)
    print(f'[OK ] column routing: nums={nums} cats={cats} dropped={dropped}')


if __name__ == '__main__':
    test_one_hot_layer()
    test_one_hot_pipeline()
    test_metrics_and_losses()
    test_column_routing()
    print('ALL REAL-CODE PARITY CHECKS PASSED')
