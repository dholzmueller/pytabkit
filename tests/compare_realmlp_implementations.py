"""Compare the standalone RealMLP with the original implementation on small datasets.

Both implementations are run through their AlgInterface on identical train/val/test splits
(best epoch selected on the validation set), with matching hyperparameters.
This is not a unit test — it prints test scores side by side.
Run with: .venv/bin/python tests/compare_realmlp_implementations.py [n_seeds] [dataset name filter]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine, make_regression

from pytabkit.models import utils
from pytabkit.models.alg_interfaces.base import InterfaceResources, SplitIdxs
from pytabkit.models.alg_interfaces.nn_interfaces import NNAlgInterface
from pytabkit.models.data.data import DictDataset, TensorInfo
from pytabkit.models.realmlp.alg_interface import StandaloneRealMLPAlgInterface
from pytabkit.models.sklearn.default_params import DefaultParams
from pytabkit.models.training.logging import StdoutLogger


def make_ds(x_num: np.ndarray, y: np.ndarray, n_classes: int, x_cat=None, cat_sizes=None) -> DictDataset:
    y_t = torch.as_tensor(y[:, None], dtype=torch.long if n_classes > 0 else torch.float32)
    tensors = {'x_cont': torch.as_tensor(x_num, dtype=torch.float32), 'y': y_t}
    infos = {'x_cont': TensorInfo(feat_shape=[x_num.shape[1]]),
             'y': TensorInfo(cat_sizes=[n_classes]) if n_classes > 0 else TensorInfo(feat_shape=[1])}
    if x_cat is not None:
        tensors['x_cat'] = torch.as_tensor(x_cat, dtype=torch.long)
        infos['x_cat'] = TensorInfo(cat_sizes=cat_sizes)
    return DictDataset(tensors, infos)


def synthetic_with_cats(seed=0, n=1000):
    """Regression with numerical + low/high-cardinality categorical features (category 0 = missing)."""
    rng = np.random.default_rng(seed)
    x_num = rng.standard_normal((n, 4))
    cat_sizes = [3, 6, 40]
    x_cat = np.stack([rng.integers(0, cs, n) for cs in cat_sizes], axis=1)
    effects = [rng.standard_normal(cs) for cs in cat_sizes]
    y = np.sin(2 * x_num[:, 0]) + x_num[:, 1] ** 2 + sum(e[x_cat[:, i]] for i, e in enumerate(effects))
    return x_num, y + 0.1 * rng.standard_normal(n), x_cat, cat_sizes


def run(alg_class, params, ds, train_idxs, val_idxs, seed):
    split = SplitIdxs(train_idxs=train_idxs[None], val_idxs=val_idxs[None], test_idxs=None, split_seed=seed,
                      sub_split_seeds=[seed], split_id=0)
    alg = alg_class(**params)
    torch.manual_seed(seed)
    alg.fit(ds, [split], InterfaceResources(n_threads=1, gpu_devices=[]), StdoutLogger(verbosity_level=0),
            [None], 'compare')
    return alg.predict(ds)[0]


def scores(y_pred: torch.Tensor, y: torch.Tensor, is_classification: bool) -> dict:
    if is_classification:
        log_probs = torch.log_softmax(y_pred, dim=-1)
        return {'acc': (y_pred.argmax(-1) == y[:, 0]).float().mean().item(),
                'logloss': -log_probs.gather(-1, y).mean().item()}
    return {'rmse': (y_pred - y).square().mean().sqrt().item()}


def compare(name, ds: DictDataset, n_seeds: int, **extra_params):
    if len(sys.argv) > 2 and sys.argv[2].lower() not in name.lower():
        return
    is_cls = ds.tensor_infos['y'].is_cat()
    # TabArena-style defaults: no label smoothing, cross-entropy for best-epoch selection
    orig_params = utils.join_dicts(DefaultParams.RealMLP_TD_CLASS if is_cls else DefaultParams.RealMLP_TD_REG,
                                   {'use_ls': False, 'val_metric_name': 'cross_entropy'} if is_cls else {},
                                   {'device': 'cpu', 'verbosity': 0}, extra_params)
    new_params = utils.join_dicts({} if is_cls else {'act': 'mish', 'lr': 0.2}, extra_params)
    results = {'original': [], 'standalone': []}
    times = {'original': 0.0, 'standalone': 0.0}
    y = ds.tensors['y']
    for seed in range(n_seeds):
        perm = torch.as_tensor(np.random.default_rng(seed).permutation(ds.n_samples))
        n_test, n_val = int(0.2 * ds.n_samples), int(0.2 * ds.n_samples)
        test_idxs, val_idxs, train_idxs = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]
        for key, alg_class, params in [('original', NNAlgInterface, orig_params),
                                       ('standalone', StandaloneRealMLPAlgInterface, new_params)]:
            start = time.time()
            y_pred = run(alg_class, params, ds, train_idxs, val_idxs, seed)
            times[key] += time.time() - start
            results[key].append(scores(y_pred[test_idxs], y[test_idxs], is_cls))
    print(f'\n=== {name} (n={ds.n_samples}, {n_seeds} seeds) ===')
    for key, res in results.items():
        summary = '  '.join(f'{m}={np.mean([r[m] for r in res]):.4f}±{np.std([r[m] for r in res]):.4f}'
                            for m in res[0])
        print(f'  {key:>10}: {summary}   ({times[key]:.1f}s)', flush=True)


if __name__ == '__main__':
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    for name, loader in [('Iris', load_iris), ('Wine', load_wine), ('Breast Cancer', load_breast_cancer)]:
        data = loader()
        compare(name, make_ds(data.data, data.target, len(np.unique(data.target))), n_seeds)
    data = load_diabetes()
    compare('Diabetes', make_ds(data.data, data.target, 0), n_seeds)
    x, y = make_regression(n_samples=1000, n_features=20, noise=10.0, random_state=42)
    compare('Synthetic regression', make_ds(x, y, 0), n_seeds)
    x_num, y, x_cat, cat_sizes = synthetic_with_cats()
    compare('Synthetic with categoricals', make_ds(x_num, y, 0, x_cat, cat_sizes), n_seeds)
    compare('Synthetic with categoricals, n_ens=4, auto batch size',
            make_ds(x_num, y, 0, x_cat, cat_sizes), n_seeds, n_ens=4, batch_size='auto')
