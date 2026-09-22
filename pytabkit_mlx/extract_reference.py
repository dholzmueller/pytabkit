"""Drive the REAL pytabkit fitter chain (no lightning) to extract exact init
values + the torch RNG state at training start.

Mirrors NNCreator.create_model for the single-split TD case:
  manual_seed(combine(split_seed)) -> data_fitter fit (deterministic)
  manual_seed(combine(sub_split_seed)) -> individual_fitter fit (all RNG draws)
then snapshots torch.get_rng_state().

Saves: <out>.npz with {scope_string: param} for all trainable Variables,
plus rng_state, plus meta (shapes needed by the mapper).

Usage:
  ./.venv/bin/python -m pytabkit_mlx.extract_reference diabetes out_dir
  ./.venv/bin/python -m pytabkit_mlx.extract_reference cancer out_dir
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import types
_pkg = types.ModuleType('pytabkit')
_pkg.__path__ = [str(Path(__file__).resolve().parents[1] / 'pytabkit')]
sys.modules['pytabkit'] = _pkg

from pytabkit.models import utils
from pytabkit.models.data.data import DictDataset, TensorInfo
from pytabkit.models.nn_models.base import set_hp_context, SequentialLayer
from pytabkit.models.nn_models.models import NNFactory
from pytabkit.models.sklearn.default_params import DefaultParams
from pytabkit.models.training.coord import HyperparamManager

from sklearn.datasets import load_diabetes, load_breast_cancer
from pytabkit_mlx.api import pytabkit_split
from pytabkit_mlx.preprocessing import TabPreprocessor


def build(task, seed=0):
    if task == 'diabetes':
        X, y = load_diabetes(return_X_y=True)
        config = dict(DefaultParams.RealMLP_TD_REG)
        is_class = False
    else:
        X, y = load_breast_cancer(return_X_y=True)
        config = dict(DefaultParams.RealMLP_TD_CLASS)
        is_class = True
    itr, ite = pytabkit_split(len(y), seed, 0.8)
    itr_tr, itr_va = pytabkit_split(len(itr), seed, 0.8)
    tr_idx, va_idx = itr[itr_tr], itr[itr_va]

    # raw inputs in reference convention (my codes match OrdinalEncoder+1)
    if task == 'diabetes':
        Xn_all = np.asarray(X, dtype=np.float32)
        cat_cols = []
    else:
        Xn_all = np.asarray(X, dtype=np.float32)
        cat_cols = []
    n_classes = len(np.unique(y)) if is_class else 0
    if is_class:
        y_idx = np.unique(y, return_inverse=True)[1].astype(np.int64)
    else:
        y_idx = None

    x_cat = np.zeros((len(X), 0), dtype=np.int64)
    y_t = (torch.as_tensor(y_idx[tr_idx]).reshape(-1, 1) if is_class
           else torch.as_tensor(y[tr_idx], dtype=torch.float32).reshape(-1, 1))
    ds = DictDataset(
        tensors={'x_cont': torch.as_tensor(Xn_all[tr_idx]),
                 'x_cat': torch.as_tensor(x_cat[tr_idx]),
                 'y': y_t},
        tensor_infos={'x_cont': TensorInfo(feat_shape=[Xn_all.shape[1]]),
                      'x_cat': TensorInfo(cat_sizes=[]),
                      'y': TensorInfo(cat_sizes=[n_classes])},
        device='cpu', n_samples=len(tr_idx))

    hp = HyperparamManager(**config)
    with set_hp_context(hp):
        factory = NNFactory(**config)
        model_fitter = factory.create(ds.tensor_infos)
        static_fitter, dynamic_fitter = model_fitter.split_off_dynamic()
        raw_ds = ds
        static_model, ds = static_fitter.fit_transform(ds)
        data_fitter, individual_fitter = dynamic_fitter.split_off_individual()
        hp.get_more_info_dict()['trainval_ds'] = ds
        torch.manual_seed(utils.combine_seeds(seed, 0))
        data_tfm, tfmd_ds = data_fitter.fit_transform_subsample(
            ds, 1.0, needs_tensors=individual_fitter.needs_tensors)
        sub_seed = int(np.random.RandomState(seed).randint(0, 2 ** 31 - 1, size=1)[0])
        torch.manual_seed(utils.combine_seeds(sub_seed, 0))
        individual_tfm = individual_fitter.fit_transform_subsample(
            tfmd_ds, ram_limit_gb=1.0, needs_tensors=False)[0]
        rng_state = torch.get_rng_state().clone()
    model = SequentialLayer([static_model, data_tfm, individual_tfm])
    params = {}
    for p in model.parameters():
        scope = str(p.context.scope)
        params[scope] = p.detach().cpu().numpy().astype(np.float32)
    # forward probe on first 8 train rows (eval) to verify value mapping
    model.eval()
    with torch.no_grad():
        probe_ds = raw_ds.get_sub_dataset(torch.arange(8))
        probe_out = model(probe_ds).tensors['x_cont'].cpu().numpy().astype(np.float32)
    return params, rng_state.numpy(), {'tr_idx': tr_idx, 'va_idx': va_idx,
                                       'ite': ite, 'sub_seed': sub_seed}, probe_out


if __name__ == '__main__':
    task, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    params, rng_state, meta, probe = build(task)
    for scope in sorted(params):
        print(f'{scope}: {params[scope].shape}')
    np.savez(os.path.join(out_dir, f'{task}_init.npz'), **params,
             __rng_state__=rng_state, __probe__=probe)
    np.savez(os.path.join(out_dir, f'{task}_meta.npz'), **meta)
    print('saved', task)
