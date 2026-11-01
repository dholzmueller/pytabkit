"""Adapter connecting the standalone RealMLP to pytabkit's AlgInterface (used by the benchmark code)."""
from __future__ import annotations

import contextlib
import dataclasses
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from pytabkit.models.alg_interfaces.alg_interfaces import AlgInterface, RandomParamsAlgInterface
from pytabkit.models.alg_interfaces.base import SplitIdxs, InterfaceResources, RequiredResources
from pytabkit.models.data.data import DictDataset
from pytabkit.models.realmlp.model import RealMLP, RealMLPConfig
from pytabkit.models.realmlp.train import auto_batch_size, fit, predict
from pytabkit.models.training.logging import Logger


@contextlib.contextmanager
def _no_tf32():
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def config_from_dict(d: dict) -> RealMLPConfig:
    """Create a RealMLPConfig from a flat dict. Unknown keys are ignored with a warning."""
    valid_fields = {f.name for f in dataclasses.fields(RealMLPConfig)}
    unknown = sorted(k for k in d if k not in valid_fields)
    if unknown:
        warnings.warn(f'RealMLPConfig: ignoring unknown parameters {unknown}')
    return RealMLPConfig(**{k: v for k, v in d.items() if k in valid_fields})


def _inputs(ds: DictDataset) -> tuple[torch.Tensor, torch.Tensor]:
    empty = lambda dtype: torch.zeros(ds.n_samples, 0, dtype=dtype)
    return ds.tensors.get('x_cont', empty(torch.float32)), ds.tensors.get('x_cat', empty(torch.long))


class StandaloneRealMLPAlgInterface(AlgInterface):
    def __init__(self, fit_params=None, **config):
        super().__init__(fit_params=fit_params, **config)
        self.realmlp_config = config_from_dict(config)
        self.model = None

    def get_refit_interface(self, n_refit: int, fit_params=None):
        return StandaloneRealMLPAlgInterface(fit_params or self.fit_params, **self.config)

    def fit(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources,
            logger: Logger, tmp_folders: List[Optional[Path]], name: str):
        n_tv_splits = idxs_list[0].train_idxs.shape[0]
        y_info = ds.tensor_infos['y']
        val_idxs, stop_epochs = None, None
        if self.fit_params is not None:  # refit: stop at the best mean epoch of the cross-validation
            stop_epochs = [epoch for params in self.fit_params for epoch in [_single_value(params['stop_epoch'])] * n_tv_splits]
        elif idxs_list[0].val_idxs is not None:
            val_idxs = torch.cat([si.val_idxs for si in idxs_list], dim=0)
        with _no_tf32():
            self.model, info = fit(
                self.realmlp_config, *_inputs(ds),
                cat_sizes=ds.tensor_infos['x_cat'].get_cat_sizes().tolist() if 'x_cat' in ds.tensor_infos else [],
                y=ds.tensors['y'], train_idxs=torch.cat([si.train_idxs for si in idxs_list], dim=0),
                val_idxs=val_idxs, n_classes=y_info.get_cat_sizes()[0].item() if y_info.is_cat() else 0,
                n_tt_splits=len(idxs_list), stop_epochs=stop_epochs, seed=idxs_list[0].sub_split_seeds[0],
                device=interface_resources.gpu_devices[0] if interface_resources.gpu_devices else 'cpu')
        if self.fit_params is None:  # one dict per train/test split, same format as the original implementation
            metric = info['val_metric_name']
            self.fit_params = [{'stop_epoch': {metric: mean_epoch}, 'best_indiv_stop_epochs': {metric: epochs}}
                               for mean_epoch, epochs in zip(info['best_mean_epochs'], info['best_epochs'])]

    def predict(self, ds: DictDataset) -> torch.Tensor:
        with _no_tf32():
            return predict(self.model, *_inputs(ds)).cpu()

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        """Rough estimate, following the heuristics of the original implementation."""
        cfg, infos = self.realmlp_config, ds.tensor_infos
        n_num = infos['x_cont'].get_n_features() if 'x_cont' in infos else 0
        cat_sizes = infos['x_cat'].get_cat_sizes().tolist() if 'x_cat' in infos else []
        is_classification = infos['y'].is_cat()
        n_out = infos['y'].get_cat_sizes()[0].item() if is_classification else infos['y'].get_n_features()
        model = RealMLP(n_num, cat_sizes, n_out, 1, cfg, is_classification)
        n_params = sum(p.numel() for p in model.parameters())
        # floats per sample in the forward pass: PBLD, MLP input, and a few tensors per hidden layer
        n_forward = n_num * (3 * cfg.plr_hidden_1 + 2 * cfg.plr_hidden_2) + 2 * model.n_mlp_in \
            + 4 * sum(cfg.get_hidden_sizes()) + n_out
        n_parallel = max(n_cv, n_refit) * n_splits * cfg.n_ens
        batch_size = auto_batch_size(n_train) if cfg.batch_size == 'auto' else cfg.batch_size
        gb = 1024 ** 3
        ds_gb = 5 * ds.n_samples * (4 * n_num + 8 * len(cat_sizes) + 8 * n_out) / gb
        # activations: training (2x for the backward pass) or prediction, whichever is larger;
        # parameters: parameters, gradients, Adam states and checkpoint
        pass_floats = n_forward * max(2 * batch_size, cfg.predict_batch_size)
        train_gb = n_parallel * 4 * (pass_floats + 5 * n_params) / gb
        gpu_ram_gb = 0.3 + ds_gb + 1.2 * train_gb
        cpu_ram_gb = 0.3 + ds_gb + 1.3 * train_gb / n_parallel
        time_s = ds.n_samples * cfg.n_epochs * 4e-5 * (2 if n_refit > 0 else 1)
        if self.config.get('use_gpu', True):
            return RequiredResources(time_s=time_s, n_threads=1.0, cpu_ram_gb=cpu_ram_gb, n_gpus=1,
                                     gpu_usage=min(1.0, n_parallel / 200), gpu_ram_gb=gpu_ram_gb)
        return RequiredResources(time_s=time_s, n_threads=1.0, cpu_ram_gb=cpu_ram_gb + gpu_ram_gb)


def _single_value(value) -> int:
    if isinstance(value, dict):
        if len(value) != 1:
            raise ValueError(f'Expected stop epochs for a single metric, got {value}')
        return next(iter(value.values()))
    return value


def sample_realmlp_params(seed: int, search_space: str = 'tabarena-new') -> dict:
    """Samples hyperparameters (RealMLPConfig fields) for random search; other fields keep their configured values.
    Both spaces extend the original ones by also tuning the scale/PLR/first-layer wd factors,
    and do not reproduce the original's random draws for a given seed.
    'tabarena-new' additionally uses 8-member ensembles and tunes the categorical encoding."""
    if search_space not in ('tabarena', 'tabarena-new'):
        raise ValueError(f'Unknown search space: {search_space}')
    rng = np.random.default_rng(seed)
    log_uniform = lambda low, high: float(np.exp(rng.uniform(np.log(low), np.log(high))))

    params = dict(
        n_hidden_layers=int(rng.integers(2, 4, endpoint=True)),
        hidden_width=int(rng.choice([256, 384, 512])),
        p_drop=float(rng.uniform(0.0, 0.5)),
        act='mish',
        plr_sigma=log_uniform(1e-2, 50),
        sq_mom=1.0 - log_uniform(5e-3, 5e-2),
        plr_lr_factor=log_uniform(5e-2, 3e-1),
        plr_wd_factor=log_uniform(5e-2, 3e-1),
        scale_lr_factor=log_uniform(2.0, 10.0),
        scale_wd_factor=log_uniform(2.0, 10.0),
        first_layer_lr_factor=log_uniform(0.3, 1.5),
        first_layer_wd_factor=log_uniform(0.3, 1.5),
        ls_eps_sched='coslog4',
        ls_eps=log_uniform(5e-3, 1e-1),
        lr=log_uniform(2e-2, 3e-1),
        wd=log_uniform(1e-3, 5e-2),
        use_ls=bool(rng.choice([False, True])),  # ignored for regression
    )
    if search_space == 'tabarena-new':
        params.update(max_one_hot_cat_size=int(np.floor(log_uniform(4.0, 33.0))),
                      embedding_size=int(rng.choice([4, 8, 16])), n_ens=8)
    if rng.uniform() > 0.5:  # large configs
        params.update(plr_hidden_1=int(rng.choice([8, 16, 32, 64])), plr_hidden_2=int(rng.choice([8, 16, 32, 64])),
                      n_epochs=int(rng.choice([256, 512])), use_early_stopping=True,
                      early_stopping_multiplicative_patience=3, early_stopping_additive_patience=40)
    else:  # always set the same parameters, so that they override the configured values
        params.update(plr_hidden_1=16, plr_hidden_2=4, n_epochs=256, use_early_stopping=False)
    return params


class RandomParamsStandaloneRealMLPAlgInterface(RandomParamsAlgInterface):
    """One step of random search HPO: hyperparameters are sampled from sample_realmlp_params with a seed
    derived from the split seed and model_idx (the step index)."""

    def _sample_params(self, is_classification: bool, seed: int, n_train: int) -> dict:
        params = sample_realmlp_params(seed, self.config.get('hpo_space_name', 'tabarena-new'))
        if 'n_epochs' in self.config:  # as in the original RealMLP random search, a configured n_epochs wins
            del params['n_epochs']
        return params

    def _create_interface_from_config(self, n_tv_splits: int, **config) -> AlgInterface:
        config.pop('hpo_space_name', None)
        return StandaloneRealMLPAlgInterface(**config)
