"""Tests for the standalone RealMLP implementation."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pytest
import torch

from pytabkit.models.alg_interfaces.base import SplitIdxs, InterfaceResources
from pytabkit.models.data.data import DictDataset, TensorInfo
from pytabkit.models.realmlp.alg_interface import StandaloneRealMLPAlgInterface, sample_realmlp_params
from pytabkit.models.realmlp.model import Bias, RealMLP, RealMLPConfig, Weight
from pytabkit.models.realmlp.train import (
    SCHEDULES, auto_batch_size, batches, compute_input_stats, compute_metric, cross_entropy, fit, mse, predict,
)
from pytabkit.models.training.logging import StdoutLogger


# ---- helpers ----

def make_data(n_samples=200, n_features=5, n_classes=3, seed=42):
    """n_classes=0 gives regression data."""
    rng = np.random.RandomState(seed)
    x = rng.randn(n_samples, n_features).astype(np.float32)
    if n_classes > 0:
        y = (x @ rng.randn(n_features, n_classes)).argmax(axis=1)[:, None]
        return torch.from_numpy(x), torch.from_numpy(y).long()
    y = x @ rng.randn(n_features, 1) + 0.1 * rng.randn(n_samples, 1)
    return torch.from_numpy(x), torch.from_numpy(y).float()


def make_ds(x, y, n_classes, x_cat=None, cat_sizes=None):
    tensors = {'x_cont': x, 'y': y}
    infos = {'x_cont': TensorInfo(feat_shape=[x.shape[1]]),
             'y': TensorInfo(cat_sizes=[n_classes]) if n_classes > 0 else TensorInfo(feat_shape=[y.shape[1]])}
    if x_cat is not None:
        tensors['x_cat'] = x_cat
        infos['x_cat'] = TensorInfo(cat_sizes=cat_sizes)
    return DictDataset(tensors=tensors, tensor_infos=infos)


def make_split(train_idxs: torch.Tensor, val_idxs: Optional[torch.Tensor], seed=0):
    """train_idxs, val_idxs: (n_tv_splits, n)."""
    return SplitIdxs(train_idxs=train_idxs, val_idxs=val_idxs, test_idxs=None, split_seed=seed,
                     sub_split_seeds=list(range(seed, seed + train_idxs.shape[0])), split_id=0)


def fit_alg(ds, splits, **config):
    alg = StandaloneRealMLPAlgInterface(**{'batch_size': 64, **config})
    alg.fit(ds, splits, InterfaceResources(n_threads=1, gpu_devices=[]), StdoutLogger(verbosity_level=0),
            [None] * len(splits), 'test')
    return alg


def fast_config(**kwargs) -> RealMLPConfig:
    return RealMLPConfig(**{'n_epochs': 5, 'num_emb_type': 'none', 'use_best_epoch': False, **kwargs})


# ---- Config ----

def test_config():
    cfg = RealMLPConfig()
    assert cfg.get_hidden_sizes() == [256, 256, 256]
    assert cfg.use_ls is False and cfg.val_metric_name is None  # TabArena-style defaults
    assert RealMLPConfig(hidden_sizes=[128, 64]).get_hidden_sizes() == [128, 64]
    cfg = RealMLPConfig.td_class()
    assert cfg.use_ls is True and cfg.val_metric_name == 'class_error'
    cfg = RealMLPConfig.td_reg(lr=0.1)
    assert cfg.act == 'mish' and cfg.lr == 0.1


# ---- Preprocessing ----

def test_categorical_encoding():
    # sizes include the missing category 0
    model = RealMLP(n_num=1, cat_sizes=[2, 3, 5, 1, 100], n_out=2, n_models=1, config=RealMLPConfig(),
                    is_classification=True)
    x_cat = torch.tensor([[0, 0, 0, 0, 7], [1, 1, 3, 0, 99], [1, 2, 4, 0, 0]])
    x = model.input_features(torch.zeros(3, 1), x_cat)
    # numerical, binary, binary + missing, one-hot without missing column; constant and large ones dropped
    assert x.shape == (3, 1 + 1 + 1 + 4)
    assert x[:, 1].tolist() == [-1, 1, 1]
    assert x[:, 2].tolist() == [0, 1, -1]
    assert x[:, 3:].tolist() == [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    assert model.emb_cats == [4]


def test_input_stats_per_split_and_nan():
    x = torch.tensor([[0.0, 1.0, np.nan], [1.0, 1.0, np.nan], [2.0, 1.0, np.nan], [3.0, 1.0, np.nan],
                      [np.nan, 5.0, np.nan], [100.0, 5.0, np.nan]])
    median, scale = compute_input_stats(x, torch.tensor([[0, 1, 2, 3, 4], [2, 3, 5, 4, 0]]))
    assert median.shape == (2, 1, 3)
    assert median[0, 0].tolist() == [1.5, 1.0, 0.0]  # NaN ignored; all-NaN column -> 0
    assert median[1, 0, 0].item() == 2.5  # median of [0, 2, 3, 100]
    # second column of split 0 has iqr 0 -> fallback 0.5 * (max - min) = 2
    assert scale[0, 0, 1].item() == pytest.approx(0.5)
    assert scale[0, 0, 2].item() == 0.0


# ---- Metrics / schedules / loader ----

def test_metrics_vectorized():
    y_pred = torch.tensor([[[2.0, 0.0], [0.0, 2.0]], [[0.0, 2.0], [0.0, 2.0]]])  # (2 models, 2 samples, 2 classes)
    y = torch.tensor([[[0], [1]], [[0], [1]]])
    assert cross_entropy(y_pred, y).shape == (2,)
    assert compute_metric(y_pred, y, 'class_error').tolist() == [0.0, 0.5]
    # label smoothing towards the uniform distribution
    soft = torch.tensor([0.95, 0.05])
    expected = -(soft * torch.log_softmax(y_pred[0, 0], dim=-1)).sum()
    assert cross_entropy(y_pred[:1, :1], y[:1, :1], ls_eps=0.1).item() == pytest.approx(expected.item())
    assert mse(torch.tensor([[1.0], [2.0]]), torch.tensor([[1.5], [2.5]])).item() == pytest.approx(0.25)


def test_schedules():
    coslog4, flat_cos = SCHEDULES['coslog4'], SCHEDULES['flat_cos']
    assert coslog4(0.0) == 0.0 and coslog4(1.0) == pytest.approx(0.0)
    assert flat_cos(0.25) == 1.0 and 0.0 < flat_cos(0.75) < 1.0 and flat_cos(1.0) == pytest.approx(0.0)


def test_batches():
    x, y = torch.randn(20, 3), torch.randn(20, 1)
    idxs = torch.stack([torch.arange(15), torch.arange(5, 20)])
    assert [b['x_num'].shape for b in batches({'x_num': x, 'y': y}, idxs, 8)] == [(2, 8, 3), (2, 7, 3)]
    shuffled = list(batches({'x_num': x}, idxs, 8, shuffle=True, drop_last=True))
    assert len(shuffled) == 1 and shuffled[0]['x_num'].shape == (2, 8, 3)


def test_auto_batch_size():
    assert [auto_batch_size(n) for n in [500, 2000, 10000, 50000, 200000]] == [64, 128, 256, 512, 1024]


# ---- Model ----

@pytest.mark.parametrize('num_emb_type', ['none', 'pbld'])
def test_model_forward_with_nan_and_cats(num_emb_type):
    torch.manual_seed(0)
    n_models, batch = 2, 16
    model = RealMLP(n_num=3, cat_sizes=[3, 5, 20, 30], n_out=4, n_models=n_models,
                    config=RealMLPConfig(num_emb_type=num_emb_type), is_classification=True)
    x_num = torch.randn(n_models, batch, 3)
    x_num[0, :4, 1] = np.nan
    x_cat = torch.randint(0, 3, (n_models, batch, 4))
    with torch.no_grad():
        model(x_num, x_cat, init=True)
    model.set_p_drop(0.1)
    out = model(x_num, x_cat)
    assert out.shape == (n_models, batch, 4)
    out.sum().backward()
    assert torch.isfinite(out).all()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.num_missing.grad[0, 0, 1] != 0.0  # learned missing value is used


@pytest.mark.parametrize('options', [dict(use_parametric_act=False, add_front_scale=False, act='relu'),
                                     dict(hidden_sizes=[32, 16], ens_av_before_softmax=True)])
def test_model_options(options):
    x, y = make_data(n_samples=64)
    model, _ = fit(fast_config(n_epochs=2, n_ens=2, **options), x, torch.zeros(64, 0, dtype=torch.long), [], y,
                   torch.arange(64)[None], n_classes=3)
    assert predict(model, x, torch.zeros(64, 0, dtype=torch.long)).shape == (1, 64, 3)


def test_layer_init():
    torch.manual_seed(0)
    x = 5 * torch.randn(3, 100, 10)
    weight, bias = Weight(3, 10, 32, 1.0, 1.0), Bias(3, 32, 1.0, 0.0)
    with torch.no_grad():
        weight.initialize(x)
        out = weight(x)
        bias.initialize(out)
    assert torch.allclose(out.std(dim=1, correction=0), torch.ones(3, 32), atol=1e-4)
    # he+5: bias is minus a convex combination of input values, so it lies within the negative input range
    assert ((-bias.bias >= out.min(dim=1, keepdim=True)[0]) & (-bias.bias <= out.max(dim=1, keepdim=True)[0])).all()


def test_refit_stop_epochs_per_split():
    x, y = make_data(n_samples=64)
    no_cat = torch.zeros(64, 0, dtype=torch.long)
    run = lambda stop_epochs: fit(fast_config(n_epochs=3), x, no_cat, [], y, torch.arange(64).expand(2, -1),
                                  n_classes=3, stop_epochs=stop_epochs, seed=1)[0].mlp[1].weight.detach()
    init_weight, weight = run([0, 0]), run([0, 2])
    assert torch.equal(weight[0], init_weight[0])  # split with stop epoch 0 keeps its initial parameters
    assert not torch.equal(weight[1], init_weight[1])


def test_core_fit_predict_with_validation():
    x, y = make_data(n_samples=150)
    no_cat = torch.zeros(150, 0, dtype=torch.long)
    model, info = fit(fast_config(n_epochs=4, use_best_epoch=True, batch_size=32), x, no_cat, [], y,
                      torch.arange(100)[None], val_idxs=torch.arange(100, 150)[None], n_classes=3)
    assert info['val_metric_name'] == 'cross_entropy'
    assert 1 <= info['best_mean_epochs'][0] <= 4 and info['best_epochs'][0][0] >= 1
    assert predict(model, x, no_cat).shape == (1, 150, 3)


# ---- AlgInterface ----

def test_alg_interface_classification_cv_refit_ens():
    x, y = make_data(n_samples=150)
    x[::7, 0] = np.nan
    x_cat = torch.stack([torch.randint(0, 3, (150,)), torch.randint(0, 30, (150,))], dim=-1)
    ds = make_ds(x, y, 3, x_cat=x_cat, cat_sizes=[3, 30])
    perm = torch.randperm(150)
    split = make_split(torch.stack([perm[:100], perm[50:]]), torch.stack([perm[100:], perm[:50]]))
    alg = fit_alg(ds, [split], n_epochs=4, n_ens=2, num_emb_type='pbld')
    preds = alg.predict(ds)
    assert preds.shape == (2, 150, 3) and torch.isfinite(preds).all()

    fit_params = alg.get_fit_params()
    assert len(fit_params) == 1 and set(fit_params[0]['stop_epoch']) == {'cross_entropy'}
    assert len(fit_params[0]['best_indiv_stop_epochs']['cross_entropy']) == 2

    refit = alg.get_refit_interface(n_refit=1)
    refit.fit(ds, [make_split(torch.arange(150)[None], None)], InterfaceResources(n_threads=1, gpu_devices=[]),
              StdoutLogger(verbosity_level=0), [None], 'test')
    assert refit.predict(ds).shape == (1, 150, 3)


def test_alg_interface_regression_uses_unnormalized_validation():
    x, y = make_data(n_samples=150, n_classes=0)
    y = 1000.0 + 100.0 * y  # validation must compare predictions and targets on the same scale
    ds = make_ds(x, y, 0)
    split = make_split(torch.arange(100)[None], torch.arange(100, 150)[None])
    alg = fit_alg(ds, [split], n_epochs=10, num_emb_type='none', verbosity=0)
    preds = alg.predict(ds)
    assert preds.shape == (1, 150, 1)
    rmse = (preds[0, 100:] - y[100:]).square().mean().sqrt().item()
    assert rmse < 0.5 * y.std().item()


def test_classification_learns():
    x, y = make_data(n_samples=300)
    alg = fit_alg(make_ds(x, y, 3), [make_split(torch.arange(300)[None], None)],
                  n_epochs=30, num_emb_type='none', use_best_epoch=False)
    accuracy = (alg.predict(make_ds(x, y, 3))[0].argmax(dim=-1) == y.squeeze(-1)).float().mean().item()
    assert accuracy > 0.7, f'Training accuracy {accuracy:.3f} too low (chance=0.33)'


def test_regression_learns():
    x, y = make_data(n_samples=300, n_classes=0)
    alg = fit_alg(make_ds(x, y, 0), [make_split(torch.arange(300)[None], None)],
                  n_epochs=30, num_emb_type='none', use_best_epoch=False, act='mish', lr=0.2)
    rmse = (alg.predict(make_ds(x, y, 0))[0] - y).square().mean().sqrt().item()
    assert rmse < 0.5 * y.std().item()


def test_vectorized_models_differ():
    x, y = make_data(n_samples=200)
    split = make_split(torch.stack([torch.arange(200), torch.randperm(200)]), None)
    preds = fit_alg(make_ds(x, y, 3), [split], n_epochs=3, num_emb_type='none').predict(make_ds(x, y, 3))
    assert not torch.allclose(preds[0], preds[1])


def test_unknown_config_key_warns():
    with pytest.warns(UserWarning, match='unknown parameters'):
        StandaloneRealMLPAlgInterface(n_epochs=1, not_a_param=3)


# ---- HPO sampling ----

def test_hpo_sampling():
    params = sample_realmlp_params(seed=42, search_space='tabarena-new')
    cfg = RealMLPConfig(**params)  # all sampled keys are config fields
    assert cfg.n_hidden_layers in [2, 3, 4] and cfg.hidden_width in [256, 384, 512]
    assert 0.0 <= cfg.p_drop <= 0.5 and cfg.act == 'mish' and cfg.n_ens == 8
    assert 'val_metric_name' not in params  # not part of the search space
    assert params['lr'] != sample_realmlp_params(seed=123)['lr']


def test_random_params_alg_interface():
    from pytabkit.models.realmlp.alg_interface import RandomParamsStandaloneRealMLPAlgInterface
    x, y = make_data(n_samples=120)
    ds = make_ds(x, y, 3)
    split = make_split(torch.arange(80)[None], torch.arange(80, 120)[None])
    alg = RandomParamsStandaloneRealMLPAlgInterface(model_idx=3, hpo_space_name='tabarena-new', n_epochs=2,
                                                    val_metric_name='class_error')
    alg.fit(ds, [split], InterfaceResources(n_threads=1, gpu_devices=[]), StdoutLogger(verbosity_level=0), [None], 't')
    assert alg.fit_params[0]['n_ens'] == 8 and 'n_epochs' not in alg.fit_params[0]  # configured n_epochs is used
    assert alg.predict(ds).shape == (1, 120, 3)


def test_required_resources():
    infos = {'x_cont': TensorInfo(feat_shape=[10]), 'x_cat': TensorInfo(cat_sizes=[3, 50]), 'y': TensorInfo(cat_sizes=[4])}
    ds = DictDataset(tensors=None, tensor_infos=infos, device='cpu', n_samples=5000)
    small = StandaloneRealMLPAlgInterface().get_required_resources(ds, 5, 0, 1, [0], 4000)
    large = StandaloneRealMLPAlgInterface(n_ens=8).get_required_resources(ds, 5, 0, 1, [0], 4000)
    assert 0 < small.gpu_ram_gb < large.gpu_ram_gb and small.cpu_ram_gb > 0 and small.time_s > 0


def test_only_embedded_categoricals():
    """No numerical or one-hot features: the robust scaling statistics are empty."""
    n = 100
    x_cat = torch.stack([torch.randint(0, 20, (n,)), torch.randint(0, 30, (n,))], dim=-1)
    y = (x_cat[:, :1] % 2).float()
    no_num = torch.zeros(n, 0)
    model, _ = fit(fast_config(n_epochs=2, batch_size=32), no_num, x_cat, [20, 30], y, torch.arange(n)[None])
    assert model.x_median.shape == (1, 1, 0)
    assert predict(model, no_num, x_cat).shape == (1, n, 1)



def test_tabarena_space():
    params = sample_realmlp_params(seed=42, search_space='tabarena')
    RealMLPConfig(**params)
    # like 'tabarena-new' (including the tuned wd factors), but without ensembles and categorical tuning
    assert 'n_ens' not in params and 'max_one_hot_cat_size' not in params and 'embedding_size' not in params
    assert 'scale_wd_factor' in params
