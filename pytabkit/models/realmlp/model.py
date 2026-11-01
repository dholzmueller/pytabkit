"""RealMLP model. All parameters have a leading dimension over n_models networks that are trained in parallel."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RealMLPConfig:
    """Defaults: RealMLP-TD for classification with TabArena changes (no label smoothing, cross-entropy for
    best-epoch selection). td_class() and td_reg() give the paper defaults."""
    # architecture
    n_hidden_layers: int = 3
    hidden_width: int = 256
    hidden_sizes: Optional[list[int]] = None  # overrides n_hidden_layers * [hidden_width]
    act: str = 'selu'  # see ACTIVATIONS
    use_parametric_act: bool = True
    add_front_scale: bool = True
    num_emb_type: str = 'pbld'  # 'pbld' or 'none'
    plr_sigma: float = 0.1
    plr_hidden_1: int = 16
    plr_hidden_2: int = 4
    max_one_hot_cat_size: int = 9  # larger categoricals get learned embeddings
    embedding_size: int = 8
    # per-parameter factors; the effective weight decay of a parameter is wd * lr * wd_factor * lr_factor
    scale_lr_factor: float = 6.0
    scale_wd_factor: float = 6.0
    first_layer_lr_factor: Optional[float] = None
    first_layer_wd_factor: Optional[float] = None  # None: same as first_layer_lr_factor (as in the original)
    bias_lr_factor: float = 0.1
    bias_wd_factor: float = 0.0
    act_lr_factor: float = 0.1
    act_wd_factor: float = 0.1
    plr_lr_factor: float = 0.1
    plr_wd_factor: float = 0.1
    # regularization
    p_drop: float = 0.15
    p_drop_sched: Optional[str] = 'flat_cos'
    wd: float = 0.02
    wd_sched: Optional[str] = 'flat_cos'
    use_ls: bool = False
    ls_eps: float = 0.1
    ls_eps_sched: Optional[str] = None
    # optimization
    lr: float = 0.04
    lr_sched: Optional[str] = 'coslog4'
    mom: float = 0.9
    sq_mom: float = 0.95
    n_epochs: int = 256
    batch_size: int | str = 256  # or 'auto'
    predict_batch_size: int = 1024
    # regression outputs
    normalize_output: bool = True
    clamp_output: bool = True
    # ensembling
    n_ens: int = 1
    ens_av_before_softmax: bool = False  # for classification, average logits instead of probabilities
    # best epoch / early stopping
    val_metric_name: Optional[str] = None  # None: 'cross_entropy' (classification) or 'rmse' (regression)
    use_best_epoch: bool = True
    use_early_stopping: bool = False
    early_stopping_additive_patience: int = 20
    early_stopping_multiplicative_patience: int = 2
    verbosity: int = 0

    def get_hidden_sizes(self) -> list[int]:
        return list(self.hidden_sizes) if self.hidden_sizes is not None \
            else [self.hidden_width] * self.n_hidden_layers

    @classmethod
    def td_class(cls, **kwargs) -> 'RealMLPConfig':
        return cls(**{'use_ls': True, 'val_metric_name': 'class_error', **kwargs})

    @classmethod
    def td_reg(cls, **kwargs) -> 'RealMLPConfig':
        return cls(**{'act': 'mish', 'lr': 0.2, **kwargs})


ACTIVATIONS = {'selu': F.selu, 'mish': F.mish, 'relu': F.relu, 'silu': F.silu, 'gelu': F.gelu, 'elu': F.elu}


def _param(data: torch.Tensor, lr_factor: float = 1.0, wd_factor: float = 1.0) -> nn.Parameter:
    param = nn.Parameter(data)
    param.lr_factor, param.wd_factor = lr_factor, wd_factor  # read by the optimizer setup in train.py
    return param


def _offsets(sizes: list[int]) -> torch.Tensor:
    return torch.as_tensor(np.cumsum([0] + sizes[:-1]), dtype=torch.long)


def _one_hot(x: torch.Tensor, cat_size: int) -> torch.Tensor:
    """Category 0 means missing. Binary features (with missing) are encoded as a single feature."""
    if cat_size == 2:  # missing -> -1, category -> +1
        return (2 * x - 1).float().unsqueeze(-1)
    if cat_size == 3:  # missing -> 0, categories -> +1 / -1
        return ((x == 1).float() - (x == 2).float()).unsqueeze(-1)
    return F.one_hot(x, cat_size)[..., 1:].float()  # missing -> all zeros


class Weight(nn.Module):
    """NTK-parameterized linear map with 'std' initialization: unit output std on the initialization batch."""

    def __init__(self, n_models: int, n_in: int, n_out: int, lr_factor: float, wd_factor: float):
        super().__init__()
        self.weight = _param(torch.randn(n_models, n_in, n_out), lr_factor, wd_factor)

    def initialize(self, x: torch.Tensor):
        w = torch.randn_like(self.weight)
        self.weight.copy_(w / (x @ (w / math.sqrt(w.shape[-2]))).std(dim=-2, correction=0, keepdim=True).clamp(min=1e-30))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ (self.weight / math.sqrt(self.weight.shape[-2]))


class Bias(nn.Module):
    """Bias with 'he+5' initialization: negative random convex combination of 5 input values per feature."""

    def __init__(self, n_models: int, n_features: int, lr_factor: float, wd_factor: float):
        super().__init__()
        self.bias = _param(torch.zeros(n_models, 1, n_features), lr_factor, wd_factor)

    def initialize(self, x: torch.Tensor):
        n_models, n_batch, n_features = x.shape
        idxs = torch.randint(0, n_batch, (n_models, n_features, 5), device=x.device)
        weights = torch.distributions.Exponential(1.0).sample((n_models, n_features, 5)).to(x.device)
        selected = torch.gather(x.transpose(-1, -2), dim=-1, index=idxs)
        self.bias.copy_(-(selected * weights / weights.sum(dim=-1, keepdim=True)).sum(dim=-1).unsqueeze(-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.bias


class Scale(nn.Module):
    def __init__(self, n_models: int, n_features: int, lr_factor: float, wd_factor: float):
        super().__init__()
        self.scale = _param(torch.ones(n_models, 1, n_features), lr_factor, wd_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class Activation(nn.Module):
    """Activation function, optionally parametric: x + weight * (f(x) - x) with weight initialized to 1."""

    def __init__(self, act: str, n_models: int, n_features: int, parametric: bool, lr_factor: float,
                 wd_factor: float):
        super().__init__()
        self.f = ACTIVATIONS[act]
        self.weight = _param(torch.ones(n_models, 1, n_features), lr_factor, wd_factor) if parametric else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x) if self.weight is None else x + (self.f(x) - x) * self.weight


class RealMLP(nn.Module):
    def __init__(self, n_num: int, cat_sizes: list[int], n_out: int, n_models: int, config: RealMLPConfig,
                 is_classification: bool):
        """n_num: number of numerical features (NaN = missing).
        cat_sizes: sizes of the categorical features, including the missing category 0.
        n_out: number of classes (classification) or targets (regression)."""
        super().__init__()
        cfg = self.config = config
        self.n_num, self.n_out, self.is_classification = n_num, n_out, is_classification
        self.normalize_y = cfg.normalize_output and not is_classification
        self.clamp_y = cfg.clamp_output and not is_classification
        hidden_sizes = cfg.get_hidden_sizes()
        assert len(hidden_sizes) >= 1 and cfg.act in ACTIVATIONS and cfg.num_emb_type in ('pbld', 'none')

        # categorical features: small ones are one-hot encoded, large ones get embeddings (constant ones are dropped)
        self.one_hot_cats = [(i, cs) for i, cs in enumerate(cat_sizes) if 1 < cs <= cfg.max_one_hot_cat_size]
        self.emb_cats = [i for i, cs in enumerate(cat_sizes) if cs > cfg.max_one_hot_cat_size]
        emb_sizes = [cat_sizes[i] for i in self.emb_cats]
        n_one_hot = sum(1 if cs <= 3 else cs - 1 for _, cs in self.one_hot_cats)

        # preprocessing statistics, computed per model on its training set (see train.fit)
        n_in = n_num + n_one_hot
        self.register_buffer('x_median', torch.zeros(n_models, 1, n_in))
        self.register_buffer('x_scale', torch.ones(n_models, 1, n_in))
        for name, value in [('y_mean', 0.0), ('y_std', 1.0), ('y_min', -np.inf), ('y_max', np.inf)]:
            self.register_buffer(name, torch.full((n_models, 1, n_out), value))

        # numerical embeddings (PBLD: periodic, cos-bias, linear, densenet) and learned values for missing inputs
        self.use_pbld = cfg.num_emb_type == 'pbld' and n_num > 0
        h1, h2 = cfg.plr_hidden_1, cfg.plr_hidden_2 - 1  # the last PBLD output is the feature itself
        plr = lambda data: _param(data, cfg.plr_lr_factor, cfg.plr_wd_factor)
        if self.use_pbld:
            self.plr_weight_1 = plr(cfg.plr_sigma * torch.randn(n_models, n_num, 1, h1))
            self.plr_bias_1 = plr(np.pi * (2 * torch.rand(n_models, n_num, 1, h1) - 1))
            self.plr_weight_2 = plr((2 * torch.rand(n_models, n_num, h1, h2) - 1) / np.sqrt(h1))
            self.plr_bias_2 = plr((2 * torch.rand(n_models, n_num, 1, h2) - 1) / np.sqrt(h1))
            self.plr_missing = plr(torch.zeros(n_models, 1, n_num, h2))
        self.num_missing = plr(torch.zeros(n_models, 1, n_num)) if self.use_pbld \
            else _param(torch.zeros(n_models, 1, n_num))

        self.register_buffer('emb_offsets', _offsets(emb_sizes))
        self.cat_emb = _param(torch.randn(n_models, sum(emb_sizes), cfg.embedding_size))

        # MLP: front scale, then (weight, bias, activation, dropout) per hidden layer, then (weight, bias)
        d = (n_num * cfg.plr_hidden_2 if self.use_pbld else n_num) + n_one_hot + len(emb_sizes) * cfg.embedding_size
        self.n_mlp_in = d
        layers = [Scale(n_models, d, cfg.scale_lr_factor, cfg.scale_wd_factor)] if cfg.add_front_scale else []
        sizes = [d] + hidden_sizes + [n_out]
        first_lr = cfg.first_layer_lr_factor or 1.0
        first_wd = first_lr if cfg.first_layer_wd_factor is None else cfg.first_layer_wd_factor
        for i, (n_in, n_out_layer) in enumerate(zip(sizes[:-1], sizes[1:])):
            lr_f, wd_f = (first_lr, first_wd) if i == 0 else (1.0, 1.0)  # first layer: also scales bias lr
            layers += [Weight(n_models, n_in, n_out_layer, lr_f, wd_f),
                       Bias(n_models, n_out_layer, cfg.bias_lr_factor * lr_f, cfg.bias_wd_factor)]
            if i < len(hidden_sizes):
                layers += [Activation(cfg.act, n_models, n_out_layer, cfg.use_parametric_act, cfg.act_lr_factor,
                                      cfg.act_wd_factor), nn.Dropout(cfg.p_drop)]
        self.mlp = nn.Sequential(*layers)

    def input_features(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Numerical features followed by one-hot encoded categorical features (before scaling)."""
        return torch.cat([x_num] + [_one_hot(x_cat[..., i], cs) for i, cs in self.one_hot_cats], dim=-1)

    def set_p_drop(self, p_drop: float):
        for layer in self.mlp:
            if isinstance(layer, nn.Dropout):
                layer.p = p_drop

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, init: bool = False):
        """x_num: (n_models, batch, n_num) float, x_cat: (n_models, batch, n_cat) long.
        Returns logits (classification) or standardized predictions (regression).
        init=True initializes the layers using this batch (must be called under torch.no_grad())."""
        # median centering, robust scaling, smooth clipping to (-3, 3)
        x = (self.input_features(x_num, x_cat) - self.x_median) * self.x_scale
        x = x / (1 + x.square() / 9).sqrt()
        x_num, x_one_hot = x[..., :self.n_num], x[..., self.n_num:]
        missing = torch.isnan(x_num)
        x_num = x_num.masked_fill(missing, 0.0)

        parts = []
        if self.use_pbld:
            z = x_num.transpose(-1, -2).unsqueeze(-1)  # (n_models, n_num, batch, 1)
            z = torch.cos(2 * torch.pi * z @ self.plr_weight_1 + self.plr_bias_1) @ self.plr_weight_2 + self.plr_bias_2
            z = z.transpose(-2, -3)  # (n_models, batch, n_num, plr_hidden_2 - 1)
            parts.append(torch.where(missing.unsqueeze(-1), self.plr_missing, z).flatten(-2))
        parts += [torch.where(missing, self.num_missing, x_num), x_one_hot]
        if self.emb_cats:
            n_models, n_cats, emb_size = self.cat_emb.shape
            idxs = x_cat[..., self.emb_cats] + self.emb_offsets + n_cats * torch.arange(n_models, device=x.device)[:, None, None]
            parts.append(F.embedding(idxs, self.cat_emb.reshape(-1, emb_size)).flatten(-2))
        x = torch.cat(parts, dim=-1)
        for layer in self.mlp:
            if init and hasattr(layer, 'initialize'):
                layer.initialize(x)
            x = layer(x)
        return x

    def transform_y(self, y: torch.Tensor) -> torch.Tensor:
        """Standardizes regression targets for training."""
        return (y - self.y_mean) / (self.y_std + 1e-30) if self.normalize_y else y

    def inverse_transform_y(self, y_pred: torch.Tensor) -> torch.Tensor:
        """Converts regression outputs back to the original scale and clamps them to the training range."""
        if self.normalize_y:
            y_pred = y_pred * self.y_std + self.y_mean
        if self.clamp_y:
            y_pred = torch.minimum(torch.maximum(y_pred, self.y_min), self.y_max)
        return y_pred
