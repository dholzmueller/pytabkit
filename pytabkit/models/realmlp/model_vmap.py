"""RealMLP written for a single network, vectorized over many networks with torch.func.vmap."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, stack_module_state, vmap

from .model import ACTIVATIONS, RealMLPConfig


def _param(data: torch.Tensor, lr_factor: float = 1.0, wd_factor: float = 1.0) -> nn.Parameter:
    param = nn.Parameter(data)
    param.lr_factor, param.wd_factor = lr_factor, wd_factor  # read by the optimizer setup in train.py
    return param


def _one_hot(x: torch.Tensor, cat_size: int) -> torch.Tensor:
    """Category 0 means missing. Binary features (with missing) are encoded as a single feature."""
    if cat_size == 2:  # missing -> -1, category -> +1
        return (2 * x - 1).float().unsqueeze(-1)
    if cat_size == 3:  # missing -> 0, categories -> +1 / -1
        return ((x == 1).float() - (x == 2).float()).unsqueeze(-1)
    return (x.unsqueeze(-1) == torch.arange(1, cat_size, device=x.device)).float()  # missing -> all zeros


def _std_init(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Random weights, normalized such that each output has unit std on the batch x."""
    w = torch.randn_like(w)
    return w / (x @ (w / math.sqrt(w.shape[0]))).std(dim=0, correction=0).clamp(min=1e-30)


def _he_plus_5_init(x: torch.Tensor) -> torch.Tensor:
    """Bias = negative random convex combination of 5 values of each feature on the batch x."""
    n_batch, n_features = x.shape
    idxs = torch.randint(0, n_batch, (n_features, 5), device=x.device)
    weights = -torch.log(torch.rand(n_features, 5, device=x.device))  # Exponential(1) samples
    return -(x.T.gather(1, idxs) * weights).sum(dim=1) / weights.sum(dim=1)


class RealMLP(nn.Module):
    """A single RealMLP network. Inputs: x_num (batch, n_num) float with NaN = missing, x_cat (batch, n_cat) long."""

    def __init__(self, n_num: int, cat_sizes: list[int], n_out: int, config: RealMLPConfig, is_classification: bool):
        super().__init__()
        cfg = self.config = config
        self.n_num = n_num
        hidden_sizes = cfg.get_hidden_sizes()
        assert len(hidden_sizes) >= 1 and cfg.act in ACTIVATIONS and cfg.num_emb_type in ('pbld', 'none')

        # categorical features: small ones are one-hot encoded, large ones get embeddings (constant ones are dropped)
        self.one_hot_cats = [(i, cs) for i, cs in enumerate(cat_sizes) if 1 < cs <= cfg.max_one_hot_cat_size]
        self.emb_cats = [i for i, cs in enumerate(cat_sizes) if cs > cfg.max_one_hot_cat_size]
        emb_sizes = [cat_sizes[i] for i in self.emb_cats]
        n_one_hot = sum(1 if cs <= 3 else cs - 1 for _, cs in self.one_hot_cats)

        # preprocessing statistics (set by train.fit)
        self.register_buffer('x_median', torch.zeros(1, n_num + n_one_hot))
        self.register_buffer('x_scale', torch.ones(1, n_num + n_one_hot))
        for name, value in [('y_mean', 0.0), ('y_std', 1.0), ('y_min', -np.inf), ('y_max', np.inf)]:
            self.register_buffer(name, torch.full((1, n_out), value))
        self.register_buffer('emb_offsets', torch.as_tensor(np.cumsum([0] + emb_sizes[:-1]), dtype=torch.long))

        # numerical embeddings (PBLD: periodic, cos-bias, linear, densenet) and learned values for missing inputs
        self.use_pbld = cfg.num_emb_type == 'pbld' and n_num > 0
        h1, h2 = cfg.plr_hidden_1, cfg.plr_hidden_2 - 1  # the last PBLD output is the feature itself
        plr = lambda data: _param(data, cfg.plr_lr_factor, cfg.plr_wd_factor)
        if self.use_pbld:
            self.plr_weight_1 = plr(cfg.plr_sigma * torch.randn(n_num, h1))
            self.plr_bias_1 = plr(np.pi * (2 * torch.rand(n_num, h1) - 1))
            self.plr_weight_2 = plr((2 * torch.rand(n_num, h1, h2) - 1) / np.sqrt(h1))
            self.plr_bias_2 = plr((2 * torch.rand(n_num, h2) - 1) / np.sqrt(h1))
            self.plr_missing = plr(torch.zeros(n_num, h2))
        self.num_missing = plr(torch.zeros(n_num)) if self.use_pbld else _param(torch.zeros(n_num))
        self.cat_emb = _param(torch.randn(sum(emb_sizes), cfg.embedding_size))

        # MLP: front scale, then (linear, bias, activation, dropout) per hidden layer, then (linear, bias)
        d = (n_num * cfg.plr_hidden_2 if self.use_pbld else n_num) + n_one_hot + len(emb_sizes) * cfg.embedding_size
        self.scale = _param(torch.ones(d), cfg.scale_lr_factor, cfg.scale_wd_factor) if cfg.add_front_scale else None
        sizes = [d] + hidden_sizes + [n_out]
        first_lr = cfg.first_layer_lr_factor or 1.0
        first_wd = first_lr if cfg.first_layer_wd_factor is None else cfg.first_layer_wd_factor
        self.weights, self.biases, self.act_weights = nn.ParameterList(), nn.ParameterList(), nn.ParameterList()
        for i, (n_in, n_out_layer) in enumerate(zip(sizes[:-1], sizes[1:])):
            lr_f, wd_f = (first_lr, first_wd) if i == 0 else (1.0, 1.0)
            self.weights.append(_param(torch.randn(n_in, n_out_layer), lr_f, wd_f))
            self.biases.append(_param(torch.zeros(n_out_layer), cfg.bias_lr_factor * lr_f, cfg.bias_wd_factor))
            if i < len(hidden_sizes) and cfg.use_parametric_act:
                self.act_weights.append(_param(torch.ones(n_out_layer), cfg.act_lr_factor, cfg.act_wd_factor))

    def input_features(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """Numerical features followed by one-hot encoded categorical features (before scaling)."""
        return torch.cat([x_num] + [_one_hot(x_cat[..., i], cs) for i, cs in self.one_hot_cats], dim=-1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, p_drop: float = 0.0, init: bool = False):
        """Returns logits (classification) or standardized predictions (regression).
        init=True initializes weights and biases using this batch (must be called under torch.no_grad())."""
        # median centering, robust scaling, smooth clipping to (-3, 3)
        x = (self.input_features(x_num, x_cat) - self.x_median) * self.x_scale
        x = x / (1 + x.square() / 9).sqrt()
        x_num, x_one_hot = x[:, :self.n_num], x[:, self.n_num:]
        missing = torch.isnan(x_num)
        x_num = x_num.masked_fill(missing, 0.0)

        parts = []
        if self.use_pbld:
            z = torch.cos(2 * torch.pi * x_num[:, :, None] * self.plr_weight_1 + self.plr_bias_1)
            z = torch.einsum('bnh,nhk->bnk', z, self.plr_weight_2) + self.plr_bias_2
            parts.append(torch.where(missing[:, :, None], self.plr_missing, z).flatten(1))
        parts += [torch.where(missing, self.num_missing, x_num), x_one_hot]
        if self.emb_cats:
            parts.append(F.embedding(x_cat[:, self.emb_cats] + self.emb_offsets, self.cat_emb).flatten(1))
        x = torch.cat(parts, dim=1)
        if self.scale is not None:
            x = x * self.scale

        act = ACTIVATIONS[self.config.act]
        n_hidden = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            if init:
                w.copy_(_std_init(x, w))
            x = x @ (w / math.sqrt(w.shape[0]))  # NTK parameterization
            if init:
                b.copy_(_he_plus_5_init(x))
            x = x + b
            if i < n_hidden:
                x = x + (act(x) - x) * self.act_weights[i] if self.act_weights else act(x)
                if self.training and p_drop > 0.0:
                    x = F.dropout(x, p_drop)
        return x


class VectorizedRealMLP(nn.Module):
    """n_models independently initialized RealMLPs whose parameters are stacked along a leading dimension.
    Has the same interface as model.RealMLP, so it can be used with train.py."""

    def __init__(self, n_num: int, cat_sizes: list[int], n_out: int, n_models: int, config: RealMLPConfig,
                 is_classification: bool):
        super().__init__()
        self.config, self.n_out, self.is_classification = config, n_out, is_classification
        self.normalize_y = config.normalize_output and not is_classification
        self.clamp_y = config.clamp_output and not is_classification
        models = [RealMLP(n_num, cat_sizes, n_out, config, is_classification) for _ in range(n_models)]
        params, buffers = stack_module_state(models)
        # plain attribute (not a submodule): provides the code, its own tensors are replaced in functional_call
        object.__setattr__(self, 'net', models[0])
        self.names = {name: name.replace('.', '_') for name in [*params, *buffers]}
        for name, p in models[0].named_parameters():
            self.register_parameter(self.names[name], _param(params[name], p.lr_factor, p.wd_factor))
        for name, b in buffers.items():
            self.register_buffer(self.names[name], b)

    def train(self, mode: bool = True):
        self.net.train(mode)
        return super().train(mode)

    def input_features(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        return self.net.input_features(x_num, x_cat)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, p_drop: float = 0.0, init: bool = False):
        """x_num: (n_models, batch, n_num), x_cat: (n_models, batch, n_cat)."""
        tensors = {name: getattr(self, key) for name, key in self.names.items()}
        single = lambda tensors, x_num, x_cat: functional_call(self.net, tensors, (x_num, x_cat),
                                                               dict(p_drop=p_drop, init=init))
        return vmap(single, randomness='different')(tensors, x_num, x_cat)

    def transform_y(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.y_mean) / (self.y_std + 1e-30) if self.normalize_y else y

    def inverse_transform_y(self, y_pred: torch.Tensor) -> torch.Tensor:
        if self.normalize_y:
            y_pred = y_pred * self.y_std + self.y_mean
        if self.clamp_y:
            y_pred = torch.minimum(torch.maximum(y_pred, self.y_min), self.y_max)
        return y_pred
