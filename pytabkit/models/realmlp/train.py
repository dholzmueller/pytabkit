"""Training of the standalone RealMLP: preprocessing statistics, data loading, training loop, prediction.

Many models (train/val splits x ensemble members) are trained in parallel as one vectorized RealMLP:
model index k belongs to split k // n_ens.
"""
from __future__ import annotations

import math
import warnings
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .model import RealMLP, RealMLPConfig


# ----- Metrics (averaged over the batch dimension, one value per model) -----

def cross_entropy(y_pred: torch.Tensor, y: torch.Tensor, ls_eps: float = 0.0) -> torch.Tensor:
    """y_pred: logits (..., batch, n_classes), y: class indices (..., batch, 1)."""
    loss = F.cross_entropy(y_pred.flatten(0, -2), y.flatten(), reduction='none', label_smoothing=ls_eps)
    return loss.view(y.shape[:-1]).mean(dim=-1)


def mse(y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (y_pred - y).square().mean(dim=(-1, -2))


def compute_metric(y_pred: torch.Tensor, y: torch.Tensor, metric_name: str) -> torch.Tensor:
    if metric_name == 'cross_entropy':
        return cross_entropy(y_pred, y)
    elif metric_name == 'class_error':
        return (y_pred.argmax(dim=-1) != y.squeeze(-1)).float().mean(dim=-1)
    elif metric_name == 'rmse':
        return mse(y_pred, y).sqrt()
    raise ValueError(f'Unknown metric: {metric_name}')


# ----- Preprocessing statistics, schedules, batching -----

def compute_input_stats(x: torch.Tensor, train_idxs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Median and robust scale (1 / IQR, or 2 / range if the IQR is zero) per split, ignoring NaNs.
    x: (n, d), train_idxs: (n_splits, n_train). Returns two tensors of shape (n_splits, 1, d)."""
    if x.shape[1] == 0:  # e.g., only embedded categorical features (np.nanquantile would return a wrong shape)
        empty = torch.zeros(train_idxs.shape[0], 1, 0, device=x.device)
        return empty, empty
    x_np = x.detach().cpu().numpy()
    medians, scales = [], []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN columns
        for idxs in train_idxs.cpu().numpy():
            x_train = x_np[idxs]
            q25, q50, q75 = np.nanquantile(x_train, [0.25, 0.5, 0.75], axis=0)
            iqr = q75 - q25
            iqr = np.where(iqr == 0.0, 0.5 * (np.nanmax(x_train, axis=0) - np.nanmin(x_train, axis=0)), iqr)
            valid = np.isfinite(iqr) & (iqr != 0.0)  # constant or all-NaN columns get scale 0
            scales.append(np.where(valid, 1.0 / np.where(valid, iqr, 1.0), 0.0))
            medians.append(np.nan_to_num(q50, nan=0.0))
    to_tensor = lambda arrs: torch.as_tensor(np.stack(arrs), dtype=torch.float32, device=x.device).unsqueeze(1)
    return to_tensor(medians), to_tensor(scales)


def auto_batch_size(n_train: int) -> int:
    for max_n_train, batch_size in [(1024, 64), (8192, 128), (30_000, 256), (100_000, 512)]:
        if n_train <= max_n_train:
            return batch_size
    return 1024


SCHEDULES = {
    None: lambda t: 1.0,
    'constant': lambda t: 1.0,
    'coslog4': lambda t: 0.5 * (1.0 - math.cos(2 * math.pi * math.log2(1.0 + 15.0 * t))),
    'flat_cos': lambda t: 1.0 if t <= 0.5 else 0.5 * (1.0 + math.cos(math.pi * (2.0 * t - 1.0))),
}


def batches(tensors: dict[str, torch.Tensor], idxs: torch.Tensor, batch_size: int, shuffle: bool = False,
            drop_last: bool = False) -> Iterator[dict[str, torch.Tensor]]:
    """Yields {name: tensor[batch_idxs]} with shape (n_parallel, batch, ...) for idxs of shape (n_parallel, n)."""
    n = idxs.shape[1]
    if shuffle:  # independent permutation for each parallel index set
        idxs = idxs.gather(1, torch.rand(idxs.shape, device=idxs.device).argsort(dim=1))
    for start in range(0, n - n % batch_size if drop_last else n, batch_size):
        batch_idxs = idxs[:, start:start + batch_size]
        yield {name: tensor[batch_idxs] for name, tensor in tensors.items()}


# ----- Prediction -----

@torch.no_grad()
def predict_idxs(model: RealMLP, tensors: dict[str, torch.Tensor], idxs: torch.Tensor) -> torch.Tensor:
    """Ensembled predictions (logits or unnormalized regression outputs) of model k on samples idxs[k],
    shape (n_splits, n_samples, n_out)."""
    model.eval()
    y_pred = torch.cat([model(b['x_num'], b['x_cat']) for b in batches(tensors, idxs, model.config.predict_batch_size)],
                       dim=-2)
    y_pred = model.inverse_transform_y(y_pred)
    n_ens = model.config.n_ens
    y_pred = y_pred.reshape(y_pred.shape[0] // n_ens, n_ens, *y_pred.shape[1:])
    if model.is_classification and not model.config.ens_av_before_softmax:  # average probabilities
        return torch.log(torch.softmax(y_pred, dim=-1).mean(dim=1) + 1e-30)
    return y_pred.mean(dim=1)


def predict(model: RealMLP, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
    """Predictions of all splits on all samples, shape (n_splits, n_samples, n_out)."""
    device = model.x_median.device
    tensors = {'x_num': x_num.float().to(device), 'x_cat': x_cat.long().to(device)}
    return predict_idxs(model, tensors, torch.arange(x_num.shape[0], device=device).expand(model.x_median.shape[0], -1))


# ----- Training -----

def fit(config: RealMLPConfig, x_num: torch.Tensor, x_cat: torch.Tensor, cat_sizes: list[int], y: torch.Tensor,
        train_idxs: torch.Tensor, val_idxs: Optional[torch.Tensor] = None, n_classes: int = 0,
        n_tt_splits: int = 1, stop_epochs: Optional[list[int]] = None, seed: int = 0,
        device: str = 'cpu') -> tuple[RealMLP, dict]:
    """Trains one RealMLP (with config.n_ens ensemble members) per train/val split, in parallel.

    x_num: (n, n_num) float, NaN = missing. x_cat: (n, n_cat) long, 0 = missing. Use (n, 0) tensors if absent.
    cat_sizes: number of categories per categorical feature, including the missing category.
    y: (n, 1) class indices (n_classes > 0) or (n, d) regression targets (n_classes = 0).
    train_idxs, val_idxs: (n_splits, n_train / n_val) with n_splits = n_tt_splits * n_tv_splits.
        val_idxs are used for best-epoch selection and early stopping (per split).
    stop_epochs: for refitting without validation data, one stop epoch per split.
    Returns the model and a dict with the best epoch per split ('best_epochs', shape (n_tt_splits, n_tv_splits)),
    the best epoch of the mean validation error per train/test split ('best_mean_epochs'), and 'val_metric_name'.
    """
    cfg = config
    is_classification = n_classes > 0
    x_num, x_cat = x_num.float().to(device), x_cat.long().to(device)
    y = (y.long() if is_classification else y.float()).to(device)
    train_idxs = train_idxs.to(device)
    n_splits, n_train = train_idxs.shape
    n_tv_splits, n_ens = n_splits // n_tt_splits, cfg.n_ens

    torch.manual_seed(seed)
    model = RealMLP(x_num.shape[1], cat_sizes, n_classes if is_classification else y.shape[1], n_splits * n_ens,
                    cfg, is_classification).to(device)

    # preprocessing statistics on the training set of each split, shared by its ensemble members
    with torch.no_grad():
        median, scale = compute_input_stats(model.input_features(x_num, x_cat), train_idxs)
        model.x_median.copy_(median.repeat_interleave(n_ens, dim=0))
        model.x_scale.copy_(scale.repeat_interleave(n_ens, dim=0))
        if not is_classification:
            y_train = y[train_idxs].repeat_interleave(n_ens, dim=0)
            model.y_mean.copy_(y_train.mean(dim=1, keepdim=True))
            model.y_std.copy_(y_train.std(dim=1, correction=0, keepdim=True))
            model.y_min.copy_(y_train.min(dim=1, keepdim=True)[0])
            model.y_max.copy_(y_train.max(dim=1, keepdim=True)[0])

    data = {'x_num': x_num, 'x_cat': x_cat}
    train_data, train_idxs = {**data, 'y': y}, train_idxs.repeat_interleave(n_ens, dim=0)
    batch_size = min(n_train, auto_batch_size(n_train) if cfg.batch_size == 'auto' else cfg.batch_size)
    has_val = val_idxs is not None and stop_epochs is None
    if has_val:
        val_idxs = val_idxs.to(device)
        val_y, val_idxs = y[val_idxs], val_idxs.repeat_interleave(n_ens, dim=0)
    val_metric_name = cfg.val_metric_name or ('cross_entropy' if is_classification else 'rmse')
    scheds = {name: SCHEDULES[getattr(cfg, name + '_sched')] for name in ['lr', 'wd', 'p_drop', 'ls_eps']}

    # AdamW with one parameter group per (lr_factor, wd_factor); AdamW decays parameters by lr * weight_decay
    groups = {}
    for p in model.parameters():
        groups.setdefault((p.lr_factor, p.wd_factor), []).append(p)
    optimizer = torch.optim.AdamW([{'params': ps, 'lr_factor': lr_f, 'wd_factor': wd_f}
                                   for (lr_f, wd_f), ps in groups.items()], betas=(cfg.mom, cfg.sq_mom))

    # data-dependent initialization on the first training batch
    model.train()
    batch = next(batches(train_data, train_idxs, batch_size, shuffle=True, drop_last=True))
    model.set_p_drop(cfg.p_drop * scheds['p_drop'](0.0))
    with torch.no_grad():
        model(batch['x_num'], batch['x_cat'], init=True)

    # checkpoint with the parameters of each split at its best epoch (or stop epoch when refitting)
    params = list(model.parameters())
    saved = [p.detach().clone() for p in params]

    @torch.no_grad()
    def save(split_mask: torch.Tensor):
        model_mask = split_mask.flatten().repeat_interleave(n_ens).to(device)
        for s, p in zip(saved, params):
            s[model_mask] = p[model_mask]

    n_epochs = cfg.n_epochs
    if stop_epochs is not None:
        stop_epochs = torch.as_tensor(stop_epochs)
        n_epochs = stop_epochs.max().item()

    best_errors = torch.full((n_tt_splits, n_tv_splits), math.inf)
    best_epochs = torch.zeros(n_tt_splits, n_tv_splits, dtype=torch.long)
    best_mean_errors = torch.full((n_tt_splits,), math.inf)
    best_mean_epochs = torch.zeros(n_tt_splits, dtype=torch.long)
    has_stopped = torch.zeros(n_tt_splits, n_tv_splits, dtype=torch.bool)

    n_steps, steps_per_epoch = 0, n_train // batch_size
    for epoch in range(1, n_epochs + 1):
        for batch in batches(train_data, train_idxs, batch_size, shuffle=True, drop_last=True):
            t = n_steps / (steps_per_epoch * cfg.n_epochs)  # training progress in [0, 1]
            lr, wd = cfg.lr * scheds['lr'](t), cfg.wd * scheds['wd'](t)
            for group in optimizer.param_groups:
                group['lr'] = lr * group['lr_factor']
                group['weight_decay'] = wd * group['wd_factor']

            model.set_p_drop(cfg.p_drop * scheds['p_drop'](t))
            y_pred = model(batch['x_num'], batch['x_cat'])
            if is_classification:
                loss = cross_entropy(y_pred, batch['y'], cfg.ls_eps * scheds['ls_eps'](t) if cfg.use_ls else 0.0)
            else:
                loss = mse(y_pred, model.transform_y(batch['y']))
            optimizer.zero_grad(set_to_none=True)
            loss.sum().backward()
            optimizer.step()
            n_steps += 1

        if stop_epochs is not None:
            save(stop_epochs == epoch)
        elif has_val:
            y_pred = predict_idxs(model, data, val_idxs)
            model.train()
            errors = compute_metric(y_pred, val_y, val_metric_name).cpu().view(n_tt_splits, n_tv_splits)
            mean_errors = errors.mean(dim=-1)
            if cfg.verbosity >= 2:
                print(f'Epoch {epoch}/{n_epochs}: val {val_metric_name} = {mean_errors.mean().item():.6f}')

            if cfg.use_early_stopping:
                has_stopped |= epoch > (cfg.early_stopping_multiplicative_patience * best_epochs
                                        + cfg.early_stopping_additive_patience)
            improved = ~has_stopped & (errors <= best_errors)
            best_errors[improved], best_epochs[improved] = errors[improved], epoch
            mean_improved = ~has_stopped.any(dim=-1) & (mean_errors <= best_mean_errors)
            best_mean_errors[mean_improved], best_mean_epochs[mean_improved] = mean_errors[mean_improved], epoch
            if cfg.use_best_epoch:
                save(improved)
            if cfg.use_early_stopping and has_stopped.all():
                break

    if stop_epochs is not None or (has_val and cfg.use_best_epoch):
        with torch.no_grad():
            for s, p in zip(saved, params):
                p.copy_(s)

    if not (has_val and cfg.use_best_epoch):
        best_epochs.fill_(cfg.n_epochs)
        best_mean_epochs.fill_(cfg.n_epochs)
    return model, {'best_epochs': best_epochs.tolist(), 'best_mean_epochs': best_mean_epochs.tolist(),
                   'val_metric_name': val_metric_name}
