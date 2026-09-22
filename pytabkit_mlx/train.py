"""Minibatch Adam training with best-epoch selection, matching RealMLP-TD.

Matches pytabkit:
  - Adam betas (mom=0.9, sq_mom=0.95), eps 1e-8, with bias correction
  - decoupled weight decay applied BEFORE the Adam step as
        p *= 1 - wd_sched_val * wd_factor * lr_sched_val * lr_factor
    (OptimizerBase.step in optim/optimizers.py)
  - per-parameter lr/wd factors (see realmlp_mlx.PARAM_META)
  - schedules evaluated at progress t = samples_seen / total_samples
    (LearnerProgress.epoch_float / max_epochs, updated per batch)
  - label smoothing (ls_eps=0.1) mixed into the CE targets for classification
  - MSE on standardized targets for regression
  - validation every epoch; best-epoch params restored at the end
    (use_early_stopping=False by default: always trains all n_epochs)
"""

import math

import mlx.core as mx
import numpy as np

from . import realmlp_mlx as R
from .schedules import get_schedule


def _ce_loss(logits, y_idx, ls_eps, n_classes):
    logp = logits - mx.logsumexp(logits, axis=1, keepdims=True)
    nll = -logp[mx.arange(logits.shape[0]), y_idx]
    if ls_eps and ls_eps > 0:
        nll = (1.0 - ls_eps) * nll + ls_eps * (-logp.mean(axis=1))
    return nll.mean()


def _mse_loss(pred, y):
    return mx.mean((pred[:, 0] - y) ** 2)


def _class_error(logits, y_idx):
    pred = mx.argmax(logits, axis=1)
    return float(mx.mean(pred != y_idx).item())


def _rmse(pred, y):
    return float(mx.sqrt(mx.mean((pred[:, 0] - y) ** 2)).item())


def fit_model(P, train, val, cfg, rng, verbosity=0):
    """Train. train/val dicts: Xn/Xo/Xc numpy, y (int64 or float32).
    cfg keys: lr, wd, n_epochs, batch_size, lr_sched, wd_sched, p_drop,
              p_drop_sched, ls_eps, task ('class'/'reg'), n_classes.
    Returns (P_best, history). P is updated in place and returned as best."""
    lr_sched = get_schedule(cfg.get('lr_sched', 'coslog4'))
    wd_sched = get_schedule(cfg.get('wd_sched', 'flat_cos'))
    drop_sched = get_schedule(cfg.get('p_drop_sched', 'flat_cos'))
    base_lr = cfg['lr']
    base_wd = cfg.get('wd', 0.0)
    n_epochs = cfg.get('n_epochs', 256)
    bs = cfg.get('batch_size', 256)
    b1, b2 = cfg.get('mom', 0.9), cfg.get('sq_mom', 0.95)
    eps = cfg.get('opt_eps', 1e-8)
    ls_eps = cfg.get('ls_eps', 0.0)
    task = cfg['task']
    n_classes = cfg.get('n_classes', 0)

    A = R.array_params(P)
    M = {k: mx.zeros_like(v) for k, v in A.items()}
    V = {k: mx.zeros_like(v) for k, v in A.items()}
    mx.eval(M, V)

    n = train['Xn'].shape[0]
    # ParallelDictDataLoader(..., drop_last=True): the last incomplete batch
    # is omitted every epoch; progress runs over iterated samples only.
    eff_bs = min(bs, n)
    n_batches = n // eff_bs
    iterated = n_batches * eff_bs
    total = n_epochs * iterated
    seen = 0
    step = 0
    history = {'train_loss': [], 'val_metric': []}
    best_metric = math.inf
    best_snap = None

    def loss_fn(Ap, Xn, Xo, Xc, y, p_drop, masks):
        Pf = dict(P)
        Pf.update(Ap)
        logits = R.forward(Pf, Xn, Xo, Xc, training=True, p_drop=p_drop,
                           masks=masks)
        if task == 'class':
            return _ce_loss(logits, y, ls_eps, n_classes)
        return _mse_loss(logits, y)

    val_and_grad = mx.value_and_grad(loss_fn)
    exact_masks = cfg.get('exact_masks', False)
    if exact_masks:
        try:
            import torch
            import torch.nn.functional as TF
        except ImportError:
            raise ImportError(
                "exact_masks=True requires torch (dropout masks are drawn "
                "with the reference aten op). Install torch or use "
                "exact_masks=False.") from None

    for epoch in range(n_epochs):
        perm = rng.permutation(n)[:iterated]
        for s in range(0, iterated, eff_bs):
            idx = perm[s:s + eff_bs]
            # HyperparamCallback updates hypers at batch START from completed
            # samples, so schedules see pre-increment progress (first batch: t=0).
            t = seen / total
            lr_t = base_lr * lr_sched(t)
            wd_t = base_wd * wd_sched(t)
            p_t = cfg.get('p_drop', 0.0) * drop_sched(t)
            seen += len(idx)
            masks = None
            if p_t > 0.0 and exact_masks:
                # identical op to reference F.dropout (same stream advance);
                # mask recovered from dropout-on-ones (kept = 1/(1-p))
                masks = [mx.array((TF.dropout(
                    torch.ones((len(idx), h)), p_t, True) > 0).numpy())
                    for h in (P['hidden_sizes'][i] for i in range(len(P['hidden_sizes'])))]
            b = {k: mx.array(v[idx]) for k, v in
                 [('Xn', train['Xn']), ('Xo', train['Xo'])]}
            b['Xc'] = mx.array(train['Xc'][idx])
            b['y'] = mx.array(train['y'][idx])
            loss, grads = val_and_grad(A, b['Xn'], b['Xo'], b['Xc'], b['y'], p_t, masks)
            mx.eval(loss, grads)
            step += 1
            b1t = 1.0 - b1 ** step
            b2t = 1.0 - b2 ** step
            new_A = {}
            for k, p in A.items():
                lr_f, wd_f = R.param_meta(k)
                lr_eff = lr_t * lr_f
                g = grads[k]
                # decoupled wd first (pytabkit OptimizerBase order). NOTE the
                # reference applies hyper factors TWICE here: get_hyper_values
                # already includes them, and they are multiplied again below.
                dec = wd_t * wd_f * lr_t * lr_f * wd_f * lr_f
                p = p * (1.0 - dec) if dec != 0.0 else p
                m = M[k] = b1 * M[k] + (1 - b1) * g
                v = V[k] = b2 * V[k] + (1 - b2) * (g * g)
                upd = (m / b1t) / (mx.sqrt(v / b2t) + eps)
                new_A[k] = p - lr_eff * upd
            A = new_A
            mx.eval(A, M, V)
        P.update(A)
        # --- validation ---
        Pv = dict(P)
        logits = R.forward(Pv, mx.array(val['Xn']), mx.array(val['Xo']),
                           mx.array(val['Xc']), training=False)
        mx.eval(logits)
        if task == 'class':
            yv = mx.array(val['y'])
            metric = _class_error(logits, yv)
        else:
            # reference selects on denormalized + clamped outputs, raw-unit RMSE
            out = (np.array(logits.tolist(), dtype=np.float64)[:, 0]
                   * cfg['y_std'] + cfg['y_mean'])
            out = np.clip(out, cfg['y_min'], cfg['y_max'])
            metric = float(np.sqrt(np.mean((out - val['y_raw']) ** 2)))
        history['val_metric'].append(metric)
        # <= on purpose: latest epoch among tied best epochs is kept
        # (use_last_best_epoch=True default in pytabkit)
        if metric <= best_metric:
            best_metric = metric
            mx.eval(A)
            best_snap = {k: np.array(v.tolist(), dtype=np.float32)
                         for k, v in A.items()}
        if verbosity >= 2:
            print(f'epoch {epoch + 1}/{n_epochs} val={metric:.4f} best={best_metric:.4f}',
                  flush=True)
    # restore best-epoch params
    for k, arr in best_snap.items():
        A[k] = mx.array(arr)
    P.update(A)
    mx.eval(P)
    return P, history
