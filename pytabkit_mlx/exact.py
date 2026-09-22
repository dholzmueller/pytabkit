"""Exact-replay mode: clone a real pytabkit run bit-stream-faithfully.

1. Load init values dumped by extract_reference.py (real fitter chain) and
   map scoped names -> this port's keys.
2. Restore the torch global RNG snapshot taken at training start.
3. Train with draws from torch's GLOBAL rng in reference order:
     per epoch: one torch.randperm (dataloader __iter__)
     per batch: one torch.rand mask per dropout layer, in forward order
   (reference: F.dropout per hidden block; val draws nothing).

Then trajectories match the reference to kernel precision (~1e-6; torch-CPU
vs MLX kernels associate differently, so not bit-exact).
"""

import math

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as TF

from . import realmlp_mlx as R
from .schedules import get_schedule
from .train import _ce_loss, _mse_loss, _class_error, _rmse


def scope_to_key(scope):
    """Map a real scope string to a port param key. Returns (key, squeeze)."""
    s = scope.strip('/')
    parts = s.split('/')
    leaf = parts[-1]
    if leaf == 'weight_1':
        return 'plr_w1', False
    if leaf == 'bias_1':
        return 'plr_b1', False
    if leaf == 'weight_2':
        return 'plr_w2', False
    if leaf == 'bias_2':
        return 'plr_b2', False
    if leaf == 'scale':
        return 'front_scale', True
    if leaf == 'weight':
        if 'last_layer' in s:
            return 'head_w', False
        i = int([p for p in parts if p.startswith('layer-')][0].split('-')[1])
        return f'w{i}', False
    if leaf == 'bias':
        if 'last_layer' in s:
            return 'head_b', True
        i = int([p for p in parts if p.startswith('layer-')][0].split('-')[1])
        return f'b{i}', True
    if leaf == 'act':
        i = int([p for p in parts if p.startswith('layer-')][0].split('-')[1])
        return f'act_w{i}', True
    if leaf.startswith('emb'):
        return None, False  # handled by index below
    raise KeyError(scope)


def load_reference_init(P, npz_path):
    """Overwrite P's arrays with real init values. Returns torch rng state."""
    z = np.load(npz_path, allow_pickle=True)
    rng_state = None
    emb_i = 0
    for scope in z.files:
        arr = z[scope]
        if scope in ('__rng_state__', '__probe__'):
            if scope == '__rng_state__':
                rng_state = torch.from_numpy(arr.astype(np.uint8))
            continue
        if 'emb' in scope:
            P[f'emb_table_{emb_i}'] = mx.array(arr.astype(np.float32))
            emb_i += 1
            continue
        key, squeeze = scope_to_key(scope)
        if squeeze:
            arr = arr.reshape(-1)
        P[key] = mx.array(arr.astype(np.float32))
    mx.eval(P)
    return rng_state


def exact_fit(P, train, val, cfg, verbosity=0):
    """Train with torch-global-RNG-synced draws. Caller must have restored
    the snapshot via torch.set_rng_state BEFORE calling (state at training
    start, i.e. right after the individual fitter fit)."""
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
    n_layers = len(P['hidden_sizes'])

    A = R.array_params(P)
    M = {k: mx.zeros_like(v) for k, v in A.items()}
    V = {k: mx.zeros_like(v) for k, v in A.items()}
    mx.eval(M, V)

    n = train['Xn'].shape[0]
    eff_bs = min(bs, n)
    n_batches = n // eff_bs
    iterated = n_batches * eff_bs
    total = n_epochs * iterated
    seen = 0
    step = 0
    history = {'val_metric': []}
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

    vg = mx.value_and_grad(loss_fn)

    for epoch in range(n_epochs):
        perm = torch.randperm(n).numpy()[:iterated]  # dataloader __iter__
        for s in range(0, iterated, eff_bs):
            idx = perm[s:s + eff_bs]
            t = seen / total
            lr_t = base_lr * lr_sched(t)
            wd_t = base_wd * wd_sched(t)
            p_t = cfg.get('p_drop', 0.0) * drop_sched(t)
            seen += len(idx)
            masks = None
            if p_t > 0.0:
                # identical op to the reference F.dropout (same stream advance);
                # mask recovered from dropout-on-ones (kept entries = 1/(1-p))
                masks = [mx.array((TF.dropout(torch.ones((len(idx), h)), p_t, True) > 0).numpy())
                         for h in (P['hidden_sizes'][i] for i in range(n_layers))]
            b = (mx.array(train['Xn'][idx]), mx.array(train['Xo'][idx]),
                 mx.array(train['Xc'][idx]), mx.array(train['y'][idx]))
            loss, grads = vg(A, *b, p_t, masks)
            mx.eval(loss, grads)
            step += 1
            b1t, b2t = 1 - b1 ** step, 1 - b2 ** step
            new_A = {}
            for k, p in A.items():
                lr_f, wd_f = R.param_meta(k)
                lr_eff = lr_t * lr_f
                # reference double-applies hyper factors in wd (see train.py)
                dec = wd_t * wd_f * lr_t * lr_f * wd_f * lr_f
                p = p * (1.0 - dec) if dec != 0.0 else p
                m_ = M[k] = b1 * M[k] + (1 - b1) * grads[k]
                v_ = V[k] = b2 * V[k] + (1 - b2) * (grads[k] * grads[k])
                new_A[k] = p - lr_eff * ((m_ / b1t) / (mx.sqrt(v_ / b2t) + eps))
            A = new_A
            mx.eval(A, M, V)
        P.update(A)
        Pv = dict(P)
        logits = R.forward(Pv, mx.array(val['Xn']), mx.array(val['Xo']),
                           mx.array(val['Xc']), training=False)
        mx.eval(logits)
        if task == 'class':
            metric = _class_error(logits, mx.array(val['y']))
        else:
            # reference val: denormalized + clamped outputs, raw-unit RMSE
            out = (np.array(logits.tolist(), dtype=np.float64)[:, 0]
                   * cfg['y_std'] + cfg['y_mean'])
            out = np.clip(out, cfg['y_min'], cfg['y_max'])
            metric = float(np.sqrt(np.mean((out - val['y_raw']) ** 2)))
        history['val_metric'].append(metric)
        if metric <= best_metric:
            best_metric = metric
            mx.eval(A)
            best_snap = {k: np.array(v.tolist(), dtype=np.float32)
                         for k, v in A.items()}
        if verbosity >= 2:
            print(f'epoch {epoch + 1}/{n_epochs} val={metric:.4f} best={best_metric:.4f}',
                  flush=True)
    for k, arr in best_snap.items():
        A[k] = mx.array(arr)
    P.update(A)
    mx.eval(P)
    return P, history
