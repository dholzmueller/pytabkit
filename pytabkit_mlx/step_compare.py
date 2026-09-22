"""Isolate step-2: real post-step params (restore disabled) vs MLX exact loop."""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.datasets import load_diabetes
from pytabkit.models.training import lightning_callbacks as _cb

_orig_restore = _cb.ParamCheckpointer.restore_all
_cb.ParamCheckpointer.restore_all = lambda self, model: None

from pytabkit import RealMLP_TD_Regressor
from pytabkit_mlx import realmlp_mlx as R
from pytabkit_mlx.api import pytabkit_split
from pytabkit_mlx.exact import load_reference_init, scope_to_key
from pytabkit_mlx.preprocessing import TabPreprocessor, fit_target_stats
from pytabkit_mlx.rng import make_rng
from pytabkit_mlx.schedules import get_schedule
from pytabkit_mlx.train import _mse_loss

E = 2
X, y = load_diabetes(return_X_y=True)
itr, ite = pytabkit_split(len(y), 0, 0.8)
itr_tr, itr_va = pytabkit_split(len(itr), 0, 0.8)
tr_idx, va_idx = itr[itr_tr], itr[itr_va]

real = RealMLP_TD_Regressor(random_state=0, n_epochs=E, device='cpu', verbosity=0)
real.fit(X[itr], y[itr], val_idxs=itr_va)
real_params = {}
for p in real.alg_interface_.model.model.parameters():
    s = str(p.context.scope)
    if 'emb' in s or s.startswith('__'):
        continue
    a = p.detach().cpu().numpy()
    while a.ndim > 1 and a.shape[0] == 1:
        a = a[0]
    real_params[s] = a

# ---- MLX: same init, synced stream, 2 steps, no restore ----
prep = TabPreprocessor().fit(X[tr_idx], [])
tr = prep.transform(X[tr_idx], [])
tgt = fit_target_stats(y[tr_idx])
std = tgt['std']
rng = make_rng(0, 'numpy')
P = R.init_params(10, 0, [], 1, [256] * 3, 'mish', rng)
P['_n_out'] = 1
rng_state = load_reference_init(P, '/tmp/refx/diabetes_init.npz')
torch.set_rng_state(rng_state)
import torch.nn.functional as TF

A = R.array_params(P)
M = {k: mx.zeros_like(v) for k, v in A.items()}
V = {k: mx.zeros_like(v) for k, v in A.items()}
n = tr[0].shape[0]
eff_bs, n_batches, iterated, total = 256, n // 256, (n // 256) * 256, E * ((n // 256) * 256)
lr_sched, wd_sched, drop_sched = (get_schedule(s) for s in ('coslog4', 'flat_cos', 'flat_cos'))
seen, step = 0, 0
ys = ((y[tr_idx] - tgt['mean']) / std).astype(np.float32)


def loss_fn(Ap, Xn, yb, p_drop, masks):
    Pf = dict(P)
    Pf.update(Ap)
    return _mse_loss(R.forward(Pf, Xn, mx.zeros((Xn.shape[0], 0)),
                               mx.zeros((Xn.shape[0], 0), dtype=mx.int64),
                               training=True, p_drop=p_drop, masks=masks), yb)


vg = mx.value_and_grad(loss_fn)
for ep in range(E):
    perm = torch.randperm(n).numpy()[:iterated]
    for s in range(0, iterated, eff_bs):
        idx = perm[s:s + eff_bs]
        t = seen / total
        lr_t, wd_t, p_t = 0.2 * lr_sched(t), 0.02 * wd_sched(t), 0.15 * drop_sched(t)
        seen += len(idx)
        masks = [mx.array((TF.dropout(torch.ones((len(idx), 256)), p_t, True) > 0).numpy())
                 for _ in range(3)] if p_t > 0 else None
        loss, grads = vg(A, mx.array(tr[0][idx]), mx.array(ys[idx]), p_t, masks)
        mx.eval(loss, grads)
        step += 1
        b1t, b2t = 1 - 0.9 ** step, 1 - 0.95 ** step
        new_A = {}
        for k, p in A.items():
            lr_f, wd_f = R.param_meta(k)
            dec = wd_t * wd_f * lr_t * lr_f * wd_f * lr_f
            p = p * (1.0 - dec) if dec != 0 else p
            m_ = M[k] = 0.9 * M[k] + 0.1 * grads[k]
            v_ = V[k] = 0.95 * V[k] + 0.05 * grads[k] * grads[k]
            new_A[k] = p - lr_t * lr_f * ((m_ / b1t) / (mx.sqrt(v_ / b2t) + 1e-8))
        A = new_A
        mx.eval(A, M, V)

P.update(A)
mx.eval(P)
print('post-step-2 param diffs:')
maxd = 0
for s, a in real_params.items():
    key, sq = scope_to_key(s)
    b = np.array(R.array_params(P)[key].tolist())
    if sq:
        b = b.reshape(a.shape)
    d = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
    maxd = max(maxd, d)
    print(f'  {key}: {d:.2e}')
print(f'MAX {maxd:.2e}')
