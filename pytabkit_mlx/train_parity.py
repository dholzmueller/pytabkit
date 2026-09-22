"""Controlled training comparison: MLX loop vs torch-Adam loop.

Same init weights, same batch order, same schedules. If the curves track,
the MLX training loop is faithful to the torch math.

Run: ./.venv/bin/python -m pytabkit_mlx.train_parity
"""
import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pytabkit_mlx import realmlp_mlx as R
from pytabkit_mlx.preprocessing import TabPreprocessor, fit_target_stats
from pytabkit_mlx.schedules import get_schedule
from pytabkit_mlx.parity import TorchMirror
from pytabkit_mlx.convert import to_torch
from pytabkit_mlx.rng import make_rng

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

N_EPOCHS = 40
BS = 256
LR = 0.2

X, y = load_diabetes(return_X_y=True)
Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, random_state=0)
idx_tr, idx_va = train_test_split(np.arange(len(ytr)), test_size=0.2, random_state=0)
prep = TabPreprocessor().fit(Xtr[idx_tr], [])
tr = prep.transform(Xtr[idx_tr], [])
va = prep.transform(Xtr[idx_va], [])
tgt = fit_target_stats(ytr[idx_tr])
std = tgt['std']
ytr_s = ((ytr[idx_tr] - tgt['mean']) / std).astype(np.float32)
yva_s = ((ytr[idx_va] - tgt['mean']) / std).astype(np.float32)

rng = make_rng(0, 'numpy')
P = R.init_params(10, 0, [], 1, [256] * 3, 'mish', rng)
P['_n_out'] = 1
m = tr[0].shape[0]
S = np.array(R.pbld_forward(P, mx.array(tr[0])).tolist()).reshape(m, -1)
R.init_data_dependent(P, S, rng)

# shared batch order (drop_last like ParallelDictDataLoader)
EFF_BS = min(BS, m)
N_BATCH = m // EFF_BS
ITER = N_BATCH * EFF_BS
batches = []
prng = np.random.default_rng(1234)
for _ in range(N_EPOCHS):
    batches.append(prng.permutation(m)[:ITER])

lr_sched = get_schedule('coslog4')

# ---- torch run (per-param lr factors like pytabkit hyper_factors) ----
M = TorchMirror(P)
M.train()
name_to_factor = {}
for n, p in M.named_parameters():
    mk = n[2:] if n.startswith('p.') else n.replace('embs.', 'emb_table_')
    lr_f, _ = R.param_meta(mk)
    name_to_factor[n] = lr_f
groups = {}
for n, p in M.named_parameters():
    groups.setdefault(name_to_factor[n], []).append(p)
opt = torch.optim.Adam([{'params': ps, 'lr': LR * f} for f, ps in groups.items()],
                       betas=(0.9, 0.95), eps=1e-8)
base_lrs = [g['lr'] for g in opt.param_groups]
Xt = torch.as_tensor(tr[0])
Z = torch.zeros((m, 0))
C = torch.zeros((m, 0), dtype=torch.long)
yt = torch.as_tensor(ytr_s)
Xv = torch.as_tensor(va[0])
Zv = torch.zeros((len(yva_s), 0))
Cv = torch.zeros((len(yva_s), 0), dtype=torch.long)
yv = torch.as_tensor(yva_s)
seen = 0
total = N_EPOCHS * ITER
torch_vals = []
for ep in range(N_EPOCHS):
    for s in range(0, ITER, EFF_BS):
        idx = batches[ep][s:s + EFF_BS]
        for g, base in zip(opt.param_groups, base_lrs):
            g['lr'] = base * lr_sched(seen / total)
        seen += len(idx)
        opt.zero_grad()
        out = M(Xt[idx], Z[idx], C[idx])[:, 0]
        ((out - yt[idx]) ** 2).mean().backward()
        opt.step()
    M.eval()
    with torch.no_grad():
        v = torch.sqrt(torch.mean((M(Xv, Zv, Cv)[:, 0] - yv) ** 2)).item()
    torch_vals.append(v)
    M.train()

# ---- MLX run (manual loop, same batches) ----
from pytabkit_mlx.train import _mse_loss
A = R.array_params(P)
Ms = {k: mx.zeros_like(v) for k, v in A.items()}
Vs = {k: mx.zeros_like(v) for k, v in A.items()}
mx.eval(Ms, Vs)


def loss_fn(Ap, Xn, y):
    Pf = dict(P)
    Pf.update(Ap)
    return _mse_loss(R.forward(Pf, Xn, mx.zeros((Xn.shape[0], 0)), mx.zeros((Xn.shape[0], 0), dtype=mx.int64), training=True, p_drop=0.0), y)


vg = mx.value_and_grad(loss_fn)
seen = 0
step = 0
mlx_vals = []
for ep in range(N_EPOCHS):
    for s in range(0, ITER, EFF_BS):
        idx = batches[ep][s:s + EFF_BS]
        lr_t = LR * lr_sched(seen / total)
        seen += len(idx)
        loss, grads = vg(A, mx.array(tr[0][idx]), mx.array(ytr_s[idx]))
        mx.eval(loss, grads)
        step += 1
        b1t, b2t = 1 - 0.9 ** step, 1 - 0.95 ** step
        new_A = {}
        for k, p in A.items():
            lr_f, _ = R.param_meta(k)
            m_ = Ms[k] = 0.9 * Ms[k] + 0.1 * grads[k]
            v_ = Vs[k] = 0.95 * Vs[k] + 0.05 * grads[k] * grads[k]
            new_A[k] = p - lr_t * lr_f * ((m_ / b1t) / (mx.sqrt(v_ / b2t) + 1e-8))
        A = new_A
        mx.eval(A, Ms, Vs)
    P.update(A)
    Pv = dict(P)
    vv = R.forward(Pv, mx.array(va[0]), mx.zeros((len(yva_s), 0)), mx.zeros((len(yva_s), 0), dtype=mx.int64))
    mx.eval(vv)
    mlx_vals.append(float(mx.sqrt(mx.mean((vv[:, 0] - mx.array(yva_s)) ** 2)).item()))

tv = np.array(torch_vals) * std
mv = np.array(mlx_vals) * std
print('epoch : torch_rmse  mlx_rmse')
for ep in [0, 4, 9, 19, 29, 39]:
    print(f'{ep:5d} : {tv[ep]:10.3f} {mv[ep]:9.3f}')
print('max|diff| =', float(np.abs(tv - mv).max()))
assert np.abs(tv - mv).max() < 2.0, 'training curves diverged'
print('TRAINING PARITY OK')
