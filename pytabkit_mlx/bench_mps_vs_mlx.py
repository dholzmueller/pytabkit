"""Head-to-head timing: full RealMLP-TD fits, torch mirror (MPS if available)
vs MLX port. Same preprocessed data, same init weights, same batch order,
same schedules — wall-clock fit() time.

Run: ./.venv/bin/python -m pytabkit_mlx.bench_mps_vs_mlx
"""
import math
import sys
from pathlib import Path
import time

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pytabkit_mlx import realmlp_mlx as R
from pytabkit_mlx.preprocessing import TabPreprocessor
from pytabkit_mlx.rng import make_rng
from pytabkit_mlx.train import _ce_loss
from pytabkit_mlx.parity import TorchMirror
from pytabkit_mlx.schedules import get_schedule

N_EPOCHS, BS, REPEATS = 32, 256, 2
N, F = 20000, 30

rng = np.random.default_rng(0)
Xn_raw = rng.normal(size=(N, F)).astype(np.float32)
city = rng.choice([f'k{i}' for i in range(25)], size=N).tolist()
color = rng.choice(['r', 'g', 'b'], size=N).tolist()
y = ((Xn_raw[:, 0] > 0).astype(int) ^ (np.array(color) == 'r').astype(int)).astype(np.int64)

prep = TabPreprocessor().fit(Xn_raw, [color, city])
Xn, Xo, Xc = prep.transform(Xn_raw, [color, city])

r = make_rng(0, 'numpy')
P = R.init_params(F, Xo.shape[1], prep.emb_cat_sizes_, 2, [256] * 3, 'selu', r)
P['_n_out'] = 2
m = min(N, 8192)
sel = r.permutation(N)[:m]
S_parts = [np.array(R.pbld_forward(P, mx.array(Xn[sel])).tolist()).reshape(m, -1),
           Xo[sel].astype(np.float64)]
for i in range(P['n_emb_tables']):
    S_parts.append(np.array(P[f'emb_table_{i}'][mx.array(Xc[sel][:, i])].tolist()).reshape(m, -1))
R.init_data_dependent(P, np.concatenate(S_parts, axis=1), r)

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print('torch device:', device, flush=True)
Td = {k: torch.from_numpy(np.array(v.tolist(), dtype=np.float32))
      for k, v in R.array_params(P).items()}

lr_sched = get_schedule('coslog4')
drop_sched = get_schedule('flat_cos')

brng = np.random.default_rng(999)
eff_bs = min(BS, N)
n_batch = N // eff_bs
order = [brng.permutation(N)[:n_batch * eff_bs] for _ in range(N_EPOCHS)]

t_torch_runs = []
for _ in range(REPEATS):
    M = TorchMirror(P).float().to(device)
    M.train()
    groups, seen_f = {}, set()
    for n_, p_ in M.named_parameters():
        mk = n_[2:] if n_.startswith('p.') else n_.replace('embs.', 'emb_table_')
        f_, __ = R.param_meta(mk)
        p_.data.copy_(Td[mk])
        if f_ not in groups:
            groups[f_] = []
            seen_f.add(f_)
        groups[f_].append(p_)
    opt = torch.optim.Adam([{'params': ps, 'lr': 0.04 * f} for f, ps in groups.items()],
                           betas=(0.9, 0.95), eps=1e-8)
    base_lrs = [g['lr'] for g in opt.param_groups]
    Xt = torch.as_tensor(Xn).to(device)
    Xo_t = torch.as_tensor(Xo).to(device)
    Xc_t = torch.as_tensor(Xc).to(device)
    yt = torch.as_tensor(y).to(device)
    seen = 0
    total = N_EPOCHS * n_batch * eff_bs
    t0 = time.time()
    for ep in range(N_EPOCHS):
        for s in range(0, n_batch * eff_bs, eff_bs):
            idx = order[ep][s:s + eff_bs]
            seen += len(idx)
            t = seen / total
            for g, b_ in zip(opt.param_groups, base_lrs):
                g['lr'] = b_ * lr_sched(t)
            opt.zero_grad()
            (TF.cross_entropy(M(Xt[idx], Xo_t[idx], Xc_t[idx]), yt[idx],
                              label_smoothing=0.1)).backward()
            opt.step()
    if device == 'mps':
        torch.mps.synchronize()
    t_torch_runs.append(time.time() - t0)
t_torch = float(np.median(t_torch_runs))

A = R.array_params(P)
Ms = {k: mx.zeros_like(v) for k, v in A.items()}
Vs = {k: mx.zeros_like(v) for k, v in A.items()}


def loss_fn(Ap, Xb, Ob, Cb, yb, p_drop):
    Pf = dict(P)
    Pf.update(Ap)
    return _ce_loss(R.forward(Pf, Xb, Ob, Cb, training=True, p_drop=p_drop), yb, 0.1, 2)


vg = mx.value_and_grad(loss_fn)
t_mlx_runs = []
for _ in range(REPEATS):
    A = R.array_params(P)
    Ms = {k: mx.zeros_like(v) for k, v in A.items()}
    Vs = {k: mx.zeros_like(v) for k, v in A.items()}
    mx.eval(A, Ms, Vs)
    seen = 0
    step = 0
    t0 = time.time()
    for ep in range(N_EPOCHS):
        for s in range(0, n_batch * eff_bs, eff_bs):
            idx = order[ep][s:s + eff_bs]
            seen += len(idx)
            t = seen / total
            lr_t = 0.04 * lr_sched(t)
            wd_t = 0.02 * get_schedule('flat_cos')(t)
            p_t = 0.15 * drop_sched(t)
            loss, grads = vg(A, mx.array(Xn[idx]), mx.array(Xo[idx]),
                             mx.array(Xc[idx]), mx.array(y[idx]), p_t)
            mx.eval(loss, grads)
            step += 1
            b1t, b2t = 1 - 0.9 ** step, 1 - 0.95 ** step
            new_A = {}
            for k, p in A.items():
                lr_f, wd_f = R.param_meta(k)
                dec = wd_t * wd_f * lr_t * lr_f
                p = p * (1.0 - dec) if dec != 0.0 else p
                m_ = Ms[k] = 0.9 * Ms[k] + 0.1 * grads[k]
                v_ = Vs[k] = 0.95 * Vs[k] + 0.05 * grads[k] * grads[k]
                new_A[k] = p - lr_t * lr_f * ((m_ / b1t) / (mx.sqrt(v_ / b2t) + 1e-8))
            A = new_A
            mx.eval(A, Ms, Vs)
    t_mlx_runs.append(time.time() - t0)
t_mlx = float(np.median(t_mlx_runs))

print(f'torch({device}) full {N_EPOCHS}-epoch fit x{REPEATS} median: {t_torch:.1f}s')
print(f'MLX            full {N_EPOCHS}-epoch fit x{REPEATS} median: {t_mlx:.1f}s')
print(f'speedup (torch/MLX): {t_torch / t_mlx:.2f}x')
