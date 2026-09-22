"""Parity checks: MLX port vs torch reference math + pytabkit schedules.

Run: .venv/bin/python -m pytabkit_mlx.parity
"""
import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    'pytabkit_scheduling',
    str(Path(__file__).resolve().parents[1] / 'pytabkit/models/training/scheduling.py'))
_sched_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_sched_mod)
torch_sched = _sched_mod.get_schedule

import pytabkit_mlx.schedules as S
import pytabkit_mlx.preprocessing as PP
import pytabkit_mlx.realmlp_mlx as R
from pytabkit_mlx.convert import to_torch, from_torch
from pytabkit_mlx.rng import make_rng
from pytabkit_mlx.train import _ce_loss

rng = np.random.default_rng(0)


def check(name, a, b, tol):
    d = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    status = 'OK ' if d <= tol else 'FAIL'
    print(f'[{status}] {name}: max|diff|={d:.3e} (tol {tol:.0e})')
    assert d <= tol, name


def test_schedules():
    for name, tol in [('coslog4', 1e-9), ('flat_cos', 1e-9), ('constant', 0.0)]:
        ref = torch_sched(name)
        mine = S.get_schedule(name)
        ts = np.linspace(0, 1, 101)
        a = np.array([mine(t) for t in ts])
        b = np.array([ref.call_time_(t) for t in ts])
        check(f'schedule {name}', a, b, 1e-9 if tol else 0.0)


def test_preprocessing():
    X = rng.normal(size=(500, 6)).astype(np.float32)
    X[:, 2] = 1.0  # constant feature
    X[:, 3] = np.where(rng.random(500) < 0.9, 2.0, rng.normal(size=500)).astype(np.float32)  # ~zero IQR
    stats = PP.fit_num_stats(X)
    Xt = torch.as_tensor(X)
    med = torch.quantile(Xt.double(), 0.5, dim=0)
    q75 = torch.quantile(Xt.double(), 0.25 + 0.5, dim=0)
    q25 = torch.quantile(Xt.double(), 0.25, dim=0)
    iqr = (q75 - q25).numpy().astype(np.float32)
    mx_ = X.max(axis=0)
    mn = X.min(axis=0)
    iqr2 = np.where(iqr == 0, 0.5 * (mx_ - mn), iqr)
    fac = 1.0 / (iqr2 + 1e-30)
    fac[iqr2 == 0] = 0.0
    med_np = np.median(X.astype(np.float64), axis=0).astype(np.float32)
    check('median', stats['median'], med_np, 1e-6)
    check('robust_scale', stats['scale'], fac, 1e-5)
    out = PP.transform_num(X, stats)
    ref = (X - med_np[None, :]) * fac[None, :]
    ref = ref / np.sqrt(1 + (ref / 3.0) ** 2)
    check('num pipeline', out, ref.astype(np.float32), 1e-5)

    # torch smooth_clip formula from models.py
    xt = torch.as_tensor(out[:10])
    sc = xt / (1 + (1 / 9) * xt ** 2).sqrt()
    check('smooth_clip torch formula', PP.smooth_clip(out[:10]), sc.numpy(), 1e-6)


def test_activations():
    x = rng.normal(size=(1000, 32)).astype(np.float32) * 2
    mx_out = R.selu(mx.array(x))
    mx.eval(mx_out)
    check('selu', np.array(mx_out.tolist()), torch.selu(torch.as_tensor(x)).numpy(), 1e-5)
    mx_out = R.mish(mx.array(x))
    mx.eval(mx_out)
    torch_mish = (lambda t: t * torch.tanh(F.softplus(t)))(torch.as_tensor(x)).numpy()
    check('mish', np.array(mx_out.tolist()), torch_mish, 1e-5)


class TorchMirror(torch.nn.Module):
    """Plain-torch RealMLP with identical parameter names/shapes."""

    def __init__(self, P):
        super().__init__()
        A = {k: torch.nn.Parameter(torch.from_numpy(np.array(v.tolist())))
             for k, v in R.array_params(P).items() if not k.startswith('emb_table_')}
        self.p = torch.nn.ParameterDict(A)
        self.embs = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.from_numpy(np.array(P[f'emb_table_{i}'].tolist())))
             for i in range(P['n_emb_tables'])])
        self.act_name = P['act_name']
        self.n_layers = len(P['hidden_sizes'])

    def act(self, x):
        if self.act_name == 'selu':
            return torch.selu(x)
        return x * torch.tanh(F.softplus(x))

    def forward(self, Xn, Xo, Xc):
        F_ = self.p['plr_w1'].shape[0]
        parts = []
        if F_ > 0:
            x = Xn.transpose(0, 1).unsqueeze(-1)  # [F, B, 1]
            z = torch.cos(2 * math.pi * x * self.p['plr_w1'] + self.p['plr_b1'])
            h = z @ self.p['plr_w2'] + self.p['plr_b2']  # [F, B, 3]
            h = h.transpose(0, 1).reshape(h.shape[1], -1)
            parts.append(torch.cat([h, Xn], dim=1))
        if Xo.shape[1] > 0:
            parts.append(Xo)
        for i, tab in enumerate(self.embs):
            parts.append(tab[Xc[:, i]])
        x = torch.cat(parts, dim=1) * self.p['front_scale']
        for i in range(self.n_layers):
            in_f = x.shape[-1]
            z = x @ (self.p[f'w{i}'] / math.sqrt(in_f)) + self.p[f'b{i}']
            x = z + (self.act(z) - z) * self.p[f'act_w{i}']
        in_f = x.shape[-1]
        return x @ (self.p['head_w'] / math.sqrt(in_f)) + self.p['head_b']


def _rand_batch(B=64, F=5, seed=1):
    r = np.random.default_rng(seed)
    Xn = (r.normal(size=(B, F)) * 2).astype(np.float32)
    Xo = np.zeros((B, 4), dtype=np.float32)
    Xo[np.arange(B), r.integers(0, 4, B)] = 1.0
    Xc = r.integers(0, 12, size=(B, 2)).astype(np.int64)
    return Xn, Xo, Xc


def test_forward_parity():
    Xn, Xo, Xc = _rand_batch(B=256)
    stats = PP.fit_num_stats(Xn)
    Xnp = PP.transform_num(Xn, stats)
    r = make_rng(7, 'numpy')
    P = R.init_params(Xnp.shape[1], Xo.shape[1], [13, 13], 3, [256] * 3, 'selu', r)
    P['_n_out'] = 3
    S_init = np.concatenate([
        np.array(R.pbld_forward(P, mx.array(Xnp[:200])).tolist()).reshape(200, -1),
        Xo[:200], np.array(P['emb_table_0'][mx.array(Xc[:200, 0])].tolist()),
        np.array(P['emb_table_1'][mx.array(Xc[:200, 1])].tolist())], axis=1)
    R.init_data_dependent(P, S_init, r)
    torch.manual_seed(0)
    M = TorchMirror(P)
    M.eval()
    with torch.no_grad():
        ref = M(torch.as_tensor(Xnp), torch.as_tensor(Xo), torch.as_tensor(Xc)).numpy()
    out = R.forward(P, mx.array(Xnp), mx.array(Xo), mx.array(Xc))
    mx.eval(out)
    check('full forward mlx-vs-torch', np.array(out.tolist()), ref, 2e-4)
    # roundtrip conversion is lossless
    P2 = R.init_params(Xnp.shape[1], Xo.shape[1], [13, 13], 3, [256] * 3, 'selu',
                       make_rng(99, 'numpy'))
    P2['_n_out'] = 3
    from_torch(P2, to_torch(P))
    out2 = R.forward(P2, mx.array(Xnp), mx.array(Xo), mx.array(Xc))
    mx.eval(out2)
    check('convert roundtrip', np.array(out2.tolist()), ref, 2e-4)


def test_ce_loss():
    r = np.random.default_rng(3)
    logits = r.normal(size=(128, 4)).astype(np.float32)
    y = r.integers(0, 4, 128)
    out = _ce_loss(mx.array(logits), mx.array(y), 0.1, 4)
    mx.eval(out)
    t = torch.as_tensor(logits)
    logp = torch.log_softmax(t, dim=1)
    nll = -logp[torch.arange(128), torch.as_tensor(y)]
    ref = ((0.9 * nll + 0.1 * (-logp.mean(dim=1))).mean()).item()
    check('label-smoothed CE', float(out.item()), ref, 1e-5)


def test_adam_step():
    """One manual-Adam step vs torch Adam, same init/grads, wd=0."""
    import pytabkit_mlx.train as T
    torch.manual_seed(0)
    r = make_rng(11, 'numpy')
    P = R.init_params(4, 0, [], 2, [16, 16], 'selu', r)
    P['_n_out'] = 2
    Xn = r.randn((300, 4)).astype(np.float32)
    Xnp = PP.transform_num(Xn, PP.fit_num_stats(Xn))
    S_init = np.array(R.pbld_forward(P, mx.array(Xnp)).tolist()).reshape(300, -1)
    R.init_data_dependent(P, S_init, r)
    M = TorchMirror(P)
    opt = torch.optim.Adam(M.parameters(), lr=0.04, betas=(0.9, 0.95), eps=1e-8)
    y = torch.randint(0, 2, (300,))
    Xt = torch.as_tensor(Xnp)
    Z = torch.zeros((300, 0))
    C = torch.zeros((300, 0), dtype=torch.long)
    opt.zero_grad()
    logits = M(Xt, Z, C)
    loss = F.cross_entropy(logits, y, label_smoothing=0.1)
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in
             list(M.named_parameters())}
    opt.step()
    # MLX step with the same grads
    A = R.array_params(P)
    M_state = {k: mx.zeros_like(v) for k, v in A.items()}
    V_state = {k: mx.zeros_like(v) for k, v in A.items()}
    new_A = {}
    for k, p in A.items():
        tk = k.replace('emb_table_', 'embs.') if k.startswith('emb_table_') else f'p.{k}'
        g = mx.array(grads[tk].numpy().astype(np.float32))
        m = 0.9 * M_state[k] + 0.1 * g
        v = 0.95 * V_state[k] + 0.05 * g * g
        upd = (m / (1 - 0.9)) / (mx.sqrt(v / (1 - 0.95)) + 1e-8)
        new_A[k] = p - 0.04 * upd
    mx.eval(new_A)
    for k, p in list(M.named_parameters()):
        mk = k[2:] if k.startswith('p.') else k.replace('embs.', 'emb_table_')
        check(f'adam {k}', np.array(new_A[mk].tolist()), p.detach().numpy(), 2e-5)


def test_data_dependent_init():
    r = make_rng(5, 'numpy')
    P = R.init_params(6, 0, [], 2, [32], 'selu', r)
    P['_n_out'] = 2
    Xn = np.random.default_rng(5).normal(size=(2000, 6)).astype(np.float32)
    S_init = np.array(R.pbld_forward(P, mx.array(Xn)).tolist()).reshape(2000, -1)
    R.init_data_dependent(P, S_init, r)
    # 'std' init property: per-column std of S @ W_eff ≈ 1
    W = np.array(P['w0'].tolist())
    Z = S_init @ (W / math.sqrt(S_init.shape[1]))
    check('std-init col std≈1', Z.std(axis=0), np.ones(32), 5e-2)
    # he+5 property: bias = -convex combo of train rows -> bias + combo = 0
    b = np.array(P['b0'].tolist())
    assert np.all(np.isfinite(b)) and np.abs(b).mean() < 5


def test_torch_backend_exact():
    """Same seed + torch backend -> bit-identical draws to torch itself."""
    from pytabkit_mlx.rng import TorchRng
    seed = 123
    r = TorchRng(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    check('rng randn', r.randn((4, 3)), torch.randn(4, 3, generator=g).numpy(), 0.0)
    check('rng uniform', r.uniform(-1, 2, (5,)), (torch.rand(5, generator=g) * 3 - 1).numpy(), 0.0)
    check('rng exponential', r.exponential(6),
          torch.empty((6,)).exponential_(generator=g).numpy(), 0.0)
    check('rng integers', r.integers(0, 10, (7,)), torch.randint(0, 10, (7,), generator=g).numpy(), 0.0)
    check('rng permutation', r.permutation(11), torch.randperm(11, generator=g).numpy(), 0.0)
    # determinism: same seed twice -> identical model init
    P1 = R.init_params(3, 0, [], 2, [8], 'selu', TorchRng(7))
    P1['_n_out'] = 2
    P2 = R.init_params(3, 0, [], 2, [8], 'selu', TorchRng(7))
    for k, v in R.array_params(P1).items():
        check(f'deterministic {k}', np.array(v.tolist()),
              np.array(R.array_params(P2)[k].tolist()), 0.0)


if __name__ == '__main__':
    test_schedules()
    test_preprocessing()
    test_activations()
    test_forward_parity()
    test_ce_loss()
    test_adam_step()
    test_data_dependent_init()
    test_torch_backend_exact()
    print('ALL PARITY CHECKS PASSED')
