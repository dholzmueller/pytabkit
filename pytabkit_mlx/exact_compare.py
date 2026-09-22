"""Exact-replay verification: MLX exact_fit vs real pytabkit training.

Same init values (real dump), same RNG stream (snapshot + synced draws).
Compares per-epoch val curves (parsed from real stdout) and final params.

Run: ./.venv/bin/python -m pytabkit_mlx.exact_compare /tmp/refx [epochs]
"""
import io
import re
import sys
from pathlib import Path
from contextlib import redirect_stdout

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.datasets import load_diabetes
from pytabkit import RealMLP_TD_Regressor
from pytabkit_mlx import realmlp_mlx as R
from pytabkit_mlx.api import pytabkit_split
from pytabkit_mlx.exact import load_reference_init, scope_to_key, exact_fit
from pytabkit_mlx.preprocessing import TabPreprocessor, fit_target_stats
from pytabkit_mlx.rng import make_rng


def main(refdir, E=8):
    X, y = load_diabetes(return_X_y=True)
    meta = np.load(f'{refdir}/diabetes_meta.npz')
    itr = meta['itr_idx'] if 'itr_idx' in meta.files else None
    # recompute identically to the extractor (seed 0)
    itr, ite = pytabkit_split(len(y), 0, 0.8)
    itr_tr, itr_va = pytabkit_split(len(itr), 0, 0.8)
    tr_idx, va_idx = itr[itr_tr], itr[itr_va]
    assert np.array_equal(va_idx, meta['va_idx']), 'split mismatch vs extractor'

    prep = TabPreprocessor().fit(X[tr_idx], [])
    tr = prep.transform(X[tr_idx], [])
    va = prep.transform(X[va_idx], [])
    tgt = fit_target_stats(y[tr_idx])
    std = tgt['std']
    train = {'Xn': tr[0], 'Xo': tr[1], 'Xc': tr[2],
             'y': ((y[tr_idx] - tgt['mean']) / std).astype(np.float32)}
    val = {'Xn': va[0], 'Xo': va[1], 'Xc': va[2],
           'y': ((y[va_idx] - tgt['mean']) / std).astype(np.float32),
           'y_raw': y[va_idx].astype(np.float64)}

    # ---- MLX exact replay ----
    rng = make_rng(0, 'numpy')
    P = R.init_params(10, 0, [], 1, [256] * 3, 'mish', rng)
    P['_n_out'] = 1
    z = np.load(f'{refdir}/diabetes_init.npz', allow_pickle=True)
    probe_ref = z['__probe__'][:, 0]  # eval output: denormalized + clamped
    rng_state = load_reference_init(P, f'{refdir}/diabetes_init.npz')
    # init-forward probe must match the real chain bit-nearly
    probe_mine = R.forward(dict(P), mx.array(tr[0][:8]), mx.array(tr[1][:8]),
                           mx.array(tr[2][:8]), training=False)
    mx.eval(probe_mine)
    probe_mine = (np.array(probe_mine.tolist(), dtype=np.float64)[:, 0] * std
                  + tgt['mean'])
    probe_mine = np.clip(probe_mine, tgt['min'], tgt['max'])
    pdiff = float(np.abs(probe_mine - probe_ref).max())
    print(f'init forward probe max|diff|={pdiff:.2e}', flush=True)
    assert pdiff < 5e-4, 'init mapping wrong'
    torch.set_rng_state(rng_state)
    cfg = dict(lr=0.2, wd=2e-2, n_epochs=E, batch_size=256, lr_sched='coslog4',
               wd_sched='flat_cos', p_drop=0.15, p_drop_sched='flat_cos',
               ls_eps=0.0, task='reg', n_classes=0, y_mean=tgt['mean'],
               y_std=std, y_min=tgt['min'], y_max=tgt['max'])
    P, hist = exact_fit(P, train, val, cfg, verbosity=0)
    mlx_curve = np.array(hist['val_metric'])

    # ---- real run (val curve parsed from its logs) ----
    buf = io.StringIO()
    real = RealMLP_TD_Regressor(random_state=0, n_epochs=E, device='cpu', verbosity=2)
    with redirect_stdout(buf):
        real.fit(X[itr], y[itr], val_idxs=itr_va)
    out = buf.getvalue()
    real_curve = [float(m) for m in re.findall(r'val rmse =\s*([0-9.eE+-]+)', out)]
    print('real val curve:', [f'{v:.2f}' for v in real_curve], flush=True)
    print('mlx  val curve:', [f'{v:.2f}' for v in mlx_curve], flush=True)
    assert len(real_curve) == E, f'parsed {len(real_curve)} val lines'
    d = np.abs(np.array(real_curve) - mlx_curve)
    print(f'curve max|diff|={d.max():.4f}', flush=True)

    # ---- final params ----
    vm = real.alg_interface_.model.model
    maxd = 0.0
    for p in vm.parameters():
        scope = str(p.context.scope)
        if 'emb' in scope:
            continue
        key, squeeze = scope_to_key(scope)
        a = p.detach().cpu().numpy()
        while a.ndim > np.array(R.array_params(P)[key].tolist()).__array__().ndim:
            a = a[0]
        b = np.array(R.array_params(P)[key].tolist())
        dd = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
        maxd = max(maxd, dd)
        print(f'  {key}: max|diff|={dd:.2e}')
    print(f'FINAL PARAM max|diff|={maxd:.2e}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8)
