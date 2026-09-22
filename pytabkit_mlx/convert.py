"""Weight save/load + torch-mirror interop for the MLX RealMLP port.

pytabkit's torch weights live inside its scoped fitter framework, so there is
no one-to-one state_dict mapping; instead this module provides:
  - save_npz / load_npz: persist MLX params (arrays only) via mx.savez/load
  - to_torch / from_torch: copy params to/from plain-torch tensors with
    IDENTICAL names and shapes (used by parity.py's torch mirror model)
"""

import mlx.core as mx
import numpy as np
import torch

from . import realmlp_mlx as R


def save_npz(P, path):
    A = R.array_params(P)
    # mx.savez caps kwargs; use numpy instead (same as limix_mlx/convert.py)
    np.savez(path, **{k: np.array(v.tolist(), dtype=np.float32)
                      for k, v in A.items()})


def load_npz(P, path):
    z = np.load(path)
    for k in z.files:
        P[k] = mx.array(z[k].astype(np.float32))
    mx.eval(P)
    return P


def to_torch(P):
    return {k: torch.from_numpy(np.array(v.tolist(), dtype=np.float32))
            for k, v in R.array_params(P).items()}


def from_torch(P, Td):
    for k, t in Td.items():
        P[k] = mx.array(t.detach().cpu().numpy().astype(np.float32))
    mx.eval(P)
    return P
