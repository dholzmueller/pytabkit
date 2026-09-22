"""Seed-compatible RNG backends for init + batch shuffling.

Same seed, same backend, same call order -> bit-identical draws.

Backends:
  'numpy': np.random.default_rng (no torch needed).
  'torch': dedicated torch.Generator; same realized values as same-order
    torch code.
  'exact': torch GLOBAL rng + reference manual_seeds. Replicates the exact
    global-stream consumption of a real pytabkit single-split TD run
    (init draws in fitter order, one randperm/epoch, dropout masks via the
    same aten op), so same random_state tracks a real run to kernel
    precision (~1e-4) with no weight dump. Requires torch.

Reference draw order (single split, n_ens=1), all on the global RNG:
  manual_seed(combine_seeds(split_seed, 0)); data phase draws nothing.
  manual_seed(combine_seeds(sub_seed, 0)); individual phase:
    PLR w1 randn, b1 uniform, w2 uniform, b2 uniform;
    emb tables randn (feature order); per layer then head:
    W randn, he+5 idx integers (out,5), he+5 exponential (out,5).
  training, per epoch: one randperm; per batch with p>0: one dropout draw
  per hidden layer in forward order.

combine_seeds/sub_seed replicate pytabkit.models.utils / sklearn_base.
"""

import numpy as np


def combine_seeds(seed_1, seed_2):
    """Exact replica of pytabkit.models.utils.combine_seeds."""
    generator = np.random.default_rng(seed=seed_1)
    return int(generator.integers(low=0, high=2 ** 24) + seed_2)


def sub_split_seed(split_seed):
    """First sub-split seed as in sklearn_base (n_cv=n_repeats=1)."""
    return int(np.random.RandomState(split_seed).randint(0, 2 ** 31 - 1, size=1)[0])


def _t(shape):
    return (shape,) if isinstance(shape, int) else tuple(shape)


class NumpyRng:
    def __init__(self, seed):
        self.r = np.random.default_rng(seed)

    def randn(self, shape):
        return self.r.standard_normal(shape)

    def uniform(self, low, high, shape):
        return self.r.uniform(low, high, shape)

    def exponential(self, shape):
        return self.r.exponential(1.0, shape)

    def integers(self, low, high, shape):
        return self.r.integers(low, high, shape)

    def permutation(self, n):
        return self.r.permutation(n)


class TorchRng:
    def __init__(self, seed):
        try:
            import torch
        except ImportError:
            raise ImportError(
                "rng_backend='torch' requires torch. Install torch or use "
                "rng_backend='numpy'.") from None
        self.torch = torch
        self.g = torch.Generator()
        self.g.manual_seed(seed)

    def randn(self, shape):
        return self.torch.randn(*_t(shape), generator=self.g).numpy()

    def uniform(self, low, high, shape):
        return (self.torch.rand(*_t(shape), generator=self.g)
                * (high - low) + low).numpy()

    def exponential(self, shape):
        # torch.empty(...).exponential_ uses the same kernel as
        # torch.distributions.Exponential(1.0).sample (bit-identical).
        return self.torch.empty(_t(shape), dtype=self.torch.float32
                                ).exponential_(generator=self.g).numpy()

    def integers(self, low, high, shape):
        return self.torch.randint(int(low), int(high), _t(shape),
                                  generator=self.g).numpy()

    def permutation(self, n):
        return self.torch.randperm(n, generator=self.g).numpy()


class GlobalTorchRng:
    """Draws from torch's GLOBAL rng (same ops as the reference fitters)."""

    def __init__(self):
        try:
            import torch
        except ImportError:
            raise ImportError(
                "rng_backend='exact' requires torch (it replicates the "
                "reference's global-RNG stream). Install torch or use "
                "rng_backend='numpy'.") from None
        self.torch = torch

    def randn(self, shape):
        return self.torch.randn(*_t(shape)).numpy()

    def uniform(self, low, high, shape):
        return (self.torch.rand(*_t(shape)) * (high - low) + low).numpy()

    def exponential(self, shape):
        # same op as BiasFitter.heplus_bias
        return self.torch.distributions.Exponential(1.0).sample(_t(shape)).numpy()

    def integers(self, low, high, shape):
        return self.torch.randint(int(low), int(high), _t(shape)).numpy()

    def permutation(self, n):
        return self.torch.randperm(n).numpy()


def make_rng(seed, backend='auto'):
    """backend: 'torch' | 'numpy' | 'exact' | 'auto' (torch when importable).

    'exact' returns GlobalTorchRng (seed applied via manual_seed by caller).
    Unknown backend names raise instead of silently falling back.
    """
    if backend == 'auto':
        try:
            import torch  # noqa
            backend = 'torch'
        except ImportError:
            backend = 'numpy'
    if backend == 'torch':
        return TorchRng(seed if seed is not None else 0)
    if backend == 'exact':
        return GlobalTorchRng()
    if backend == 'numpy':
        return NumpyRng(seed)
    raise ValueError(
        f"unknown rng_backend={backend!r} (expected 'torch', 'numpy', "
        f"'exact', or 'auto')")
