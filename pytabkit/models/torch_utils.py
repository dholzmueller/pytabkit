from typing import List, Union, Optional

import torch
import numpy as np


def get_available_device_names() -> List['str']:
    device_names = ['cpu'] + [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    if torch.backends.mps.is_available():
        device_names.append('mps')
    return device_names


def seeded_randperm(n, device, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    # todo: can this not be generated directly on the device?
    return torch.randperm(n, generator=generator).to(device)


def permute_idxs(idxs, seed):
    return idxs[seeded_randperm(idxs.shape[0], idxs.device, seed)]


def batch_randperm(n_batch, n, device='cpu'):
    # batched randperm:
    # https://discuss.pytorch.org/t/batched-shuffling-of-feature-vectors/30188/4
    # https://github.com/pytorch/pytorch/issues/42502
    return torch.stack([torch.randperm(n, device=device) for i in range(n_batch)], dim=0)


# from https://github.com/runopti/stg/blob/9f630968c4f14cff6da4e54421c497f24ac1e08e/python/stg/layers.py#L10
def gauss_cdf(x):
    return 0.5 * (1 + torch.erf(x / np.sqrt(2)))


class ClampWithIdentityGradientFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, low: torch.Tensor, high: torch.Tensor):
        return torch.minimum(torch.maximum(input, low), high)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None, None


def clamp_with_identity_gradient_func(x, low, high):
    return ClampWithIdentityGradientFunc.apply(x, low, high)


def cat_if_necessary(tensors: List[torch.Tensor], dim: int):
    """
    Implements torch.cat() but doesn't copy if only one tensor is provided.
    This can make it faster if no copying behavior is needed.
    :param tensors: Tensors to be concatenated.
    :param dim: Dimension in which the tensor should be concatenated.
    :return: The concatenated tensor.
    """
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim=dim)


def hash_tensor(tensor: torch.Tensor) -> int:
    # for debugging purposes, to print two tensor's hashes to see if they are equal
    # from https://discuss.pytorch.org/t/defining-hash-function-for-multi-dimensional-tensor/107531
    import pickle
    # the .numpy() appears to be necessary for equal tensors to have equal hashes
    return hash(pickle.dumps(tensor.detach().cpu().numpy()))


def torch_np_quantile(tensor: torch.Tensor, q: float, dim: int, keepdim: bool = False) -> torch.Tensor:
    """
    Alternative implementation for torch.quantile() using np.quantile()
    since the implementation of torch.quantile() uses too much RAM (extreme for Airlines_DepDelay_10M)
    and can fail for too large tensors.
    See also https://github.com/pytorch/pytorch/issues/64947
    :param tensor: tensor
    :param q: Quantile value.
    :param dim: As in torch.quantile()
    :param keepdim: As in torch.quantile()
    :return: Tensor with quantiles.
    """
    x_np = tensor.detach().cpu().numpy()
    q_np = np.quantile(x_np, q=q, axis=dim, keepdims=keepdim)
    return torch.as_tensor(q_np, device=tensor.device, dtype=tensor.dtype)


from time import perf_counter
import torch


def _cuda_in_use() -> bool:
    """Return True if CUDA is available and initialized."""
    if not torch.cuda.is_available():
        return False
    # is_initialized exists in recent PyTorch; fall back to True if missing
    is_initialized = getattr(torch.cuda, "is_initialized", None)
    if is_initialized is None:
        return True
    return is_initialized()


class TorchTimer:
    """
    Timer for measuring code blocks, with optional CUDA synchronization.

    Usage:
        with TorchTimer() as t:
            y = model(x)
        print(t.elapsed)

        # Or manual start/stop:
        t = TorchTimer()
        t.start()
        y = model(x)
        t.stop()
        print(t.elapsed)
    """

    def __init__(self, use_cuda: Optional[bool] = None, record_history: bool = False):
        """
        Args:
            use_cuda:
                - None (default): auto-detect; sync only if CUDA is in use.
                - True: force CUDA sync (if available).
                - False: never sync CUDA.
            record_history:
                If True, every measurement is appended to `self.history`.
        """
        self._user_use_cuda = use_cuda
        self.record_history = record_history
        self.elapsed = None
        self.history = [] if record_history else None
        self._start = None

    @property
    def _do_cuda_sync(self) -> bool:
        if self._user_use_cuda is False:
            return False
        if self._user_use_cuda is True:
            return torch.cuda.is_available()
        # Auto mode: only if CUDA is available *and* initialized
        return _cuda_in_use()

    # ------- context manager API -------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ------- manual API -------

    def start(self):
        if self._do_cuda_sync:
            torch.cuda.synchronize()
        self._start = perf_counter()

    def stop(self):
        if self._start is None:
            raise RuntimeError("TorchTimer.stop() called before start().")
        if self._do_cuda_sync:
            torch.cuda.synchronize()
        self.elapsed = perf_counter() - self._start
        if self.record_history:
            self.history.append(self.elapsed)
        return self.elapsed


def get_available_memory_gb(device: Union[str, torch.device]) -> float:
    """
    Return the available memory (in GB) on the given device.

    Parameters
    ----------
    device : str or torch.device
        Device identifier, e.g. "cuda", "cuda:0", or torch.device("cuda:0").

    Returns
    -------
    float
        Available memory in gigabytes.

    Notes
    -----
    - For CUDA devices, this uses torch.cuda.mem_get_info if available.
    - For CPU, it uses psutil.virtual_memory().available.
    - For other device types, NotImplementedError is raised.
    """
    dev = torch.device(device)

    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but a CUDA device was requested.")

        # Ensure we are querying the correct device
        torch.cuda.synchronize(dev)

        if hasattr(torch.cuda, "mem_get_info"):
            free_bytes, total_bytes = torch.cuda.mem_get_info(dev)
        else:
            # Fallback: approximate using total_memory - reserved_by_pytorch
            props = torch.cuda.get_device_properties(dev)
            total_bytes = props.total_memory
            reserved_bytes = torch.cuda.memory_reserved(dev)
            free_bytes = max(total_bytes - reserved_bytes, 0)

        return free_bytes / (1024 ** 3)  # bytes -> GiB

    elif dev.type == "cpu":
        try:
            import psutil
        except ImportError as e:
            raise ImportError(
                "psutil is required to query CPU memory. Install via `pip install psutil`."
            ) from e

        mem = psutil.virtual_memory()
        return mem.available / (1024 ** 3)

    else:
        raise NotImplementedError(f"Memory query not implemented for device type '{dev.type}'")

