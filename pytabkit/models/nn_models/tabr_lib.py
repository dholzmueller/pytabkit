import math
import inspect
import warnings
import dataclasses
from typing import Any, Callable, Optional, Union, cast, Iterator, Iterable, List, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.nn.parameter import Parameter

# we copied this file from https://github.com/yandex-research/tabular-dl-tabr/blob/main/lib/deep.py
# to limit the number of dependencies

# ======================================================================================
# >>> modules <<<
# ======================================================================================
# When an instance of ModuleSpec is a dict,
# it must contain the key "type" with a string value
ModuleSpec = Union[str, dict[str, Any], Callable[..., nn.Module]]
T = TypeVar('T')


def _initialize_embeddings(weight: Tensor, d: Optional[int]) -> None:
    if d is None:
        d = weight.shape[-1]
    d_sqrt_inv = 1 / math.sqrt(d)
    nn.init.uniform_(weight, a=-d_sqrt_inv, b=d_sqrt_inv)


def make_trainable_vector(d: int) -> Parameter:
    x = torch.empty(d)
    _initialize_embeddings(x, None)
    return Parameter(x)


# class OneHotEncoder(nn.Module):
#     cardinalities: Tensor

#     def __init__(self, cardinalities: list[int]) -> None:
#         # cardinalities[i]`` is the number of unique values for the i-th categorical feature.
#         super().__init__()
#         self.register_buffer('cardinalities', torch.tensor(cardinalities))

#     def forward(self, x: Tensor) -> Tensor:
#         encoded_columns = [
#             F.one_hot(x[..., column], cardinality)
#             for column, cardinality in zip(range(x.shape[-1]), self.cardinalities)
#         ]

#         return torch.cat(encoded_columns, -1)
# This is modified to allow to encode unknown categories with zeros
class OneHotEncoder(nn.Module):
    cardinalities: torch.Tensor

    def __init__(self, cardinalities: list[int]) -> None:
        super().__init__()
        self.register_buffer('cardinalities', torch.tensor(cardinalities))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded_columns = []
        for column, cardinality in enumerate(self.cardinalities):
            column_values = x[..., column]
            # Replace -1 with a temporary valid index (e.g., 0)
            temp_index = torch.where(column_values == -1, 0, column_values)
            # Perform one-hot encoding
            one_hot = F.one_hot(temp_index, cardinality)
            # Zero out the vectors where original value was -1
            mask = column_values == -1
            one_hot[mask] = 0
            encoded_columns.append(one_hot)

        return torch.cat(encoded_columns, -1)


class CLSEmbedding(nn.Module):
    def __init__(self, d_embedding: int) -> None:
        super().__init__()
        self.weight = make_trainable_vector(d_embedding)

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 3
        assert x.shape[-1] == len(self.weight)
        return torch.cat([self.weight.expand(len(x), 1, -1), x], dim=1)


class CatEmbeddings(nn.Module):
    def __init__(
            self,
            _cardinalities_and_maybe_dimensions: Union[list[int], list[tuple[int, int]]],
            d_embedding: Optional[int] = None,
            *,
            stack: bool = False,
    ) -> None:
        assert _cardinalities_and_maybe_dimensions
        spec = _cardinalities_and_maybe_dimensions
        if not (
                (isinstance(spec[0], tuple) and d_embedding is None)
                or (isinstance(spec[0], int) and d_embedding is not None)
        ):
            raise ValueError(
                'Invalid arguments. Valid combinations are:'
                ' (1) the first argument is a list of (cardinality, embedding)-tuples AND d_embedding is None'
                ' (2) the first argument is a list of cardinalities AND d_embedding is an integer'
            )
        if stack and d_embedding is None:
            raise ValueError('stack can be True only when d_embedding is not None')

        super().__init__()
        spec_ = cast(
            list[tuple[int, int]],
            spec if d_embedding is None else [(x, d_embedding) for x in spec],
        )
        self._embeddings = nn.ModuleList()
        for cardinality, d_embedding in spec_:
            self._embeddings.append(nn.Embedding(cardinality, d_embedding))
        self.stack = stack
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self._embeddings:
            _initialize_embeddings(module.weight, None)  # type: ignore[code]

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2
        assert x.shape[1] == len(self._embeddings)
        out = [module(column) for module, column in zip(self._embeddings, x.T)]
        return torch.stack(out, dim=1) if self.stack else torch.cat(out, dim=1)


class LinearEmbeddings(nn.Module):
    def __init__(self, n_features: int, d_embedding: int, bias: bool = True):
        super().__init__()
        self.weight = Parameter(Tensor(n_features, d_embedding))
        self.bias = Parameter(Tensor(n_features, d_embedding)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in [self.weight, self.bias]:
            if parameter is not None:
                _initialize_embeddings(parameter, parameter.shape[-1])

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class PeriodicEmbeddings(nn.Module):
    def __init__(
            self, n_features: int, n_frequencies: int, frequency_scale: float
    ) -> None:
        super().__init__()
        self.frequencies = Parameter(
            torch.normal(0.0, frequency_scale, (n_features, n_frequencies))
        )

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2
        x = 2 * torch.pi * self.frequencies[None] * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)
        return x


class NLinear(nn.Module):
    def __init__(
            self, n_features: int, d_in: int, d_out: int, bias: bool = True
    ) -> None:
        super().__init__()
        self.weight = Parameter(Tensor(n_features, d_in, d_out))
        self.bias = Parameter(Tensor(n_features, d_out)) if bias else None
        with torch.no_grad():
            for i in range(n_features):
                layer = nn.Linear(d_in, d_out)
                self.weight[i] = layer.weight.T
                if self.bias is not None:
                    self.bias[i] = layer.bias

    def forward(self, x):
        assert x.ndim == 3
        x = x[..., None] * self.weight[None]
        x = x.sum(-2)
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class LREmbeddings(nn.Sequential):
    """The LR embeddings from the paper 'On Embeddings for Numerical Features in Tabular Deep Learning'."""  # noqa: E501

    def __init__(self, n_features: int, d_embedding: int) -> None:
        super().__init__(LinearEmbeddings(n_features, d_embedding), nn.ReLU())


class PLREmbeddings(nn.Sequential):
    """The PLR embeddings from the paper 'On Embeddings for Numerical Features in Tabular Deep Learning'.

    Additionally, the 'lite' option is added. Setting it to `False` gives you the original PLR
    embedding from the above paper. We noticed that `lite=True` makes the embeddings
    noticeably more lightweight without critical performance loss, and we used that for our model.
    """  # noqa: E501

    def __init__(
            self,
            n_features: int,
            n_frequencies: int,
            frequency_scale: float,
            d_embedding: int,
            lite: bool,
    ) -> None:
        super().__init__(
            PeriodicEmbeddings(n_features, n_frequencies, frequency_scale),
            (
                nn.Linear(2 * n_frequencies, d_embedding)
                if lite
                else NLinear(n_features, 2 * n_frequencies, d_embedding)
            ),
            nn.ReLU(),
        )


class PBLDEmbeddings(nn.Module):
    def __init__(self, n_features: int,
                 n_frequencies: int,
                 frequency_scale: float,
                 d_embedding: int,
                 plr_act_name: str = 'linear',
                 plr_use_densenet: bool = True):
        super().__init__()
        print(f'Constructing PBLD embeddings')
        hidden_2 = d_embedding-1 if plr_use_densenet else d_embedding
        self.weight_1 = nn.Parameter(frequency_scale * torch.randn(n_features, 1, n_frequencies))
        self.weight_2 = nn.Parameter((-1 + 2 * torch.rand(n_features, n_frequencies, hidden_2))
                / np.sqrt(n_frequencies))
        self.bias_1 = nn.Parameter(np.pi * (-1 + 2 * torch.rand(n_features, 1, n_frequencies)))
        self.bias_2 = nn.Parameter((-1 + 2 * torch.rand(n_features, 1, hidden_2)) / np.sqrt(n_frequencies))
        self.plr_act_name = plr_act_name
        self.plr_use_densenet = plr_use_densenet

    def forward(self, x):
        # transpose to treat the continuous feature dimension like a batched dimension
        # then add a new channel dimension
        # shape will be (vectorized..., n_cont, batch, 1)
        x_orig = x
        x = x.transpose(-1, -2).unsqueeze(-1)
        x = 2 * torch.pi * x.matmul(self.weight_1)  # matmul is automatically batched
        x = x + self.bias_1
        # x = torch.sin(x)
        x = torch.cos(x)
        x = x.matmul(self.weight_2)  # matmul is automatically batched
        x = x + self.bias_2
        if self.plr_act_name == 'relu':
            x = torch.relu(x)
        elif self.plr_act_name == 'linear':
            pass
        else:
            raise ValueError(f'Unknown plr_act_name "{self.plr_act_name}"')
        # bring back n_cont dimension after n_batch
        # then flatten the last two dimensions
        x = x.transpose(-2, -3)
        x = x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])
        if self.plr_use_densenet:
            x = torch.cat([x, x_orig], dim=-1)
        return x


class MLP(nn.Module):
    class Block(nn.Module):
        def __init__(
                self,
                *,
                d_in: int,
                d_out: int,
                bias: bool,
                activation: str,
                dropout: float,
        ) -> None:
            super().__init__()
            self.linear = nn.Linear(d_in, d_out, bias)
            self.activation = make_module(activation)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: Tensor) -> Tensor:
            return self.dropout(self.activation(self.linear(x)))

    Head = nn.Linear

    def __init__(
            self,
            *,
            d_in: int,
            d_out: Optional[int],
            n_blocks: int,
            d_layer: int,
            activation: str,
            dropout: float,
    ) -> None:
        assert n_blocks > 0
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                MLP.Block(
                    d_in=d_layer if block_i else d_in,
                    d_out=d_layer,
                    bias=True,
                    activation=activation,
                    dropout=dropout,
                )
                for block_i in range(n_blocks)
            ]
        )
        self.head = None if d_out is None else MLP.Head(d_layer, d_out)

    @property
    def d_out(self) -> int:
        return (
            self.blocks[-1].linear.out_features  # type: ignore[code]
            if self.head is None
            else self.head.out_features
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        if self.head is not None:
            x = self.head(x)
        return x


_CUSTOM_MODULES = {
    x.__name__: x
    for x in [
        LinearEmbeddings,
        LREmbeddings,
        PLREmbeddings,
        PBLDEmbeddings,
        MLP,
    ]
}


def register_module(key: str, f: Callable[..., nn.Module]) -> None:
    assert key not in _CUSTOM_MODULES
    _CUSTOM_MODULES[key] = f


def make_module(spec: ModuleSpec, *args, **kwargs) -> nn.Module:
    """
    >>> make_module('ReLU')
    >>> make_module(nn.ReLU)
    >>> make_module('Linear', 1, out_features=2)
    >>> make_module((lambda *args: nn.Linear(*args)), 1, out_features=2)
    >>> make_module({'type': 'Linear', 'in_features' 1}, out_features=2)
    """
    if isinstance(spec, str):
        Module = getattr(nn, spec, None)
        if Module is None:
            Module = _CUSTOM_MODULES[spec]
        else:
            assert spec not in _CUSTOM_MODULES
        return make_module(Module, *args, **kwargs)
    elif isinstance(spec, dict):
        assert not (set(spec) & set(kwargs))
        spec = spec.copy()
        return make_module(spec.pop('type'), *args, **spec, **kwargs)
    elif callable(spec):
        return spec(*args, **kwargs)
    else:
        raise ValueError()


def get_n_parameters(m: nn.Module):
    return sum(x.numel() for x in m.parameters() if x.requires_grad)


def get_d_out(n_classes: Optional[int]) -> int:
    return 1 if n_classes is None or n_classes == 2 else n_classes


# ======================================================================================
# >>> optimization <<<
# ======================================================================================
def default_zero_weight_decay_condition(
        module_name: str, module: nn.Module, parameter_name: str, parameter: Parameter
):
    del module_name, parameter
    return parameter_name.endswith('bias') or isinstance(
        module,
        (
            nn.BatchNorm1d,
            nn.LayerNorm,
            nn.InstanceNorm1d,
            LinearEmbeddings,
            PeriodicEmbeddings,
        ),
    )


def make_parameter_groups(
        model: nn.Module,
        zero_weight_decay_condition,
        custom_groups: dict[tuple[str], dict],  # [(fullnames, options), ...]
) -> list[dict[str, Any]]:
    custom_fullnames = set()
    custom_fullnames.update(*custom_groups)
    assert sum(map(len, custom_groups)) == len(
        custom_fullnames
    ), 'Custom parameter groups must not intersect'

    parameters_info = {}  # fullname -> (parameter, needs_wd)
    for module_name, module in model.named_modules():
        for name, parameter in module.named_parameters():
            fullname = f'{module_name}.{name}' if module_name else name
            parameters_info.setdefault(fullname, (parameter, []))[1].append(
                not zero_weight_decay_condition(module_name, module, name, parameter)
            )
    parameters_info = {k: (v[0], all(v[1])) for k, v in parameters_info.items()}

    params_with_wd = {'params': []}
    params_without_wd = {'params': [], 'weight_decay': 0.0}
    custom_params = {k: {'params': []} | v for k, v in custom_groups.items()}

    for fullname, (parameter, needs_wd) in parameters_info.items():
        for fullnames, group in custom_params.items():
            if fullname in fullnames:
                custom_fullnames.remove(fullname)
                group['params'].append(parameter)
                break
        else:
            (params_with_wd if needs_wd else params_with_wd)['params'].append(parameter)
    assert (
        not custom_fullnames
    ), f'Some of the custom parameters were not found in the model: {custom_fullnames}'
    return [params_with_wd, params_without_wd] + list(custom_params.values())


def make_optimizer(
        module: nn.Module,
        type: str,
        *,
        zero_weight_decay_condition=default_zero_weight_decay_condition,
        custom_parameter_groups: Optional[dict[tuple[str], dict]] = None,
        **optimizer_kwargs,
) -> torch.optim.Optimizer:
    if custom_parameter_groups is None:
        custom_parameter_groups = {}
    Optimizer = getattr(optim, type)
    parameter_groups = make_parameter_groups(
        module, zero_weight_decay_condition, custom_parameter_groups
    )
    print(f'{optimizer_kwargs=}')
    return Optimizer(parameter_groups, **optimizer_kwargs)


def get_lr(optimizer: optim.Optimizer) -> float:
    return next(iter(optimizer.param_groups))['lr']


def set_lr(optimizer: optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group['lr'] = lr


## We also package useful delu functions to limit the number of dependencies
# copied from https://github.com/Yura52/delu/blob/5f0015cbdff86f64aff8199123012a9663538fcf/delu/nn.py
class Lambda(torch.nn.Module):
    """A wrapper for functions from `torch` and methods of `torch.Tensor`.

    An important "feature" of this module is that it is intentionally limited:

    - Only the functions from the `torch` module and the methods of `torch.Tensor`
      are allowed.
    - The passed callable must accept a single `torch.Tensor`
      and return a single `torch.Tensor`.
    - The allowed keyword arguments must be of simple types (see the docstring).

    **Usage**

    >>> m = delu.nn.Lambda(torch.squeeze)
    >>> m(torch.randn(2, 1, 3, 1)).shape
    torch.Size([2, 3])
    >>> m = delu.nn.Lambda(torch.squeeze, dim=1)
    >>> m(torch.randn(2, 1, 3, 1)).shape
    torch.Size([2, 3, 1])
    >>> m = delu.nn.Lambda(torch.Tensor.abs_)
    >>> m(torch.tensor(-1.0))
    tensor(1.)

    Custom functions are not allowed
    (technically, they are **temporarily** allowed,
    but this functionality is deprecated and will be removed in future releases):

    >>> # xdoctest: +SKIP
    >>> m = delu.nn.Lambda(lambda x: torch.abs(x))
    Traceback (most recent call last):
        ...
    ValueError: fn must be a function from `torch` or a method of `torch.Tensor`, but ...

    Non-trivial keyword arguments are not allowed:

    >>> m = delu.nn.Lambda(torch.mul, other=torch.tensor(2.0))
    Traceback (most recent call last):
        ...
    ValueError: For kwargs, the allowed value types include: ...
    """  # noqa: E501

    def __init__(self, fn: Callable[..., torch.Tensor], /, **kwargs) -> None:
        """
        Args:
            fn: the callable.
            kwargs: the keyword arguments for ``fn``. The allowed values types include:
                None, bool, int, float, bytes, str
                and (nested) tuples of these simple types.
        """
        super().__init__()
        if not callable(fn) or (
                fn not in vars(torch).values()
                and (
                        fn not in (member for _, member in inspect.getmembers(torch.Tensor))
                        or inspect.ismethod(fn)  # Check if fn is a @classmethod
                )
        ):
            warnings.warn(
                'Passing custom functions to delu.nn.Lambda is deprecated'
                ' and will be removed in future releases.'
                ' Only functions from the `torch` module and methods of `torch.Tensor`'
                ' are allowed',
                DeprecationWarning,
            )
            # NOTE: in future releases, replace the above warning with this exception:
            # raise ValueError(
            #     'fn must be a function from `torch` or a method of `torch.Tensor`,'
            #     f' but this is not true for the passed {fn=}'
            # )

        def is_valid_value(x):
            return (
                    x is None
                    or isinstance(x, (bool, int, float, bytes, str))
                    or isinstance(x, tuple)
                    and all(map(is_valid_value, x))
            )

        for k, v in kwargs.items():
            if not is_valid_value(v):
                raise ValueError(
                    'For kwargs, the allowed value types include:'
                    ' None, bool, int, float, bytes, str and (nested) tuples containing'
                    ' values of these simple types. This is not true for the passed'
                    f' argument {k} with the value {v}'
                )

        self._function = fn
        self._function_kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do the forward pass."""
        return self._function(x, **self._function_kwargs)


# copied from https://github.com/Yura52/delu/blob/5f0015cbdff86f64aff8199123012a9663538fcf/delu/_tensor_ops.py#L339

def _make_index_batches(
        x: torch.Tensor,
        batch_size: int,
        shuffle: bool,
        generator: Optional[torch.Generator],
        drop_last: bool,
) -> Iterable[torch.Tensor]:
    size = len(x)
    if not size:
        raise ValueError('data must not contain empty tensors')
    batch_indices = (
        torch.randperm(size, generator=generator, device=x.device)
        if shuffle
        else torch.arange(size, device=x.device)
    ).split(batch_size)
    return (
        batch_indices[:-1]
        if batch_indices and drop_last and len(batch_indices[-1]) < batch_size
        else batch_indices
    )


def iter_batches(
        data: T,
        /,
        batch_size: int,
        *,
        shuffle: bool = False,
        generator: Optional[torch.Generator] = None,
        drop_last: bool = False,
) -> Iterator[T]:
    """Iterate over a tensor or a collection of tensors by (random) batches.

    The function makes batches along the first dimension of the tensors in ``data``.

    TL;DR (assuming that ``X`` and ``Y`` denote full tensors
    and ``xi`` and ``yi`` denote batches):

    - ``delu.iter_batches: X -> [x1, x2, ..., xN]``
    - ``delu.iter_batches: (X, Y) -> [(x1, y1), (x2, y2), ..., (xN, yN)]``
    - ``delu.iter_batches: {'x': X, 'y': Y} -> [{'x': x1, 'y': y1}, ...]``
    - Same for named tuples.
    - Same for dataclasses.

    .. note::
        `delu.iter_batches` is significantly faster for in-memory tensors
        than `torch.utils.data.DataLoader`, because, when building batches,
        it uses batched indexing instead of one-by-one indexing.

    **Usage**

    >>> X = torch.randn(12, 32)
    >>> Y = torch.randn(12)

    `delu.iter_batches` can be applied to tensors:

    >>> for x in delu.iter_batches(X, batch_size=5):
    ...     print(len(x))
    5
    5
    2

    `delu.iter_batches` can be applied to tuples:

    >>> # shuffle=True can be useful for training.
    >>> dataset = (X, Y)
    >>> for x, y in delu.iter_batches(dataset, batch_size=5, shuffle=True):
    ...     print(len(x), len(y))
    5 5
    5 5
    2 2
    >>> # Drop the last incomplete batch.
    >>> for x, y in delu.iter_batches(
    ...     dataset, batch_size=5, shuffle=True, drop_last=True
    ... ):
    ...     print(len(x), len(y))
    5 5
    5 5
    >>> # The last batch is complete, so drop_last=True does not have any effect.
    >>> batches = []
    >>> for x, y in delu.iter_batches(dataset, batch_size=6, drop_last=True):
    ...     print(len(x), len(y))
    ...     batches.append((x, y))
    6 6
    6 6

    By default, ``shuffle`` is set to `False`, i.e. the order of items is preserved:

    >>> X2, Y2 = delu.cat(list(delu.iter_batches((X, Y), batch_size=5)))
    >>> print((X == X2).all().item(), (Y == Y2).all().item())
    True True

    `delu.iter_batches` can be applied to dictionaries:

    >>> dataset = {'x': X, 'y': Y}
    >>> for batch in delu.iter_batches(dataset, batch_size=5, shuffle=True):
    ...     print(isinstance(batch, dict), len(batch['x']), len(batch['y']))
    True 5 5
    True 5 5
    True 2 2

    `delu.iter_batches` can be applied to named tuples:

    >>> from typing import NamedTuple
    >>> class Data(NamedTuple):
    ...     x: torch.Tensor
    ...     y: torch.Tensor
    >>> dataset = Data(X, Y)
    >>> for batch in delu.iter_batches(dataset, batch_size=5, shuffle=True):
    ...     print(isinstance(batch, Data), len(batch.x), len(batch.y))
    True 5 5
    True 5 5
    True 2 2

    `delu.iter_batches` can be applied to dataclasses:

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Data:
    ...     x: torch.Tensor
    ...     y: torch.Tensor
    >>> dataset = Data(X, Y)
    >>> for batch in delu.iter_batches(dataset, batch_size=5, shuffle=True):
    ...     print(isinstance(batch, Data), len(batch.x), len(batch.y))
    True 5 5
    True 5 5
    True 2 2

    Args:
        data: the tensor or the non-empty collection of tensors.
            If data is a collection, then the tensors must be of the same size
            along the first dimension.
        batch_size: the batch size. If ``drop_last`` is False,
            then the last batch can be smaller than ``batch_size``.
        shuffle: if True, iterate over random batches (without replacement),
            not sequentially.
        generator: when ``shuffle`` is True, passing ``generator`` makes the function
            reproducible.
        drop_last: when ``True`` and the last batch is smaller then ``batch_size``,
            then this last batch is not returned
            (in other words,
            same as the ``drop_last`` argument for `torch.utils.data.DataLoader`).
    Returns:
        the iterator over batches.
    """
    if not shuffle and generator is not None:
        raise ValueError('When shuffle is False, generator must be None.')

    constructor: Callable[[Any], T]
    args = (batch_size, shuffle, generator, drop_last)

    if isinstance(data, torch.Tensor):
        item = data
        for idx in _make_index_batches(item, *args):
            yield data[idx]  # type: ignore

    elif isinstance(data, tuple):
        if not data:
            raise ValueError('data must be non-empty')
        item = data[0]
        for x in data:
            if not isinstance(x, torch.Tensor) or len(x) != len(item):
                raise ValueError(
                    'If data is a tuple, it must contain only tensors,'
                    ' and they must have the same first dimension'
                )
        constructor = type(data)  # type: ignore
        constructor = getattr(constructor, '_make', constructor)  # Handle named tuples.
        for idx in _make_index_batches(item, *args):
            yield constructor(x[idx] for x in data)

    elif isinstance(data, dict):
        if not data:
            raise ValueError('data must be non-empty')
        item = next(iter(data.values()))
        for x in data.values():
            if not isinstance(x, torch.Tensor) or len(x) != len(item):
                raise ValueError(
                    'If data is a dict, it must contain only tensors,'
                    ' and they must have the same first dimension'
                )
        constructor = type(data)  # type: ignore
        for idx in _make_index_batches(item, *args):
            yield constructor((k, v[idx]) for k, v in data.items())

    elif dataclasses.is_dataclass(data):
        fields = list(dataclasses.fields(data))
        if not fields:
            raise ValueError('data must be non-empty')
        item = getattr(data, fields[0].name)
        for field in fields:
            if field.type is not torch.Tensor:
                raise ValueError('All dataclass fields must be tensors.')
            if len(getattr(data, field.name)) != len(item):
                raise ValueError(
                    'All dataclass tensors must have the same first dimension.'
                )
        constructor = type(data)  # type: ignore
        for idx in _make_index_batches(item, *args):
            yield constructor(
                **{field.name: getattr(data, field.name)[idx] for field in fields}  # type: ignore
            )

    else:
        raise ValueError(f'The collection {type(data)} is not supported.')


def cat(data: List[T], /, dim: int = 0) -> T:
    """Concatenate a sequence of collections of tensors.

    `delu.cat` is a generalized version of `torch.cat` for concatenating
    not only tensors, but also (nested) collections of tensors.

    **Usage**

    Let's see how a sequence of model outputs for batches can be concatenated
    into a output tuple for the whole dataset:

    >>> from torch.utils.data import DataLoader, TensorDataset
    >>> dataset = TensorDataset(torch.randn(320, 24))
    >>> batch_size = 32
    >>>
    >>> # The model returns not only predictions, but also embeddings.
    >>> def model(x_batch):
    ...     # A dummy forward pass.
    ...     embeddings_batch = torch.randn(batch_size, 16)
    ...     y_pred_batch = torch.randn(batch_size)
    ...     return (y_pred_batch, embeddings_batch)
    ...
    >>> y_pred, embeddings = delu.cat(
    ...     [model(batch) for batch in DataLoader(dataset, batch_size, shuffle=True)]
    ... )
    >>> len(y_pred) == len(dataset)
    True
    >>> len(embeddings) == len(dataset)
    True

    The same works for dictionaries:

    >>> def model(x_batch):
    ...     return {
    ...         'y_pred': torch.randn(batch_size),
    ...         'embeddings': torch.randn(batch_size, 16)
    ...     }
    ...
    >>> outputs = delu.cat(
    ...     [model(batch) for batch in DataLoader(dataset, batch_size, shuffle=True)]
    ... )
    >>> len(outputs['y_pred']) == len(dataset)
    True
    >>> len(outputs['embeddings']) == len(dataset)
    True

    The same works for sequences of named tuples, dataclasses, tensors and
    nested combinations of all mentioned collection types.

    *Below, additinal technical examples are provided.*

    The common setup:

    >>> # First batch.
    >>> x1 = torch.randn(64, 10)
    >>> y1 = torch.randn(64)
    >>> # Second batch.
    >>> x2 = torch.randn(64, 10)
    >>> y2 = torch.randn(64)
    >>> # The last (incomplete) batch.
    >>> x3 = torch.randn(7, 10)
    >>> y3 = torch.randn(7)
    >>> total_size = len(x1) + len(x2) + len(x3)

    `delu.cat` can be applied to tuples:

    >>> batches = [(x1, y1), (x2, y2), (x3, y3)]
    >>> X, Y = delu.cat(batches)
    >>> len(X) == total_size and len(Y) == total_size
    True

    `delu.cat` can be applied to dictionaries:

    >>> batches = [
    ...     {'x': x1, 'y': y1},
    ...     {'x': x2, 'y': y2},
    ...     {'x': x3, 'y': y3},
    ... ]
    >>> result = delu.cat(batches)
    >>> isinstance(result, dict)
    True
    >>> len(result['x']) == total_size and len(result['y']) == total_size
    True

    `delu.cat` can be applied to named tuples:

    >>> from typing import NamedTuple
    >>> class Data(NamedTuple):
    ...     x: torch.Tensor
    ...     y: torch.Tensor
    ...
    >>> batches = [Data(x1, y1), Data(x2, y2), Data(x3, y3)]
    >>> result = delu.cat(batches)
    >>> isinstance(result, Data)
    True
    >>> len(result.x) == total_size and len(result.y) == total_size
    True

    `delu.cat` can be applied to dataclasses:

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Data:
    ...     x: torch.Tensor
    ...     y: torch.Tensor
    ...
    >>> batches = [Data(x1, y1), Data(x2, y2), Data(x3, y3)]
    >>> result = delu.cat(batches)
    >>> isinstance(result, Data)
    True
    >>> len(result.x) == total_size and len(result.y) == total_size
    True

    `delu.cat` can be applied to nested collections:

    >>> batches = [
    ...     (x1, {'a': {'b': y1}}),
    ...     (x2, {'a': {'b': y2}}),
    ...     (x3, {'a': {'b': y3}}),
    ... ]
    >>> X, Y_nested = delu.cat(batches)
    >>> len(X) == total_size and len(Y_nested['a']['b']) == total_size
    True

    **Lists are not supported:**

    >>> # This does not work. Instead, use tuples.
    >>> # batches = [[x1, y1], [x2, y2], [x3, y3]]
    >>> # delu.cat(batches)  # Error

    Args:
        data: the list of collections of tensors.
            All items of the list must be of the same type, structure and layout, only
            the ``dim`` dimension can vary (same as for `torch.cat`).
            All the "leaf" values must be of the type `torch.Tensor`.
        dim: the dimension along which the tensors are concatenated.
    Returns:
        The concatenated items of the list.
    """
    if not isinstance(data, list):
        raise ValueError('The input must be a list')
    if not data:
        raise ValueError('The input must be non-empty')

    first = data[0]

    if isinstance(first, torch.Tensor):
        return torch.cat(data, dim=dim)  # type: ignore

    elif isinstance(first, tuple):
        constructor = type(first)
        constructor = getattr(constructor, '_make', constructor)  # Handle named tuples.
        return constructor(
            cat([x[i] for x in data], dim=dim) for i in range(len(first))  # type: ignore
        )

    elif isinstance(first, dict):
        return type(first)((key, cat([x[key] for x in data], dim=dim)) for key in first)  # type: ignore

    elif dataclasses.is_dataclass(first):
        return type(first)(
            **{
                field.name: cat([getattr(x, field.name) for x in data], dim=dim)
                for field in dataclasses.fields(first)
            }
        )  # type: ignore

    else:
        raise ValueError(f'The collection type {type(first)} is not supported.')


def is_oom_exception(err: RuntimeError) -> bool:
    return isinstance(err, torch.cuda.OutOfMemoryError) or any(
        x in str(err)
        for x in [
            'CUDA out of memory',
            'CUBLAS_STATUS_ALLOC_FAILED',
            'CUDA error: out of memory',
        ]
    )
