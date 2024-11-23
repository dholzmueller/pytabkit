import math
import numbers
import typing as ty

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor
import skorch
import numpy as np
import pandas as pd
import torch.nn as nn
from skorch.callbacks import Checkpoint, EarlyStopping, LRScheduler, PrintLog
from skorch import NeuralNetRegressor, NeuralNetClassifier
from skorch.dataset import Dataset
from skorch.callbacks import EpochScoring
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW, Adam, SGD
from skorch.callbacks import WandbLogger
# import sys
# sys.path.append("")
from skorch.callbacks import Callback, Checkpoint
import numpy as np
import os
from functools import partial
from copy import deepcopy
from .rtdl_num_embeddings import PeriodicEmbeddings


# code adapted from https://github.com/yandex-research/rtdl/tree/e5dac7f1bb33078699f5079ce301dc907c5b512a/bin

def reglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


def get_activation_fn(name: str) -> ty.Callable[[Tensor], Tensor]:
    return (
        reglu
        if name == 'reglu'
        else geglu
        if name == 'geglu'
        else torch.sigmoid
        if name == 'sigmoid'
        else getattr(F, name)
    )


def get_nonglu_activation_fn(name: str) -> ty.Callable[[Tensor], Tensor]:
    return (
        F.relu
        if name == 'reglu'
        else F.gelu
        if name == 'geglu'
        else get_activation_fn(name)
    )


def print_but_serializable(*args, **kwargs):
    # this is a dummy function to prevent an obscure error in pickling skorch objects
    # containing callbacks with sink=print
    # The error occurs when ray.init() and FunctionProcess() are both used. Error message:
    # _pickle.PicklingError: Can't pickle <built-in function print>: it's not the same object as builtins.print
    print(*args, **kwargs)


class RTDL_MLP(nn.Module):
    # baseline MLP
    def __init__(
            self,
            *,
            d_in: int,
            n_layers: int,
            d_layers: ty.Union[int, ty.List[int]],
            d_first_layer: int,
            d_last_layer: int,
            dropout: float,
            d_out: int,
            categories: ty.Optional[ty.List[int]],
            d_embedding: int,
            regression: bool,
            categorical_indicator,
            num_emb_type: str = 'none',
            num_emb_dim: int = 24,
            num_emb_hidden_dim: int = 48,
            num_emb_sigma: float = 0.01,
            num_emb_lite: bool = False
    ) -> None:
        super().__init__()

        self.regression = regression
        self.categorical_indicator = categorical_indicator  # Added
        self.categories = categories  # Added

        if num_emb_type == 'none':
            self.num_emb_layer = nn.Identity()
        elif num_emb_type == 'plr':
            self.num_emb_layer = nn.Sequential(PeriodicEmbeddings(d_in, num_emb_dim,
                                                                  n_frequencies=num_emb_hidden_dim,
                                                                  frequency_init_scale=num_emb_sigma,
                                                                  lite=num_emb_lite), nn.Flatten())
            d_in = d_in * num_emb_dim
        elif num_emb_type == 'pl':
            self.num_emb_layer = nn.Sequential(PeriodicEmbeddings(d_in, num_emb_dim,
                                                                  n_frequencies=num_emb_hidden_dim,
                                                                  frequency_init_scale=num_emb_sigma,
                                                                  activation=False,
                                                                  lite=num_emb_lite), nn.Flatten())
            d_in = d_in * num_emb_dim
        else:
            raise ValueError(f'Unknown numerical embedding type "{num_emb_type}"')

        if categories is not None and len(categories) > 0:
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor(
                np.concatenate([[0],
                                np.array(categories[:-1], dtype=np.int64)
                                ])
            ).cumsum(0)
            self.register_buffer("category_offsets", category_offsets)
            self.category_embeddings = nn.Embedding(int(sum(categories)), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            # set the embedding of the last category of each feature to zero
            # it represents the "missing" category, i.e. the categories that is not present
            # in the training set
            for i, c in enumerate(categories):
                self.category_embeddings.weight.data[
                    category_offsets[i] + c - 1
                    ].zero_()

        if isinstance(d_layers, numbers.Number):
            d_layers = [d_first_layer] + [d_layers for _ in range(n_layers)] + [d_last_layer]  # CHANGED
        else:
            assert len(d_layers) == n_layers

        self.layers = nn.ModuleList(
            [
                nn.Linear(d_layers[i - 1] if i else d_in, x)
                for i, x in enumerate(d_layers)
            ]
        )
        self.dropout = dropout
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)

    def forward(self, x):

        if not self.categorical_indicator is None:
            x_num = x[:, ~self.categorical_indicator].float()
            x_cat = x[:, self.categorical_indicator].long()
        else:
            x_num = x
            x_cat = None

        # Added: Numerical embeddings
        x_num = self.num_emb_layer(x_num)
        x = []
        if x_num is not None:
            x.append(x_num)
        if x_cat is not None:
            # replace -1 by the last category
            for i in range(x_cat.shape[1]):
                x_cat[:, i][x_cat[:, i] == -1] = self.categories[i] - 1
            x.append(
                self.category_embeddings(x_cat + self.category_offsets[None]).view(
                    x_cat.size(0), -1
                )
            )
        x = torch.cat(x, dim=-1)

        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, self.dropout, self.training)
        x = self.head(x)
        if not self.regression:
            x = x.squeeze(-1)
        return x


class ResNet(nn.Module):
    def __init__(
            self,
            *,
            d_in: int,
            categories: ty.Optional[ty.List[int]],
            d_embedding: int,
            d: int,
            d_hidden_factor: float,
            n_layers: int,
            activation: str,
            normalization: str,
            hidden_dropout: float,
            residual_dropout: float,
            d_out: int,
            regression: bool,
            categorical_indicator
    ) -> None:
        super().__init__()

        def make_normalization():
            return {"batchnorm": nn.BatchNorm1d, "layernorm": nn.LayerNorm}[
                normalization
            ](d)

        self.categorical_indicator = categorical_indicator  # Added
        self.regression = regression
        self.main_activation = get_activation_fn(activation)
        self.last_activation = get_nonglu_activation_fn(activation)
        self.residual_dropout = residual_dropout
        self.hidden_dropout = hidden_dropout

        d_hidden = int(d * d_hidden_factor)
        self.categories = categories
        if categories is not None and len(categories) > 0:
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor(
                np.concatenate([[0],
                                np.array(categories[:-1], dtype=np.int64)
                                ])
            ).cumsum(0)
            self.register_buffer("category_offsets", category_offsets)
            self.category_embeddings = nn.Embedding(int(sum(categories)), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            # set the embedding of the last category of each feature to zero
            # it represents the "missing" category, i.e. the categories that is not present
            # in the training set
            for i, c in enumerate(categories):
                self.category_embeddings.weight.data[
                    category_offsets[i] + c - 1
                    ].zero_()

        self.first_layer = nn.Linear(d_in, d)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": make_normalization(),
                        "linear0": nn.Linear(
                            d, d_hidden * (2 if activation.endswith("glu") else 1)
                        ),
                        "linear1": nn.Linear(d_hidden, d),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.last_normalization = make_normalization()
        self.head = nn.Linear(d, d_out)

    def forward(self, x) -> Tensor:
        if not self.categorical_indicator is None:
            x_num = x[:, ~self.categorical_indicator].float()
            x_cat = x[:, self.categorical_indicator].long()
        else:
            x_num = x
            x_cat = None
        x = []
        if x_num is not None and x_num.numel() > 0:
            x.append(x_num)
        if x_cat is not None and x_cat.numel() > 0:
            # replace -1 by the last category
            for i in range(x_cat.shape[1]):
                x_cat[:, i][x_cat[:, i] == -1] = self.categories[i] - 1
            x.append(
                self.category_embeddings(x_cat + self.category_offsets[None]).view(
                    x_cat.size(0), -1
                )
            )
        x = torch.cat(x, dim=-1)

        x = self.first_layer(x)
        for layer in self.layers:
            layer = ty.cast(ty.Dict[str, nn.Module], layer)
            z = x
            z = layer["norm"](z)
            z = layer["linear0"](z)
            z = self.main_activation(z)
            if self.hidden_dropout:
                z = F.dropout(z, self.hidden_dropout, self.training)
            z = layer["linear1"](z)
            if self.residual_dropout:
                z = F.dropout(z, self.residual_dropout, self.training)
            x = x + z
        x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        if not self.regression:
            x = x.squeeze(-1)
        return x
    
class Tokenizer(nn.Module):
    category_offsets: ty.Optional[Tensor]

    def __init__(
        self,
        d_numerical: int,
        categories: ty.Optional[ty.List[int]],
        d_token: int,
        bias: bool,
    ) -> None:
        #categories = None
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')
            # set the embedding of the last category of each feature to zero
            # it represents the "missing" category, i.e. the categories that is not present
            # in the training set
            for i, c in enumerate(categories):
                self.category_embeddings.weight.data[
                    category_offsets[i] + c - 1
                    ].zero_()
            

        # take [CLS] token into account
        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        # The initialization is inspired by nn.Linear
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))
        self.categories = categories

    @property
    def n_tokens(self) -> int:
        return len(self.weight) + (
            0 if self.category_offsets is None else len(self.category_offsets)
        )

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)]  # [CLS]
            + ([] if x_num is None else [x_num]),
            dim=1,
        )
        x = self.weight[None] * x_num[:, :, None]
        if x_cat is not None:
            # replace -1 by the last category
            for i in range(x_cat.shape[1]):
                x_cat[:, i][x_cat[:, i] == -1] = self.categories[i] - 1
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]
        return x


class MultiheadAttention(nn.Module):
    def __init__(
        self, d: int, n_heads: int, dropout: float, initialization: str
    ) -> None:
        if n_heads > 1:
            assert d % n_heads == 0
        assert initialization in ['xavier', 'kaiming']

        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                # gain is needed since W_qkv is represented with 3 separate layers
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: ty.Optional[nn.Linear],
        value_compression: ty.Optional[nn.Linear],
    ) -> Tensor:
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0
        if key_compression is not None:
            assert value_compression is not None
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)
        else:
            assert value_compression is None

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention = F.softmax(q @ k.transpose(1, 2) / math.sqrt(d_head_key), dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x


class FT_Transformer(nn.Module):
    """Transformer.

    References:
    - https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
    - https://github.com/facebookresearch/pytext/tree/master/pytext/models/representations/transformer
    - https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/examples/linformer/linformer_src/modules/multihead_linear_attention.py#L19
    """

    def __init__(
        self,
        *,
        # tokenizer
        d_in: int, #changed name
        categories: ty.Optional[ty.List[int]],
        token_bias: bool,
        # transformer
        n_layers: int,
        d_token: int,
        n_heads: int,
        d_ffn_factor: float,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        activation: str,
        prenormalization: bool,
        initialization: str,
        # linformer
        kv_compression: ty.Optional[float],
        kv_compression_sharing: ty.Optional[str],
        #
        d_out: int,
        regression: bool,
        categorical_indicator
    ) -> None:
        assert (kv_compression is None) ^ (kv_compression_sharing is not None)
        super().__init__()
        self.tokenizer = Tokenizer(d_in, categories, d_token, token_bias)
        n_tokens = self.tokenizer.n_tokens
        # print("d_token {}".format(d_token))

        self.categorical_indicator = categorical_indicator
        self.regression = regression

        def make_kv_compression():
            assert kv_compression
            compression = nn.Linear(
                n_tokens, int(n_tokens * kv_compression), bias=False
            )
            if initialization == 'xavier':
                nn_init.xavier_uniform_(compression.weight)
            return compression

        self.shared_kv_compression = (
            make_kv_compression()
            if kv_compression and kv_compression_sharing == 'layerwise'
            else None
        )

        def make_normalization():
            return nn.LayerNorm(d_token)

        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList([])
        for layer_idx in range(n_layers):
            layer = nn.ModuleDict(
                {
                    'attention': MultiheadAttention(
                        d_token, n_heads, attention_dropout, initialization
                    ),
                    'linear0': nn.Linear(
                        d_token, d_hidden * (2 if activation.endswith('glu') else 1)
                    ),
                    'linear1': nn.Linear(d_hidden, d_token),
                    'norm1': make_normalization(),
                }
            )
            if not prenormalization or layer_idx:
                layer['norm0'] = make_normalization()
            if kv_compression and self.shared_kv_compression is None:
                layer['key_compression'] = make_kv_compression()
                if kv_compression_sharing == 'headwise':
                    layer['value_compression'] = make_kv_compression()
                else:
                    assert kv_compression_sharing == 'key-value'
            self.layers.append(layer)

        self.activation = get_activation_fn(activation)
        self.last_activation = get_nonglu_activation_fn(activation)
        self.prenormalization = prenormalization
        self.last_normalization = make_normalization() if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)

    def _get_kv_compressions(self, layer):
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer['key_compression'], layer['value_compression'])
            if 'key_compression' in layer and 'value_compression' in layer
            else (layer['key_compression'], layer['key_compression'])
            if 'key_compression' in layer
            else (None, None)
        )

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x) -> Tensor:
        if not self.categorical_indicator is None:
            x_num = x[:, ~self.categorical_indicator].float()
            x_cat = x[:, self.categorical_indicator].long() #TODO
        else:
            x_num = x
            x_cat = None
        #x_cat = None #FIXME
        x = self.tokenizer(x_num, x_cat)

        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx + 1 == len(self.layers)
            layer = ty.cast(ty.Dict[str, nn.Module], layer)

            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](
                # for the last attention, it is enough to process only [CLS]
                (x_residual[:, :1] if is_last_layer else x_residual),
                x_residual,
                *self._get_kv_compressions(layer),
            )
            if is_last_layer:
                x = x[:, : x_residual.shape[1]]
            x = self._end_residual(x, x_residual, layer, 0)

            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)

        assert x.shape[1] == 1
        x = x[:, 0]
        if self.last_normalization is not None:
            x = self.last_normalization(x)
        x = self.last_activation(x)
        x = self.head(x)
        if not self.regression:
            x = x.squeeze(-1)
        return x


class InputShapeSetterResnet(skorch.callbacks.Callback):
    def __init__(
            self, regression=False, batch_size=None, cat_features=None, categories=None
    ):
        self.cat_features = cat_features
        self.regression = regression
        self.batch_size = batch_size
        self.categories = categories

    def on_train_begin(self, net, X, y):
        if net.categorical_indicator is None:
            if self.cat_features is not None:
                # TODO: it's redundant
                net.set_categorical_indicator(
                    np.array([i in self.cat_features for i in range(X.shape[1])])
                )
            else:
                d_in = X.shape[1]
                categories = None
        else:
            d_in = X.shape[1] - sum(net.categorical_indicator)
            if self.categories is None:
                categories = [
                    # +1 for the unknown category
                    len(set(X[:, i])) + 1 for i in np.where(net.categorical_indicator)[0]
                ]
            else:
                categories = self.categories
        if self.regression:
            d_out = 1
        else:
            if hasattr(net, "n_classes"):
                d_out = net.n_classes
            else:
                assert y.max() + 1 == len(set(y))
                d_out = int(y.max() + 1)

        net.set_params(
            module__d_in=d_in,
            module__categories=categories,  # FIXME #lib.get_categories(X_cat),
            module__categorical_indicator=torch.BoolTensor(net.categorical_indicator)
            if net.categorical_indicator is not None
            else None,
            module__d_out=d_out,
        )


class LearningRateLogger(Callback):
    def on_epoch_begin(self, net, dataset_train=None, dataset_valid=None, **kwargs):
        callbacks = net.callbacks
        for callback in callbacks:
            if isinstance(callback, WandbLogger):
                callback.wandb_run.log(
                    {"log_lr": np.log10(net.optimizer_.param_groups[0]["lr"])}
                )


class UniquePrefixCheckpoint(Checkpoint):
    """
    This class has two purposes:
    - add a unique prefix to the checkpoint file to avoid
    conflicts between different runs in parallel
    - remove the checkpoint file when training is finished
    to avoid having too many files
    """

    def initialize(self):
        print("Initializing UniquePrefixCheckpoint")
        self.fn_prefix = str(id(self))
        print("fn_prefix is {}".format(self.fn_prefix))
        return super(UniquePrefixCheckpoint, self).initialize()

    # override method to delete the checkpoint file
    def on_train_end(self, net, **kwargs):
        print("train end")
        if not self.load_best or self.monitor is None:
            return
        self._sink("Loading best checkpoint after training.", net.verbose)
        is_regression = isinstance(net, NeuralNetRegressorWrapped)
        try:
            net.load_params(checkpoint=self, use_safetensors=self.use_safetensors)
            # addition
            print(f"removing {self.dirname}/{self.fn_prefix}params.pt")
            os.remove(f"{self.dirname}/{self.fn_prefix}params.pt")
            # if doing regression check if constant_val_mse is better than valid_loss_best
            # if so, replace the model prediction with constant prediction
            if is_regression:
                constant_val_mse = net.history[:, "constant_val_mse"][0]  # all the same
                all_val_mse = net.history[:, "valid_loss"]
                # remove nan and inf
                all_val_mse = np.array(all_val_mse)[~np.isnan(all_val_mse)]

                if not len(all_val_mse) or np.all(all_val_mse > constant_val_mse):
                    print("All valid loss are worse than constant prediction")
                    print("Replacing model prediction with constant prediction")
                    net.set_predict_mean(True)
        except FileNotFoundError:
            print("COULD NOT FIND CHECKPOINT FILE")
            if not is_regression:
                # this should only happen for regression
                raise
            # check that valid loss is always nan or inf
            valid_loss = net.history[:, "valid_loss"]
            assert np.all(np.isnan(valid_loss) | np.isinf(valid_loss))
            print("valid loss is always nan or inf")
            print("Replacing model prediction with constant prediction")
            net.set_predict_mean(True)


class MyCustomError(Exception):
    pass


class EarlyStoppingCustomError(EarlyStopping):
    def on_epoch_end(self, net, **kwargs):
        current_score = net.history[-1, self.monitor]
        if not self._is_score_improved(current_score):
            self.misses_ += 1
        else:
            self.misses_ = 0
            self.dynamic_threshold_ = self._calc_new_threshold(current_score)
            self.best_epoch_ = net.history[-1, "epoch"]
            if self.load_best:
                self.best_model_weights_ = deepcopy(net.module_.state_dict())
        if self.misses_ == self.patience:
            if net.verbose:
                self._sink("Stopping since {} has not improved in the last "
                           "{} epochs.".format(self.monitor, self.patience),
                           verbose=net.verbose)
            raise MyCustomError


class NeuralNetRegressorWrapped(NeuralNetRegressor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.categorical_indicator = None
        self.predict_mean = False  # whether to predict y_train mean if
        # the network predictions are nan or too bad
        self.y_train_mean = None

    def set_categorical_indicator(self, categorical_indicator):
        self.categorical_indicator = categorical_indicator

    def set_predict_mean(self, predict_mean):
        self.predict_mean = predict_mean

    def set_y_train_mean(self, y_train_mean):
        self.y_train_mean = y_train_mean

    def get_default_callbacks(self):
        callbacks = [cb for cb in super().get_default_callbacks() if not isinstance(cb[1], PrintLog)]
        callbacks.append(('print_log', PrintLog(sink=print_but_serializable)))
        print(callbacks)
        return callbacks

    def fit(self, X, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.set_y_train_mean(np.mean(y))
        return super().fit(X, y)

    def predict(self, X):
        if self.predict_mean:
            return np.ones((X.shape[0], 1)) * self.y_train_mean
        else:
            return super().predict(X)

    # adapted from skorch code 
    # to remove ignoring keyboard interrupt
    # as it can be dangerous for benchmarking
    # pylint: disable=unused-argument
    def partial_fit(self, X, y=None, classes=None, **fit_params):
        if not self.initialized_:
            self.initialize()

        self.notify('on_train_begin', X=X, y=y)
        try:
            self.fit_loop(X, y, **fit_params)
        # except KeyboardInterrupt:
        except MyCustomError:
            pass
        self.notify('on_train_end', X=X, y=y)
        return self


class NeuralNetClassifierWrapped(NeuralNetClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.categorical_indicator = None
        self.n_classes = None  # automatically infered from train if not set

    def set_categorical_indicator(self, categorical_indicator):
        self.categorical_indicator = categorical_indicator

    def set_n_classes(self, n_classes):
        self.n_classes = n_classes

    def fit(self, X, y):
        y = y.astype(np.int64)
        return super().fit(X, y)

    def get_default_callbacks(self):
        callbacks = [cb for cb in super().get_default_callbacks() if not isinstance(cb[1], PrintLog)]
        callbacks.append(('print_log', PrintLog(sink=print_but_serializable)))
        print(callbacks)
        return callbacks

    # adapted from skorch code 
    # to remove ignoring keyboard interrupt
    # as it can be dangerous for benchmarking
    # pylint: disable=unused-argument
    def partial_fit(self, X, y=None, classes=None, **fit_params):
        if not self.initialized_:
            self.initialize()

        self.notify('on_train_begin', X=X, y=y)
        try:
            self.fit_loop(X, y, **fit_params)
        # except KeyboardInterrupt:
        except MyCustomError:
            pass
        self.notify('on_train_end', X=X, y=y)
        return self
    
# for FT-Transformer, we extend the NeuralNet class to allow different weight decay for different
# parts of the network
def initialize_optimizer_ft_transformer(self, triggered_directly=None):
    """Initialize the model optimizer. If ``self.optimizer__lr``
    is not set, use ``self.lr`` instead.

    Parameters
    ----------
    triggered_directly
        Deprecated, don't use it anymore.

    """
    # handle deprecated paramter
    # if triggered_directly is not None:
    #     warnings.warn(
    #         "The 'triggered_directly' argument to 'initialize_optimizer' is "
    #         "deprecated, please don't use it anymore.", DeprecationWarning)

    named_parameters = list(self.get_all_learnable_params())
    # print
    no_wd_names = ['tokenizer', '.norm', '.bias']
    for x in ['tokenizer', '.norm', '.bias']:
        assert any(x in a for a in (b[0] for b in named_parameters)) #TODO improve this

    def needs_wd(name):
        return all(x not in name for x in no_wd_names)

    named_parameters_grouped = [
        {'params': [v for k, v in named_parameters if needs_wd(k)]},
        {
            'params': [v for k, v in named_parameters if not needs_wd(k)],
            'weight_decay': 0.0,
        }]
    
    args, kwargs = self.get_params_for_optimizer(
        'optimizer', named_parameters)

    # pylint: disable=attribute-defined-outside-init
    self.optimizer_ = self.optimizer(named_parameters_grouped, **kwargs)
    return self

class NeuralNetClassifierCustomOptim(NeuralNetClassifierWrapped):
    def initialize_optimizer(self, triggered_directly=None):
        return initialize_optimizer_ft_transformer(self, triggered_directly)
    
class NeuralNetRegressorCustomOptim(NeuralNetRegressorWrapped):
    def initialize_optimizer(self, triggered_directly=None):
        return initialize_optimizer_ft_transformer(self, triggered_directly)

def mse_constant_predictor(model, X, y):
    return np.mean((y - model.y_train_mean) ** 2)


def create_regressor_skorch(
        id=None, wandb_run=None, use_checkpoints=True, cat_features=None,
        model_name="resnet", checkpoint_dir="skorch_cp", **kwargs
):
    print("RTDL regressor")
    if "lr_scheduler" not in kwargs:
        lr_scheduler = False
    else:
        lr_scheduler = kwargs.pop("lr_scheduler")
    if "es_patience" not in kwargs.keys():
        es_patience = 40
    else:
        es_patience = kwargs.pop("es_patience")
    if "lr_patience" not in kwargs.keys():
        lr_patience = 30
    else:
        lr_patience = kwargs.pop("lr_patience")
    if "optimizer" not in kwargs.keys():
        optimizer = "adamw"
    else:
        optimizer = kwargs.pop("optimizer")
    if optimizer == "adam":
        optimizer = Adam
    elif optimizer == "adamw":
        optimizer = AdamW
    elif optimizer == "sgd":
        optimizer = SGD
    if "batch_size" not in kwargs.keys():
        batch_size = 128
    else:
        batch_size = kwargs.pop("batch_size")
    if "categories" not in kwargs.keys():
        categories = None
    else:
        categories = kwargs.pop("categories")
    callbacks = [
        InputShapeSetterResnet(
            regression=True, cat_features=cat_features, categories=categories,
            batch_size=batch_size
        ),
        EpochScoring(scoring=mse_constant_predictor, name="constant_val_mse", on_train=False),
        EarlyStoppingCustomError(monitor="valid_loss", patience=es_patience, sink=print_but_serializable),
    ]

    if lr_scheduler:
        callbacks.append(
            LRScheduler(
                policy=ReduceLROnPlateau, patience=lr_patience, min_lr=2e-5, factor=0.2
            )
        )  # FIXME make customizable
    if use_checkpoints:
        callbacks.append(
            UniquePrefixCheckpoint(
                dirname=checkpoint_dir,
                f_params=r"params.pt",
                f_optimizer=None,
                f_criterion=None,
                f_history=None,
                load_best=True,
                monitor="valid_loss_best",
                sink=print_but_serializable,
            )
        )
    if not wandb_run is None:
        callbacks.append(WandbLogger(wandb_run, save_model=False))
        callbacks.append(LearningRateLogger())

    nn_class = NeuralNetRegressorCustomOptim if model_name == "ft_transformer" else NeuralNetRegressorWrapped
    if model_name == "ft_transformer":
        model_class = FT_Transformer
    elif model_name == "resnet":
        model_class = ResNet
    elif model_name == "mlp":
        model_class = RTDL_MLP
    else:
        raise ValueError(f'Model {model_name} not implemented here! Choose from "ft_transformer", "resnet", "mlp"')
    model = nn_class(
        model_class,
        # Shuffle training data on each epoch
        optimizer=optimizer,
        batch_size=max(
            batch_size, 1
        ),  # if batch size is float, it will be reset during fit
        iterator_train__shuffle=True,
        module__d_in=1,  # will be change when fitted
        module__categories=None,  # will be change when fitted
        module__d_out=1,  # idem
        module__regression=True,
        module__categorical_indicator=None,  # will be change when fitted
        callbacks=callbacks,
        **kwargs,
    )

    return model


def create_classifier_skorch(
        id=None, wandb_run=None, use_checkpoints=True, cat_features=None,
        model_name="resnet", checkpoint_dir="skorch_cp", val_metric_name: str = 'class_error',
        **kwargs
):
    print("RTDL classifier")
    if "lr_scheduler" not in kwargs:
        lr_scheduler = False
    else:
        lr_scheduler = kwargs.pop("lr_scheduler")
    if "es_patience" not in kwargs.keys():
        es_patience = 40
    else:
        es_patience = kwargs.pop("es_patience")
    if "lr_patience" not in kwargs.keys():
        lr_patience = 30
    else:
        lr_patience = kwargs.pop("lr_patience")
    if "optimizer" not in kwargs.keys():
        optimizer = "adamw"
    else:
        optimizer = kwargs.pop("optimizer")
    if optimizer == "adam":
        optimizer = Adam
    elif optimizer == "adamw":
        optimizer = AdamW
    elif optimizer == "sgd":
        optimizer = SGD
    if "batch_size" not in kwargs.keys():
        batch_size = 128
    else:
        batch_size = kwargs.pop("batch_size")
    if "categories" not in kwargs.keys():
        categories = None
    else:
        categories = kwargs.pop("categories")
    callbacks = [
        InputShapeSetterResnet(
            regression=False, cat_features=cat_features, categories=categories,
            batch_size=batch_size
        ),
        EpochScoring(scoring="accuracy", name="train_accuracy", on_train=True),
    ]
    if val_metric_name == 'class_error':
        callbacks.append(EarlyStoppingCustomError(monitor="valid_acc", patience=es_patience,
                                                  lower_is_better=False, sink=print_but_serializable))
    elif val_metric_name == 'cross_entropy':
        print(f'Using early stopping on cross-entropy loss')
        callbacks.append(EarlyStoppingCustomError(monitor='valid_loss', patience=es_patience,
                                                  lower_is_better=True, sink=print_but_serializable))
    else:
        raise ValueError(f'Validation metric {val_metric_name} not implemented here!')

    if lr_scheduler:
        callbacks.append(
            LRScheduler(
                policy=ReduceLROnPlateau, patience=lr_patience, min_lr=2e-5, factor=0.2
            )
        )  # FIXME make customizable
    if use_checkpoints:
        callbacks.append(
            UniquePrefixCheckpoint(
                dirname=checkpoint_dir,
                f_params=r"params.pt",
                f_optimizer=None,
                f_criterion=None,
                f_history=None,
                load_best=True,
                monitor="valid_acc_best" if val_metric_name == 'class_error' else 'valid_loss_best',
                sink=print_but_serializable,
            )
        )
    if not wandb_run is None:
        callbacks.append(WandbLogger(wandb_run, save_model=False))
        callbacks.append(LearningRateLogger())

    nn_class = NeuralNetClassifierCustomOptim if model_name == "ft_transformer" else NeuralNetClassifierWrapped
    if model_name == "ft_transformer":
        model_class = FT_Transformer
    elif model_name == "resnet":
        model_class = ResNet
    elif model_name == "mlp":
        model_class = RTDL_MLP
    else:
        raise ValueError(f'Model {model_name} not implemented here! Choose from "ft_transformer", "resnet", "mlp"')
    model = nn_class(
        model_class,
        # Shuffle training data on each epoch
        criterion=nn.CrossEntropyLoss,
        optimizer=optimizer,
        batch_size=max(
            batch_size, 1
        ),  # if batch size is float, it will be reset during fit
        iterator_train__shuffle=True,
        module__d_in=1,  # will be change when fitted
        module__categories=None,  # will be change when fitted
        module__d_out=1,  # idem
        module__regression=False,
        module__categorical_indicator=None,  # will be change when fitted
        callbacks=callbacks,
        **kwargs,
    )

    return model


create_resnet_regressor_skorch = partial(create_regressor_skorch, model_name="resnet", use_checkpoints=True)
create_resnet_classifier_skorch = partial(create_classifier_skorch, model_name="resnet", use_checkpoints=True)
create_mlp_regressor_skorch = partial(create_regressor_skorch, model_name="mlp", use_checkpoints=True)
create_mlp_classifier_skorch = partial(create_classifier_skorch, model_name="mlp", use_checkpoints=True)
create_ft_transformer_regressor_skorch = partial(create_regressor_skorch, model_name="ft_transformer", use_checkpoints=True)
create_ft_transformer_classifier_skorch = partial(create_classifier_skorch, model_name="ft_transformer", use_checkpoints=True)
