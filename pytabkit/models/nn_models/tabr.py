import os
import inspect
import warnings
import math
from functools import partial

import numpy as np
import torch
from torch import Tensor
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from pytabkit.models.nn_models import tabr_lib as lib
import torch.nn as nn
from torchmetrics import Accuracy, Precision, Recall, F1Score, MeanSquaredError, AUROC, MeanAbsoluteError
from typing import Any, Optional, Union, Literal, Callable

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl


class NTPLinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, bias_factor: float = 0.1, linear_init_type: str = 'default'):
        super().__init__()
        self.use_bias = bias
        if linear_init_type == 'default':
            self.weight = nn.Parameter(-1+2*torch.rand(in_features, out_features))
            if self.use_bias:
                self.bias = nn.Parameter((-1+2*torch.rand(1, out_features)) / np.sqrt(in_features))
        elif linear_init_type == 'normal':
            self.weight = nn.Parameter(torch.randn(in_features, out_features))
            if self.use_bias:
                self.bias = nn.Parameter(torch.randn(1, out_features))
        else:
            raise ValueError(f'Unknown linear_init_type "{linear_init_type}"')
        self.bias_factor = bias_factor
        self.weight_factor = 1./np.sqrt(in_features)

    def forward(self, x):
        x = self.weight_factor * x @ self.weight
        if self.use_bias:
            x = x + self.bias_factor * self.bias
        return x


class ParametricMishActivationLayer(nn.Module):
    def __init__(self, n_features: int, lr_factor: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter((1. / lr_factor) * torch.ones(n_features))
        self.lr_factor = lr_factor

    def f(self, x):
        return x.mul(torch.tanh(F.softplus(x)))

    def forward(self, x):
        # print(f'{self.weight.mean().item()=:g}')
        return x + self.lr_factor * (self.f(x) - x) * self.weight


class ParametricReluActivationLayer(nn.Module):
    def __init__(self, n_features: int, lr_factor: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter((1. / lr_factor) * torch.ones(n_features))
        self.lr_factor = lr_factor

    def f(self, x):
        return torch.relu(x)

    def forward(self, x):
        # print(f'{self.weight.mean().item()=:g}')
        return x + self.lr_factor * (self.f(x) - x) * self.weight


class ScalingLayer(nn.Module):
    def __init__(self, n_features: int, lr_factor: float = 6.0):
        super().__init__()
        self.weight = nn.Parameter((1. / lr_factor) * torch.ones(n_features))
        self.lr_factor = lr_factor

    def forward(self, x):
        return self.lr_factor * x * self.weight[None, :]


def bce_with_logits_and_label_smoothing(inputs, *args, ls_eps: float, **kwargs):
    return (1 - 0.5 * ls_eps) * F.binary_cross_entropy_with_logits(inputs, *args, **kwargs) \
        + 0.5 * ls_eps * F.binary_cross_entropy_with_logits(-inputs, *args, **kwargs)


# adapted from https://github.com/yandex-research/tabular-dl-tabr/tree/main/bin
class TabrModel(nn.Module):
    def __init__(
            self,
            *,
            #
            n_num_features: int,
            n_bin_features: int,
            cat_cardinalities: list[int],
            n_classes: Optional[int],
            #
            num_embeddings: Optional[dict],  # lib.deep.ModuleSpec
            d_main: int,
            d_multiplier: float,
            encoder_n_blocks: int,
            predictor_n_blocks: int,
            mixer_normalization: Union[bool, Literal['auto']],
            context_dropout: float,
            dropout0: float,
            dropout1: Union[float, Literal['dropout0']],
            normalization: str,
            activation: str,
            #
            # The following options should be used only when truly needed.
            memory_efficient: bool = False,
            candidate_encoding_batch_size: Optional[int] = None,
            # extra options not in the original tabr
            add_scaling_layer: bool = False,
            scale_lr_factor: float = 6.0,
            use_ntp_linear: bool = False,
            linear_init_type: str = 'default',  # only relevant if use_ntp_linear=True
            use_ntp_encoder: bool = False,
    ) -> None:
        # import locally so importing this file doesn't cause problems if faiss is not installed
        # import in constructor as well to make model fail earlier if not installed
        import faiss
        import faiss.contrib.torch_utils  # noqa  << this line makes faiss work with PyTorch
        if not memory_efficient:
            assert candidate_encoding_batch_size is None
        if mixer_normalization == 'auto':
            mixer_normalization = encoder_n_blocks > 0
        if encoder_n_blocks == 0:
            assert not mixer_normalization
        super().__init__()
        if dropout1 == 'dropout0':
            dropout1 = dropout0

        self.one_hot_encoder = (
            lib.OneHotEncoder(cat_cardinalities) if cat_cardinalities else None
        )
        self.num_embeddings = (
            None
            if num_embeddings is None
            else lib.make_module(num_embeddings, n_features=n_num_features)
        )

        print(f'{add_scaling_layer=}')
        print(f'{activation=}')
        print(f'{scale_lr_factor=}')

        # >>> E
        d_in = (
                n_num_features
                * (1 if num_embeddings is None else num_embeddings['d_embedding'])
                + n_bin_features
                + sum(cat_cardinalities)
        )
        d_block = int(d_main * d_multiplier)
        Normalization = getattr(nn, normalization)
        if activation == 'pmish':
            Activation = lambda n_features: ParametricMishActivationLayer(n_features=n_features)
        elif activation == 'prelu':
            Activation = lambda n_features: ParametricReluActivationLayer(n_features=n_features)
        else:
            Activation = lambda n_features: getattr(nn, activation)()

        if use_ntp_linear:
            print(f'Using NTP linear layer with init {linear_init_type}')
            Linear = lambda in_features, out_features, bias=True: NTPLinearLayer(in_features, out_features, bias=bias, bias_factor=0.1, linear_init_type=linear_init_type)
        else:
            Linear = nn.Linear

        def make_block(prenorm: bool) -> nn.Sequential:
            return nn.Sequential(
                *([Normalization(d_main)] if prenorm else []),
                Linear(d_main, d_block),
                Activation(d_block),
                nn.Dropout(dropout0),
                Linear(d_block, d_main),
                nn.Dropout(dropout1),
            )

        self.scale = ScalingLayer(d_in, lr_factor=scale_lr_factor) if add_scaling_layer else nn.Identity()
        self.linear = Linear(d_in, d_main)
        self.blocks0 = nn.ModuleList(
            [make_block(i > 0) for i in range(encoder_n_blocks)]
        )

        # >>> R
        self.normalization = Normalization(d_main) if mixer_normalization else None
        self.label_encoder = (
            Linear(1, d_main) if use_ntp_encoder else nn.Linear(1, d_main)
            if n_classes is None
            else nn.Sequential(
                nn.Embedding(n_classes, d_main), lib.Lambda(lambda x: x.squeeze(-2))
            )
        )
        self.K = Linear(d_main, d_main)
        self.T = nn.Sequential(
            Linear(d_main, d_block),
            Activation(d_block),
            nn.Dropout(dropout0),
            Linear(d_block, d_main, bias=False),
        )
        self.dropout = nn.Dropout(context_dropout)

        # >>> P
        self.blocks1 = nn.ModuleList(
            [make_block(True) for _ in range(predictor_n_blocks)]
        )
        self.head = nn.Sequential(
            Normalization(d_main),
            Activation(d_main),
            Linear(d_main, lib.get_d_out(n_classes)),
        )

        # >>>
        self.search_index = None
        self.memory_efficient = memory_efficient
        self.candidate_encoding_batch_size = candidate_encoding_batch_size
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.label_encoder, nn.Linear) or isinstance(self.label_encoder, NTPLinearLayer):
            bound = 1 / math.sqrt(2.0)
            nn.init.uniform_(self.label_encoder.weight, -bound, bound)  # type: ignore[code]  # noqa: E501
            nn.init.uniform_(self.label_encoder.bias, -bound, bound)  # type: ignore[code]  # noqa: E501
        else:
            assert isinstance(self.label_encoder[0], nn.Embedding)
            nn.init.uniform_(self.label_encoder[0].weight, -1.0, 1.0)  # type: ignore[code]  # noqa: E501

    def _encode(self, x_: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        x_num = x_.get('num')
        x_bin = x_.get('bin')
        x_cat = x_.get('cat')
        del x_

        x = []
        if x_num is None:
            # assert self.num_embeddings is None
            pass  # changed to make it easier to use with all-categorical datasets
        else:
            x.append(
                x_num
                if self.num_embeddings is None
                else self.num_embeddings(x_num).flatten(1)
            )
        if x_bin is not None:
            x.append(x_bin)
        if x_cat is None:
            assert self.one_hot_encoder is None
        else:
            assert self.one_hot_encoder is not None
            x.append(self.one_hot_encoder(x_cat))
        assert x
        x = torch.cat(x, dim=1).float()

        x = self.scale(x)
        x = self.linear(x)
        for block in self.blocks0:
            x = x + block(x)
        k = self.K(x if self.normalization is None else self.normalization(x))
        return x, k

    def forward(
            self,
            *,
            x_: dict[str, Tensor],
            y: Optional[Tensor],
            candidate_x_: dict[str, Tensor],
            candidate_y: Tensor,
            context_size: int,
            is_train: bool,
    ) -> Tensor:
        # print('forward()')
        # import locally so importing this file doesn't cause problems if faiss is not installed
        import faiss
        import faiss.contrib.torch_utils  # noqa  << this line makes faiss work with PyTorch

        # >>>
        with torch.set_grad_enabled(
                torch.is_grad_enabled() and not self.memory_efficient
        ):
            # NOTE: during evaluation, candidate keys can be computed just once, which
            # looks like an easy opportunity for optimization. However:
            # - if your dataset is small or/and the encoder is just a linear layer
            #   (no embeddings and encoder_n_blocks=0), then encoding candidates
            #   is not a bottleneck.
            # - implementing this optimization makes the code complex and/or unobvious,
            #   because there are many things that should be taken into account:
            #     - is the input coming from the "train" part?
            #     - is self.training True or False?
            #     - is PyTorch autograd enabled?
            #     - is saving and loading checkpoints handled correctly?
            # This is why we do not implement this optimization.

            # When memory_efficient is True, this potentially heavy computation is
            # performed without gradients.
            # Later, it is recomputed with gradients only for the context objects.
            candidate_k = (
                self._encode(candidate_x_)[1]
                if self.candidate_encoding_batch_size is None
                else torch.cat(
                    [
                        self._encode(x)[1]
                        for x in lib.iter_batches(
                        candidate_x_, self.candidate_encoding_batch_size
                    )
                    ]
                )
            )
        x, k = self._encode(x_)
        if is_train:
            # NOTE: here, we add the training batch back to the candidates after the
            # function `apply_model` removed them. The further code relies
            # on the fact that the first batch_size candidates come from the
            # training batch.
            assert y is not None
            candidate_k = torch.cat([k, candidate_k])
            candidate_y = torch.cat([y, candidate_y])
        else:
            assert y is None

        # >>>
        # The search below is optimized for larger datasets and is significantly faster
        # than the naive solution (keep autograd on + manually compute all pairwise
        # squared L2 distances + torch.topk).
        # For smaller datasets, however, the naive solution can actually be faster.
        batch_size, d_main = k.shape
        device = k.device
        with torch.no_grad():
            if self.search_index is None:
                # self.search_index = (
                #     faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d_main)
                #     if device.type == 'cuda'
                #     else faiss.IndexFlatL2(d_main)
                # )
                if device.type == 'cpu':
                    self.search_index = faiss.IndexFlatL2(d_main)
                elif device.type == 'cuda':
                    gpu_index = 0 if device.index is None else device.index
                    cfg = faiss.GpuIndexFlatConfig()
                    cfg.device = gpu_index
                    self.search_index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d_main, cfg)
                else:
                    raise ValueError()
            # Updating the index is much faster than creating a new one.
            self.search_index.reset()
            self.search_index.add(candidate_k)  # type: ignore[code]
            distances: Tensor
            context_idx: Tensor
            distances, context_idx = self.search_index.search(  # type: ignore[code]
                k, context_size + (1 if is_train else 0)
            )
            if is_train:
                # NOTE: to avoid leakage, the index i must be removed from the i-th row,
                # (because of how candidate_k is constructed).
                distances[
                    context_idx == torch.arange(batch_size, device=device)[:, None]
                    ] = torch.inf
                # Not the most elegant solution to remove the argmax, but anyway.
                context_idx = context_idx.gather(-1, distances.argsort()[:, :-1])

        if self.memory_efficient and torch.is_grad_enabled():
            assert is_train
            # Repeating the same computation,
            # but now only for the context objects and with autograd on.
            context_k = self._encode(
                {
                    ftype: torch.cat([x_[ftype], candidate_x_[ftype]])[
                        context_idx
                    ].flatten(0, 1)
                    for ftype in x_
                }
            )[1].reshape(batch_size, context_size, -1)
        else:
            context_k = candidate_k[context_idx]

        # In theory, when autograd is off, the distances obtained during the search
        # can be reused. However, this is not a bottleneck, so let's keep it simple
        # and use the same code to compute `similarities` during both
        # training and evaluation.
        similarities = (
                -k.square().sum(-1, keepdim=True)
                + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2)
                - context_k.square().sum(-1)
        )
        probs = F.softmax(similarities, dim=-1)
        probs = self.dropout(probs)

        context_y_emb = self.label_encoder(candidate_y[context_idx][..., None])
        values = context_y_emb + self.T(k[:, None] - context_k)
        context_x = (probs[:, None] @ values).squeeze(1)
        x = x + context_x

        # >>>
        for block in self.blocks1:
            x = x + block(x)
        x = self.head(x)
        return x


def zero_wd_condition(
        module_name: str,
        module: nn.Module,
        parameter_name: str,
        parameter: nn.parameter.Parameter,
):
    return (
            'label_encoder' in module_name
            or 'label_encoder' in parameter_name
            or lib.default_zero_weight_decay_condition(
        module_name, module, parameter_name, parameter
    )
    )


class TabrLightning(pl.LightningModule):
    def __init__(self, model, train_dataset,
                 val_dataset, C, n_classes):
        super().__init__()
        self.model = model
        self.dataset = train_dataset
        self.val_dataset = val_dataset
        self.C = C
        if n_classes == 2:
            self.task_type = "binary"
        elif n_classes > 2:
            self.task_type = "multiclass"
        else:
            self.task_type = "regression"

        ls_eps = self.C.get('ls_eps', 0.0)
        print(f'{ls_eps=}')

        self.loss_fn = (
            partial(bce_with_logits_and_label_smoothing, ls_eps=ls_eps)
            if self.task_type == "binary"
            else partial(F.cross_entropy, label_smoothing=ls_eps)
            if self.task_type == "multiclass"
            else F.mse_loss
        )
        # Define metrics for binary and multiclass classification
        if self.task_type in ["binary", "multiclass"]:
            self.train_accuracy = Accuracy(task=self.task_type, num_classes=n_classes)
            self.train_precision = Precision(average='macro', num_classes=n_classes, task=self.task_type)
            self.train_recall = Recall(average='macro', num_classes=n_classes, task=self.task_type)
            self.train_f1_score = F1Score(average='macro', num_classes=n_classes, task=self.task_type)
            self.val_accuracy = Accuracy(task=self.task_type, num_classes=n_classes)
            self.val_precision = Precision(average='macro', num_classes=n_classes, task=self.task_type)
            self.val_recall = Recall(average='macro', num_classes=n_classes, task=self.task_type)
            self.val_f1_score = F1Score(average='macro', num_classes=n_classes, task=self.task_type)

        # Define metrics for regression
        elif self.task_type == "regression":
            self.train_mse = MeanSquaredError()
            self.val_mse = MeanSquaredError()
            self.train_mae = MeanAbsoluteError()
            self.val_mae = MeanAbsoluteError()

    def setup(self, stage=None):
        self.train_size = len(self.dataset)
        self.train_indices = torch.arange(self.train_size, device=self.device)
        # move the dataset to the device
        # I think that's what tabr does, but
        # we could also keep it on the cpu
        for key in self.dataset.data:
            if self.dataset.data[key] is not None:
                self.dataset.data[key] = self.dataset.data[key].to(self.device)
        for key in self.val_dataset.data:
            if self.val_dataset.data[key] is not None:
                self.val_dataset.data[key] = self.val_dataset.data[key].to(self.device)

    def get_Xy(self, part: str, idx) -> tuple[dict[str, Tensor], Tensor]:
        if self.val_dataset.data['Y'].get_device() == -1:
            # is still on CPU
            self.setup()
        if part == "train":
            dataset = self.dataset
        elif part == "val":
            dataset = self.val_dataset
        batch = (
            {
                key[2:]: dataset.data[key]
                for key in dataset.data
                if key.startswith('X_')
            },
            dataset.data["Y"],
        )
        return (
            batch
            if idx is None
            else ({k: v[idx] for k, v in batch[0].items()}, batch[1][idx])
        )

    def training_step(self, batch, batch_idx):
        # batch should contain dictionaries with keys
        # "x_num", "x_bin", "x_cat", "y" and "indices"
        batch_indices = batch["indices"]  # batch_idx is the id of the batch itself
        # batch_indices contains the ids of the samples in the batch

        x, y = self.get_Xy('train', batch_indices)

        # we're in training mode
        # Remove the training batch from the candidates
        candidate_indices = self.train_indices[~torch.isin(self.train_indices, batch_indices)]

        candidate_x, candidate_y = self.get_Xy('train', candidate_indices)

        # Call the model's forward method
        output = self.model(
            x_=x,
            y=y,
            candidate_x_=candidate_x,
            candidate_y=candidate_y,
            context_size=self.C["context_size"],
            is_train=True
        ).squeeze(-1)
        y = y.float() if self.task_type == "regression" else y.long()
        # binary cross entropy with logits needs float
        loss = self.loss_fn(output, y.float() \
            if self.task_type == "binary" \
            else y)
        # Log the loss and return it
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        if self.task_type in ["binary", "multiclass"]:
            self.train_accuracy.update(output, y)
            self.train_precision.update(output, y)
            self.train_recall.update(output, y)
            self.train_f1_score.update(output, y)
            self.log('train_accuracy', self.train_accuracy, on_epoch=True, prog_bar=True)
            self.log('train_precision', self.train_precision, on_epoch=True)
            self.log('train_recall', self.train_recall, on_epoch=True)
            self.log('train_f1_score', self.train_f1_score, on_epoch=True)
        elif self.task_type == "regression":
            self.train_mse.update(output, y)
            self.train_mae.update(output, y)
            self.log('train_mse', self.train_mse, on_epoch=True)
            self.log('train_mae', self.train_mae, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            print(f'Validation in epoch {self.current_epoch}', flush=True)
        # print(f'Validation step', flush=True)
        # TODO: do like test to save gpu memory?
        batch_indices = batch["indices"]  # batch_idx is the idxs of the batch samples
        x, y = self.get_Xy("val", batch_indices)

        candidate_indices = self.train_indices
        candidate_x, candidate_y = self.get_Xy('train', candidate_indices)

        output = self.model(
            x_=x,
            y=None,
            candidate_x_=candidate_x,
            candidate_y=candidate_y,
            context_size=self.C["context_size"],
            is_train=False,
        ).squeeze(-1)
        y = y.float() if self.task_type == "regression" else y.long()
        # binary cross entropy with logits needs float
        loss = self.loss_fn(output, y.float() \
            if self.task_type == "binary" \
            else y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)  # Log validation loss

        if self.task_type in ["binary", "multiclass"]:
            self.val_accuracy.update(output, y)
            self.val_precision.update(output, y)
            self.val_recall.update(output, y)
            self.val_f1_score(output, y)
            self.log('val_accuracy', self.val_accuracy, on_epoch=True, prog_bar=True)
            self.log('val_precision', self.val_precision, on_epoch=True)
            self.log('val_recall', self.val_recall, on_epoch=True)
            self.log('val_f1_score', self.val_f1_score, on_epoch=True)
        elif self.task_type == "regression":
            self.val_mse.update(output, y)
            self.log('val_mse', self.val_mse, on_epoch=True)
            self.val_mae.update(output, y)
            self.log('val_mae', self.val_mae, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        # here batch shouldn't contain indices nor y
        x = {
            key[2:]: batch[key]
            for key in batch
            if key.startswith('X_')
        }
        candidate_indices = self.train_indices
        candidate_x, candidate_y = self.get_Xy('train', candidate_indices)

        output = self.model(
            x_=x,
            y=None,
            candidate_x_=candidate_x,
            candidate_y=candidate_y,
            context_size=self.C["context_size"],
            is_train=False,
        ).squeeze(-1)

        # in binary case, we need to convert it to 2-class logits
        if self.task_type == "binary":
            # it will be passed to a softmax, so we need to add a 0
            # to make the probabilities right
            output = torch.stack([torch.zeros_like(output), output], dim=1)
        elif self.task_type == "regression":
            output = output.unsqueeze(1)

        return output

    def configure_optimizers(self):
        optimizer_config = self.C["optimizer"].copy()
        optimizer = lib.make_optimizer(
            self.model, **optimizer_config, zero_weight_decay_condition=zero_wd_condition
        )
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.C["batch_size"], shuffle=True,
                          num_workers=0, #max(1, min(self.C["n_threads"] - 1, 8)),
                          persistent_workers=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.C["eval_batch_size"], shuffle=False,
                          num_workers=0, #max(1, min(self.C["n_threads"] - 1, 8)),
                          persistent_workers=False)
