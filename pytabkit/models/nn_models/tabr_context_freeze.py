import os
import inspect
import warnings
import math
from functools import partial

import torch
from torch import Tensor
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from pytabkit.models.nn_models import tabr_lib as lib
import torch.nn as nn
from torchmetrics import Accuracy, Precision, Recall, F1Score, MeanSquaredError, AUROC, MeanAbsoluteError
from typing import Any, Optional, Union, Literal, Callable, NamedTuple
from tqdm import tqdm

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl

from pytabkit.models.nn_models.tabr import ParametricMishActivationLayer, ParametricReluActivationLayer, ScalingLayer, \
    bce_with_logits_and_label_smoothing


# taken from https://github.com/yandex-research/tabular-dl-tabr/tree/main/bin
# and https://github.com/yandex-research/tabular-dl-tabr/blob/main/bin/tabr_scaling.py
class TabrModelContextFreeze(nn.Module):
    class ForwardOutput(NamedTuple):
        y_pred: Tensor
        context_idx: Tensor
        context_probs: Tensor

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
            add_scaling_layer: bool = False,
            scale_lr_factor: float = 6.0,
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

        def make_block(prenorm: bool) -> nn.Sequential:
            return nn.Sequential(
                *([Normalization(d_main)] if prenorm else []),
                nn.Linear(d_main, d_block),
                Activation(d_block),
                nn.Dropout(dropout0),
                nn.Linear(d_block, d_main),
                nn.Dropout(dropout1),
            )

        self.scale = ScalingLayer(d_in, lr_factor=scale_lr_factor) if add_scaling_layer else nn.Identity()
        self.linear = nn.Linear(d_in, d_main)
        self.blocks0 = nn.ModuleList(
            [make_block(i > 0) for i in range(encoder_n_blocks)]
        )

        # >>> R
        self.normalization = Normalization(d_main) if mixer_normalization else None
        self.label_encoder = (
            nn.Linear(1, d_main)
            if n_classes is None
            else nn.Sequential(
                nn.Embedding(n_classes, d_main), lib.Lambda(lambda x: x.squeeze(-2))
            )
        )
        self.K = nn.Linear(d_main, d_main)
        self.T = nn.Sequential(
            nn.Linear(d_main, d_block),
            Activation(d_block),
            nn.Dropout(dropout0),
            nn.Linear(d_block, d_main, bias=False),
        )
        self.dropout = nn.Dropout(context_dropout)

        # >>> P
        self.blocks1 = nn.ModuleList(
            [make_block(True) for _ in range(predictor_n_blocks)]
        )
        self.head = nn.Sequential(
            Normalization(d_main),
            Activation(d_main),
            nn.Linear(d_main, lib.get_d_out(n_classes)),
        )

        # >>>
        self.search_index = None
        self.memory_efficient = memory_efficient
        self.candidate_encoding_batch_size = candidate_encoding_batch_size
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.label_encoder, nn.Linear):
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
            idx: Optional[Tensor],
            candidate_x_: dict[str, Tensor],
            candidate_y: Tensor,
            candidate_idx: Tensor,
            context_size: int,
            context_idx: Optional[Tensor],
            is_train: bool,
    ):
        # import locally so importing this file doesn't cause problems if faiss is not installed
        import faiss
        import faiss.contrib.torch_utils  # noqa  << this line makes faiss work with PyTorch
        # >>> E
        with torch.set_grad_enabled(
                torch.is_grad_enabled() and not self.memory_efficient
        ):
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
            assert y is not None
            assert idx is not None
            if context_idx is None:
                candidate_k = torch.cat([k, candidate_k])
                candidate_y = torch.cat([y, candidate_y])
                candidate_idx = torch.cat([idx, candidate_idx])
        else:
            assert y is None
            assert idx is None

        # >>>
        batch_size, d_main = k.shape
        device = k.device
        if context_idx is None:
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
                self.search_index.reset()
                self.search_index.add(candidate_k)  # type: ignore[code]
                distances: Tensor
                distances, context_idx = self.search_index.search(  # type: ignore[code]
                    k, context_size + (1 if is_train else 0)
                )
                assert isinstance(context_idx, Tensor)
                if is_train:
                    distances[
                        context_idx == torch.arange(batch_size, device=device)[:, None]
                        ] = torch.inf
                    context_idx = context_idx.gather(-1, distances.argsort()[:, :-1])
        # print("context_idx", context_idx)
        # "absolute" means "not relative", i.e. the original indices in the train set.
        absolute_context_idx = candidate_idx[context_idx]

        if self.memory_efficient and torch.is_grad_enabled():
            assert is_train
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

        similarities = (
                -k.square().sum(-1, keepdim=True)
                + (2 * (k[..., None, :] @ context_k.transpose(-1, -2))).squeeze(-2)
                - context_k.square().sum(-1)
        )
        raw_probs = F.softmax(similarities, dim=-1)
        probs = self.dropout(raw_probs)

        context_y_emb: Tensor = self.label_encoder(candidate_y[context_idx][..., None])
        values: Tensor = context_y_emb + self.T(k[:, None] - context_k)
        context_x = (probs[:, None] @ values).squeeze(1)
        x = x + context_x

        # >>>
        for block in self.blocks1:
            x: Tensor = x + block(x)
        x: Tensor = self.head(x)
        return TabrModelContextFreeze.ForwardOutput(x, absolute_context_idx, raw_probs)


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


class TabrLightningContextFreeze(pl.LightningModule):
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

        self.frozen_contexts = None

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

    def apply_model(self, part, batch, batch_idx, training):
        # batch should contain dictionaries with keys
        # "x_num", "x_bin", "x_cat", "y" and "indices"
        batch_indices = batch["indices"].to(self.device)  # batch_idx is the id of the batch itself
        # batch_indices contains the ids of the samples in the batch

        # batch_indices contains the ids of the samples in the batch
        x, y = self.get_Xy(part, batch_indices)

        is_train = part == 'train'
        if training and self.frozen_contexts is not None:
            candidate_indices, context_idx = self.frozen_contexts[batch_indices].unique(
                return_inverse=True
            )
        else:
            # Importantly, `training`, not `is_train` should be used to choose the queue
            candidate_indices = self.train_indices
            context_idx = None
            if is_train:
                # This is not done when there are frozen contexts, because they are
                # already valid.
                candidate_indices = candidate_indices[
                    ~torch.isin(candidate_indices, batch_indices)
                ]
        candidate_x, candidate_y = self.get_Xy(
            'train',
            candidate_indices,  # TODO check
        )

        fwd_out = self.model(
            x_=x,
            y=y if is_train else None,
            idx=batch_indices if is_train else None,
            candidate_x_=candidate_x,
            candidate_y=candidate_y,
            candidate_idx=candidate_indices,
            context_idx=context_idx,
            context_size=self.C["context_size"],
            is_train=is_train,
        )
        return fwd_out._replace(y_pred=fwd_out.y_pred.squeeze(-1)), y

    def training_step(self, batch, batch_idx):
        if batch_idx == 0 and self.current_epoch == self.C["freeze_contexts_after_n_epochs"]:
            # freeze the contexts
            print(f'Freezing contexts after {self.current_epoch} epochs', flush=True)
            # Get context_ids using evaluate?
            _, _, context_idx, _, _ = self.evaluate(self.C["eval_batch_size"],
                                                    progress_bar=True  # TODO
                                                    )
            self.frozen_contexts = torch.tensor(context_idx['train'], device=self.device)

        # # batch should contain dictionaries with keys
        # # "x_num", "x_bin", "x_cat", "y" and "indices"
        # batch_indices = batch["indices"] # batch_idx is the id of the batch itself
        # # batch_indices contains the ids of the samples in the batch

        # x, y = self.get_Xy('train', batch_indices)

        # if self.frozen_contexts is not None:
        #     candidate_indices, context_idx = self.frozen_contexts[batch_indices].unique(
        #         return_inverse=True
        #     )
        # else:
        #     context_idx = None
        #     # we're in training mode
        #     # Remove the training batch from the candidates
        #     # This is not done when there are frozen contexts, because they are
        #     # already valid.
        #     candidate_indices = self.train_indices[~torch.isin(self.train_indices, batch_indices)]

        # candidate_x, candidate_y = self.get_Xy('train', candidate_indices) #TODO check

        # fwd_out = self.model(
        #     x_=x,
        #     y=y,
        #     idx=batch_indices,
        #     candidate_x_=candidate_x,
        #     candidate_y=candidate_y,
        #     candidate_idx=candidate_indices,
        #     context_idx=context_idx,
        #     context_size=self.C["context_size"],
        #     is_train=True
        # )
        # fwd_out = fwd_out._replace(y_pred=fwd_out.y_pred.squeeze(-1))
        fwd_out, y = self.apply_model("train", batch, batch_idx, training=True)
        output, _, _ = fwd_out

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
        # batch_indices = batch["indices"]  # batch_idx is the idxs of the batch samples
        # x, y = self.get_Xy("val", batch_indices)

        # if self.frozen_contexts is not None:
        #     candidate_indices, context_idx = self.frozen_contexts[batch_indices].unique(
        #         return_inverse=True
        #     )
        # else:
        #     context_idx = None
        #     candidate_indices = self.train_indices

        # candidate_x, candidate_y = self.get_Xy('train', candidate_indices)

        # fwd_out = self.model(
        #     x_=x,
        #     y=None,
        #     idx=None,
        #     candidate_x_=candidate_x,
        #     candidate_y=candidate_y,
        #     candidate_idx=candidate_indices,
        #     context_idx=context_idx,
        #     context_size=self.C["context_size"],
        #     is_train=False
        # )
        # fwd_out = fwd_out._replace(y_pred=fwd_out.y_pred.squeeze(-1))
        fwd_out, y = self.apply_model("val", batch, batch_idx, training=False)
        output, _, _ = fwd_out
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
        # TODO: use apply_model
        x = {
            key[2:]: batch[key]
            for key in batch
            if key.startswith('X_')
        }
        context_idx = None
        candidate_indices = self.train_indices
        candidate_x, candidate_y = self.get_Xy('train', candidate_indices)

        fwd_out = self.model(
            x_=x,
            y=None,
            idx=None,
            candidate_x_=candidate_x,
            candidate_y=candidate_y,
            candidate_idx=candidate_indices,
            context_idx=context_idx,
            context_size=self.C["context_size"],
            is_train=False
        )
        fwd_out = fwd_out._replace(y_pred=fwd_out.y_pred.squeeze(-1))
        # fwd_out, y = self.apply_model("test", batch, batch_idx, training=False)
        output, _, _ = fwd_out

        # in binary case, we need to convert it to 2-class logits
        if self.task_type == "binary":
            # it will be passed to a softmax, so we need to add a 0
            # to make the probabilities right
            output = torch.stack([torch.zeros_like(output), output], dim=1)
        elif self.task_type == "regression":
            output = output.unsqueeze(1)

        return output

    # here we only use it to get context_idx for the frozen contexts
    # so we only need to do it on train
    @torch.inference_mode()
    def evaluate(self, eval_batch_size: int, *, progress_bar: bool = False):
        self.eval()
        predictions = {}
        context_idx = {}
        context_probs = {}
        while eval_batch_size:
            try:
                # fwd_out = []
                # for idx in tqdm(
                #     torch.arange(len(self.dataset), device=self.device).split(
                #         eval_batch_size
                #     ),
                #     desc=f'Evaluation ("train"))',
                #     disable=not progress_bar,
                # ):
                #     batch = {
                #         key: self.dataset.data[key][idx]
                #         for key in self.dataset.data
                #     }
                #     x = {
                #         key[2:]: batch[key]
                #         for key in batch
                #         if key.startswith('X_')
                #     }
                #     #TODO check
                #     fwd_out.append(
                #         self.model(
                #             x_=x,
                #             y=None,
                #             idx=None,
                #             candidate_x_=x,
                #             candidate_y=batch['Y'],
                #             candidate_idx=idx,
                #             context_idx=None,
                #             context_size=self.C["context_size"],
                #             is_train=False
                #         )
                #     )
                fwd_out = lib.cat(
                    [
                        self.apply_model("train", batch, batch_idx, training=False)[0]
                        for batch_idx, batch in enumerate(
                        DataLoader(
                            self.dataset, batch_size=eval_batch_size, shuffle=False
                        )
                    )
                    ]
                )
                # fwd_out = lib.cat(fwd_out)
                predictions["train"], context_idx["train"], context_probs["train"] = (
                    e.cpu().numpy() for e in fwd_out
                )
            except RuntimeError as err:
                if not lib.is_oom_exception(err):
                    raise
                eval_batch_size //= 2
                print(f'eval_batch_size = {eval_batch_size}')
            else:
                break
        if not eval_batch_size:
            RuntimeError('Not enough memory even for eval_batch_size=1')
        metrics = None
        self.train()
        return metrics, predictions, context_idx, context_probs, eval_batch_size

    def configure_optimizers(self):
        optimizer_config = self.C["optimizer"].copy()
        optimizer = lib.make_optimizer(
            self.model, **optimizer_config, zero_weight_decay_condition=zero_wd_condition
        )
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.C["batch_size"], shuffle=True,
                          num_workers=max(1, min(self.C["n_threads"] - 1, 8)),
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.C["eval_batch_size"], shuffle=False,
                          num_workers=max(1, min(self.C["n_threads"] - 1, 8)),
                          persistent_workers=True)
