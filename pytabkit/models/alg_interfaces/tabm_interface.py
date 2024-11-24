import math
import random
from pathlib import Path

import scipy
import sklearn
import torch
import numpy as np
from torch import nn

from pytabkit.models import utils
from pytabkit.models.alg_interfaces.alg_interfaces import SingleSplitAlgInterface
from typing import Optional, List, Dict, Any, Union, Tuple, Literal

from pytabkit.models.alg_interfaces.base import SplitIdxs, InterfaceResources, RequiredResources
from pytabkit.models.alg_interfaces.resource_computation import ResourcePredictor
from pytabkit.models.data.data import DictDataset
from pytabkit.models.nn_models import rtdl_num_embeddings
from pytabkit.models.nn_models.base import Fitter
from pytabkit.models.nn_models.models import PreprocessingFactory
from pytabkit.models.nn_models.tabm import Model
from pytabkit.models.training.logging import Logger


class TabMSubSplitInterface(SingleSplitAlgInterface):
    def __init__(self, fit_params: Optional[List[Dict[str, Any]]] = None, **config):
        super().__init__(fit_params=fit_params, **config)

    def get_refit_interface(self, n_refit: int, fit_params: Optional[List[Dict]] = None) -> 'AlgInterface':
        raise NotImplementedError()

    def fit(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources,
            logger: Logger, tmp_folders: List[Optional[Path]], name: str) -> Optional[
        List[List[List[Tuple[Dict, float]]]]]:
        assert len(idxs_list) == 1
        assert idxs_list[0].n_trainval_splits == 1

        seed = idxs_list[0].sub_split_seeds[0]
        print(f'Setting seed: {seed}')
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # hyperparams
        arch_type = self.config.get('arch_type', 'tabm')
        num_emb_type = self.config.get('num_emb_type', 'none')
        n_epochs = self.config.get('n_epochs', 1_000_000_000)
        patience = self.config.get('patience', 16)
        batch_size = self.config.get('batch_size', 256)
        compile_model = self.config.get('compile_model', False)
        lr = self.config.get('lr', 2e-3)
        d_embedding = self.config.get('d_embedding', 16)
        d_block = self.config.get('d_block', 512)
        dropout = self.config.get('dropout', 0.1)
        tabm_k = self.config.get('tabm_k', 32)
        allow_amp = self.config.get('allow_amp', False)
        n_blocks = self.config.get('n_blocks', 'auto')
        num_emb_n_bins = self.config.get('num_emb_n_bins', 48)

        weight_decay = self.config.get('weight_decay', 0.0)
        gradient_clipping_norm = self.config.get('gradient_clipping_norm', None)

        TaskType = Literal['regression', 'binclass', 'multiclass']

        n_train = idxs_list[0].n_train
        n_classes = ds.get_n_classes()
        cat_cardinalities = ds.tensor_infos['x_cat'].get_cat_sizes().numpy().tolist()
        task_type: TaskType = 'regression' if n_classes == 0 else ('binclass' if n_classes == 2 else 'multiclass')
        device = interface_resources.gpu_devices[0] if len(interface_resources.gpu_devices) >= 1 else 'cpu'
        device = torch.device(device)

        self.n_classes_ = n_classes
        self.task_type_ = task_type
        self.device_ = device

        # create preprocessing factory
        factory = self.config.get('factory', None)
        if 'tfms' not in self.config:
            self.config['tfms'] = ['quantile_tabr']
        if factory is None:
            factory = PreprocessingFactory(**self.config)

        if idxs_list[0].val_idxs is None:
            raise ValueError(f'Training without validation set is currently not implemented')

        ds_parts = {'train': ds.get_sub_dataset(idxs_list[0].train_idxs[0]),
                    'val': ds.get_sub_dataset(idxs_list[0].val_idxs[0]),
                    # 'test': ds.get_sub_dataset(idxs_list[0].test_idxs)
                    }

        part_names = ['train', 'val']  # no test
        non_train_part_names = ['val']

        # transform according to factory
        fitter: Fitter = factory.create(ds.tensor_infos)
        self.tfm_, ds_parts['train'] = fitter.fit_transform(ds_parts['train'])
        for part in non_train_part_names:
            ds_parts[part] = self.tfm_(ds_parts[part])

        # filter out numerical columns with only a single value
        x_cont_train = ds_parts['train'].tensors['x_cont']

        for part in part_names:
            ds_parts[part] = ds_parts[part].to(device)

        # mask of which columns are not constant
        self.num_col_mask_ = ~torch.all(x_cont_train == x_cont_train[0:1, :], dim=0)

        for part in part_names:
            ds_parts[part].tensors['x_cont'] = ds_parts[part].tensors['x_cont'][:, self.num_col_mask_]
            # tensor infos are not correct anymore, but might not be used either

        # update
        n_cont_features = ds_parts['train'].tensors['x_cont'].shape[1]

        Y_train = ds_parts['train'].tensors['y'].clone()
        if task_type == 'regression':
            assert ds.tensor_infos['y'].get_n_features() == 1
            self.y_mean_ = ds_parts['train'].tensors['y'].mean().item()
            self.y_std_ = ds_parts['train'].tensors['y'].std(correction=0).item()

            Y_train = (Y_train - self.y_mean_) / (self.y_std_ + 1e-30)

        data = {part: utils.join_dicts(
            dict(x_cont=ds_parts[part].tensors['x_cont'], y=ds_parts[part].tensors['y']),
            dict(x_cat=ds_parts[part].tensors['x_cat']) if ds.tensor_infos['x_cat'].get_n_features() > 0 else dict())
                for part in part_names}

        # adapted from https://github.com/yandex-research/tabm/blob/main/example.ipynb

        # Automatic mixed precision (AMP)
        # torch.float16 is implemented for completeness,
        # but it was not tested in the project,
        # so torch.bfloat16 is used by default.
        amp_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
            if torch.cuda.is_available()
            else None
        )
        # Changing False to True will result in faster training on compatible hardware.
        amp_enabled = allow_amp and amp_dtype is not None
        grad_scaler = torch.cuda.amp.GradScaler() if amp_dtype is torch.float16 else None  # type: ignore

        # fmt: off
        logger.log(1,
            f'Device:        {device.type.upper()}'
            f'\nAMP:           {amp_enabled} (dtype: {amp_dtype})'
            f'\ntorch.compile: {compile_model}'
        )
        # fmt: on
        pass

        # Choose one of the two configurations below.

        # TabM
        bins = None if num_emb_type != 'pwl' or n_cont_features == 0 else rtdl_num_embeddings.compute_bins(data['train']['x_cont'], n_bins=num_emb_n_bins)

        model = Model(
            n_num_features=n_cont_features,
            cat_cardinalities=cat_cardinalities,
            n_classes=n_classes if n_classes > 0 else None,
            backbone={
                'type': 'MLP',
                'n_blocks': n_blocks if n_blocks != 'auto' else (3 if bins is None else 2),
                'd_block': d_block,
                'dropout': dropout,
            },
            bins=bins,
            num_embeddings=(
                None
                if bins is None
                else {
                    'type': 'PiecewiseLinearEmbeddings',
                    'd_embedding': d_embedding,
                    'activation': False,
                    'version': 'B',
                }
            ),
            arch_type=arch_type,
            k=tabm_k,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


        if compile_model:
            # NOTE
            # `torch.compile` is intentionally called without the `mode` argument
            # (mode="reduce-overhead" caused issues during training with torch==2.0.1).
            model = torch.compile(model)
            evaluation_mode = torch.no_grad
        else:
            evaluation_mode = torch.inference_mode

        @torch.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype)  # type: ignore[code]
        def apply_model(part: str, idx: torch.Tensor) -> torch.Tensor:
            return (
                model(
                    data[part]['x_cont'][idx],
                    data[part]['x_cat'][idx] if 'x_cat' in data[part] else None,
                )
                .squeeze(-1)  # Remove the last dimension for regression tasks.
                .float()
            )

        base_loss_fn = torch.nn.functional.mse_loss if task_type == 'regression' else torch.nn.functional.cross_entropy

        def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            # TabM produces k predictions per object. Each of them must be trained separately.
            # (regression)     y_pred.shape == (batch_size, k)
            # (classification) y_pred.shape == (batch_size, k, n_classes)
            k = y_pred.shape[-1 if task_type == 'regression' else -2]
            return base_loss_fn(y_pred.flatten(0, 1), y_true.repeat_interleave(k))

        @evaluation_mode()
        def evaluate(part: str) -> float:
            model.eval()

            # When using torch.compile, you may need to reduce the evaluation batch size.
            eval_batch_size = 1024
            y_pred: np.ndarray = (
                torch.cat(
                    [
                        apply_model(part, idx)
                        for idx in torch.arange(len(data[part]['y']), device=device).split(
                        eval_batch_size
                    )
                    ]
                )
                .cpu()
                .numpy()
            )
            if task_type == 'regression':
                # Transform the predictions back to the original label space.
                y_pred = y_pred * self.y_std_ + self.y_mean_

            # Compute the mean of the k predictions.
            if task_type != 'regression':
                # For classification, the mean must be computed in the probability space.
                y_pred = scipy.special.softmax(y_pred, axis=-1)
            y_pred = y_pred.mean(1)

            y_true = data[part]['y'].cpu().numpy()
            score = (
                -(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5)
                if task_type == 'regression'
                else sklearn.metrics.accuracy_score(y_true, y_pred.argmax(1))
            )
            return float(score)  # The higher -- the better.

        # print(f'Test score before training: {evaluate("test"):.4f}')

        epoch_size = math.ceil(n_train / batch_size)
        best = {
            'val': -math.inf,
            # 'test': -math.inf,
            'epoch': -1,
        }
        best_params = [p.clone() for p in model.parameters()]
        # Early stopping: the training stops when
        # there are more than `patience` consequtive bad updates.
        remaining_patience = patience

        try:
            if self.config.get('verbosity', 0) >= 1:
                from tqdm.std import tqdm
            else:
                tqdm = lambda arr, desc, total: arr
        except ImportError:
            tqdm = lambda arr, desc, total: arr

        logger.log(1, '-' * 88 + '\n')
        for epoch in range(n_epochs):
            for batch_idx in tqdm(
                    torch.randperm(len(data['train']['y']), device=device).split(batch_size),
                    desc=f'Epoch {epoch}',
                    total=epoch_size,
            ):
                model.train()
                optimizer.zero_grad()
                loss = loss_fn(apply_model('train', batch_idx), Y_train[batch_idx])

                # added from https://github.com/yandex-research/tabm/blob/main/bin/model.py
                if gradient_clipping_norm is not None and gradient_clipping_norm != 'none':
                    if grad_scaler is not None:
                        grad_scaler.unscale_(optimizer)
                    nn.utils.clip_grad.clip_grad_norm_(
                        model.parameters(), gradient_clipping_norm
                    )

                if grad_scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    grad_scaler.scale(loss).backward()  # type: ignore
                    grad_scaler.step(optimizer)
                    grad_scaler.update()



            val_score = evaluate('val')
            # test_score = evaluate('test')
            # logger.log(1, f'(val) {val_score:.4f} (test) {test_score:.4f}')
            logger.log(1, f'(val) {val_score:.4f}')

            if val_score > best['val']:
                logger.log(1, '🌸 New best epoch! 🌸')
                # best = {'val': val_score, 'test': test_score, 'epoch': epoch}
                best = {'val': val_score, 'epoch': epoch}
                remaining_patience = patience
                with torch.no_grad():
                    for bp, p in zip(best_params, model.parameters()):
                        bp.copy_(p)
            else:
                remaining_patience -= 1

            if remaining_patience < 0:
                break

            logger.log(1, '')

        logger.log(1, '\n\nResult:')
        logger.log(1, str(best))

        logger.log(1, f'Restoring best model')
        with torch.no_grad():
            for bp, p in zip(best_params, model.parameters()):
                p.copy_(bp)

        self.model_ = model

        return None

    def predict(self, ds: DictDataset) -> torch.Tensor:
        self.model_.eval()

        ds = self.tfm_(ds).to(self.device_)

        ds.tensors['x_cont'] = ds.tensors['x_cont'][:, self.num_col_mask_]

        eval_batch_size = 1024
        with torch.no_grad():
            y_pred: torch.Tensor = (
                torch.cat(
                    [
                        self.model_(
                            ds.tensors['x_cont'][idx],
                            ds.tensors['x_cat'][idx] if not ds.tensor_infos['x_cat'].is_empty() else None,
                        )
                        .squeeze(-1)  # Remove the last dimension for regression tasks.
                        .float()
                        for idx in torch.arange(ds.n_samples, device=self.device_).split(
                        eval_batch_size
                    )
                    ]
                )
            )
        if self.task_type_ == 'regression':
            # Transform the predictions back to the original label space.
            y_pred = y_pred * self.y_std_ + self.y_mean_
            y_pred = y_pred.mean(1)
            y_pred = y_pred.unsqueeze(-1)  # add extra "features" dimension
        else:
            # For classification, the mean must be computed in the probability space.
            y_pred = torch.log(torch.softmax(y_pred, dim=-1).mean(1) + 1e-30)

        return y_pred[None]  # add n_models dimension

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        assert n_cv == 1
        assert n_refit == 0
        assert n_splits == 1
        updated_config = utils.join_dicts(dict(n_estimators=100, max_n_threads=2), self.config)
        time_params = {'': 10, 'ds_onehot_size_gb': 10.0, 'n_train': 8e-5, 'n_samples*n_features': 8e-8}
        ram_params = {'': 0.15, 'ds_onehot_size_gb': 2.0}
        # gpu_ram_params = {'': 0.3, 'ds_onehot_size_gb': 1.0, 'n_train': 1e-6, 'n_features': 3e-4,
        #                   'cat_size_sum': 2e-3}
        gpu_ram_params = {'': 0.5, 'ds_onehot_size_gb': 5.0, 'n_train': 6e-6, 'n_features': 2e-3,
                          'cat_size_sum': 1e-3}
        rc = ResourcePredictor(config=updated_config, time_params=time_params, gpu_ram_params=gpu_ram_params,
                               cpu_ram_params=ram_params, n_gpus=1, gpu_usage=0.02)  # , gpu_ram_params)
        return rc.get_required_resources(ds, n_train=n_train)
