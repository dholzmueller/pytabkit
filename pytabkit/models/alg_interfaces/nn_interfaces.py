import copy
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import numpy as np
import torch
try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl

import logging

from datetime import timedelta

from pytabkit.models import utils
from pytabkit.models.data.data import DictDataset
from pytabkit.models.hyper_opt.hyper_optimizers import HyperoptOptimizer, SMACOptimizer
from pytabkit.models.nn_models.base import Layer, Variable
from pytabkit.models.nn_models.models import NNFactory
from pytabkit.models.sklearn.default_params import DefaultParams
from pytabkit.models.torch_utils import cat_if_necessary
from pytabkit.models.training.lightning_modules import TabNNModule
from pytabkit.models.training.logging import Logger
from pytabkit.models.alg_interfaces.alg_interfaces import AlgInterface, SingleSplitAlgInterface, OptAlgInterface
from pytabkit.models.alg_interfaces.base import SplitIdxs, InterfaceResources, RequiredResources


class NNAlgInterface(AlgInterface):
    def __init__(self, fit_params: Optional[List[Dict[str, Any]]] = None, **config):
        super().__init__(fit_params=fit_params, **config)
        self.model: Optional[TabNNModule] = None
        self.trainer: Optional[pl.Trainer] = None
        self.device = None

    def get_refit_interface(self, n_refit: int, fit_params: Optional[List[Dict]] = None) -> 'AlgInterface':
        return NNAlgInterface(fit_params if fit_params is not None else self.fit_params, **self.config)

    def fit(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources,
            logger: Logger, tmp_folders: List[Optional[Path]], name: str):
        # the code below requires all splits to have the same number of sub-splits
        assert np.all([idxs_list[i].train_idxs.shape[0] == idxs_list[0].train_idxs.shape[0]
                       for i in range(len(idxs_list))])
        # we can then decompose the overall number of sub-splits into the number of splits
        # and the number of sub-splits per split

        # have the option to change the seeds (for comparing NNs with different random seeds)
        random_seed_offset = self.config.get('random_seed_offset', 0)
        if random_seed_offset != 0:
            idxs_list = [SplitIdxs(train_idxs=idxs.train_idxs, val_idxs=idxs.val_idxs,
                                   test_idxs=idxs.test_idxs, split_seed=idxs.split_seed + random_seed_offset,
                                   sub_split_seeds=[seed + random_seed_offset for seed in idxs.sub_split_seeds],
                                   split_id=idxs.split_id) for idxs in idxs_list]

        # https://stackoverflow.com/questions/74364944/how-to-get-rid-of-info-logging-messages-in-pytorch-lightning
        log = logging.getLogger("lightning")
        log.propagate = False
        log.setLevel(logging.ERROR)

        warnings.filterwarnings("ignore", message="You defined a `validation_step` but have no `val_dataloader`.")

        old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False  # to be safe wrt rounding errors, but might not be necessary
        # todo: allow preprocessing on CPU and then only put batches on GPU in data loader?
        gpu_devices = interface_resources.gpu_devices
        self.device = gpu_devices[0] if len(gpu_devices) > 0 else 'cpu'
        ds = ds.to(self.device)

        n_epochs = self.config.get('n_epochs', 256)
        self.model = TabNNModule(**utils.join_dicts({'n_epochs': 256, 'logger': logger}, self.config),
                                 fit_params=self.fit_params)
        self.model.compile_model(ds, idxs_list, interface_resources)

        if self.device == 'cpu':
            pl_accelerator = 'cpu'
            pl_devices = 'auto'
        elif self.device == 'mps':
            pl_accelerator = 'mps'
            pl_devices = 'auto'
        elif self.device == 'cuda':
            pl_accelerator = 'gpu'
            pl_devices = [0]
        elif self.device.startswith('cuda:'):
            pl_accelerator = 'gpu'
            pl_devices = [int(self.device[len('cuda:'):])]
        else:
            raise ValueError(f'Unknown device "{self.device}"')

        max_time = None if interface_resources.time_in_seconds is None else timedelta(seconds=interface_resources.time_in_seconds)

        self.trainer = pl.Trainer(
            max_time=max_time,
            accelerator=pl_accelerator,
            devices=pl_devices,
            callbacks=self.model.create_callbacks(),
            max_epochs=n_epochs,
            enable_checkpointing=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            logger=pl.loggers.logger.DummyLogger(),
            enable_model_summary=False,
            log_every_n_steps=1,
        )

        self.trainer.fit(
            model=self.model, train_dataloaders=self.model.train_dl, val_dataloaders=self.model.val_dl
        )

        if hasattr(self.model, 'fit_params'):
            self.fit_params = self.model.fit_params

        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32

        # self.model.to('cpu')  # to allow serialization without GPU issues, but doesn't work

        # print(f'Importances (sorted):', self.get_importances().sort()[0])  # todo
        self.trainer.max_time = None

    def predict(self, ds: DictDataset) -> torch.Tensor:
        old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.model.to(self.device)
        ds = ds.to(self.device)
        ds_x, _ = ds.split_xy()
        y_pred = self.trainer.predict(model=self.model, dataloaders=self.model.get_predict_dataloader(ds_x))
        y_pred = cat_if_necessary(y_pred, dim=-2).to('cpu')  # concat along batch dimension
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
        # self.model.to('cpu')  # to allow serialization without GPU issues, but doesn't work
        return y_pred

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        tensor_infos = ds.tensor_infos
        factory = self.config.get('factory', None)
        if factory is None:
            factory = NNFactory(**self.config)
        fitter = factory.create(tensor_infos)
        static_fitter, dynamic_fitter = fitter.split_off_dynamic()
        static_tensor_infos = static_fitter.forward_tensor_infos(tensor_infos)
        n_params = fitter.get_n_params(tensor_infos)
        n_forward = fitter.get_n_forward(tensor_infos)
        n_parallel = max(n_cv, n_refit) * n_splits
        batch_size = self.config.get('batch_size', 256)
        n_epochs = self.config.get('n_epochs', 256)
        # per-element RAM usage:
        # continuous data requires 4 bytes for forward pass and 4 for backward pass
        # categorical data requires 8 bytes for forward pass (because torch.long is required) and none for backward pass
        pass_memory = n_forward * batch_size * 8  # initial batch size ignored
        ds_size_gb = ds.n_samples * sum([ti.get_n_features() * (8 if ti.is_cat() else 4)
                                         for ti in static_tensor_infos.values()]) / (1024 ** 3)
        ds_ram_gb = 5 * ds_size_gb
        # ds_ram_gb = 3 * task_info.get_ds_size_gb() / (1024**3)
        param_memory = 5 * n_params * 8  # 5 because of model, model copy, grads, adam mom, adam sq_mom
        fixed_ram_gb = 0.3  # go safe

        # print(f'{pass_memory=}, {param_memory=}')

        # max memory that would be used if the dataset wasn't used
        init_ram_gb_full = n_forward * ds.n_samples * 8 / (1024**3)
        init_ram_gb_max = 1.2  # todo: rough estimate, a bit larger than what is allowed in fit_transform_subsample()
        init_ram_gb = min(init_ram_gb_max, init_ram_gb_full)
        # init_ram_gb = 1.5

        factor = 1.2  # to go safe on ram
        gpu_ram_gb = fixed_ram_gb + ds_ram_gb + max(init_ram_gb,
                                                    factor * (n_parallel * (pass_memory + param_memory)) / (1024 ** 3))

        gpu_usage = min(1.0, n_parallel / 200)  # rather underestimate it and use up all the ram on the gpu
        # go somewhat safe, should be small anyway
        cpu_ram_gb = 0.3 + ds_ram_gb + 1.3 * (pass_memory + param_memory) / (1024 ** 3)

        time_approx = ds.n_samples * n_epochs * 4e-5 * (2 if n_refit > 0 else 1)
        if self.config.get('use_gpu', True):
            return RequiredResources(time_s=time_approx, n_threads=1.0, cpu_ram_gb=cpu_ram_gb,
                                     n_gpus=1, gpu_usage=gpu_usage, gpu_ram_gb=gpu_ram_gb)
        else:
            return RequiredResources(time_s=time_approx, n_threads=1.0, cpu_ram_gb=cpu_ram_gb + gpu_ram_gb)

    def get_model_ram_gb(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int]):
        tensor_infos = ds.tensor_infos
        factory = self.config.get('factory', None)
        if factory is None:
            factory = NNFactory(**self.config)
        fitter = factory.create(tensor_infos)
        n_params = fitter.get_n_params(tensor_infos)
        n_parallel = max(n_cv, n_refit) * n_splits

        factor = 1.2  # to go safe on ram
        return factor * n_parallel * n_params * 4 / (1024 ** 3)

    def get_importances(self) -> torch.Tensor:
        net: Layer = self.model.model
        params = net.parameters()
        scale = None
        weight = None
        importances_param = self.config.get('feature_importances', None)
        for param in params:
            param: Variable = param
            scope_str = str(param.context.scope)
            if scope_str.endswith('layer-0/scale'):
                scale = param
            elif scope_str.endswith('layer-0/weight'):
                weight = param

            # print(scope_str)

        assert weight is not None

        with torch.no_grad():
            # shape: (vectorized network dims) x n_features
            importances = weight.norm(dim=-1)

            if scale is not None:
                importances *= scale[..., 0, :].abs()

            p = self.config.get('importances_exponent', 1.0)
            importances = importances ** p
            #
            # # hard feature selection
            # n_remove = int(0.9 * importances.shape[-1])
            # new_importances = torch.ones_like(importances)
            # for i in range(importances.shape[0]):
            #     new_importances[i, torch.argsort(importances[i])[:n_remove]] = 0.0
            # importances = new_importances
            # print(importances)

            if importances_param is not None:
                print(f'Using importances_param')
                importances *= importances_param[..., :]

            importances /= (importances.norm(dim=-1, keepdim=True) / np.sqrt(importances.shape[-1]))
            return importances

    def get_first_layer_weights(self, with_scale: bool) -> torch.Tensor:
        net: Layer = self.model.model
        params = net.parameters()
        scale = None
        weight = None
        for param in params:
            param: Variable = param
            scope_str = str(param.context.scope)
            if scope_str.endswith('layer-0/scale'):
                scale = param
            elif scope_str.endswith('layer-0/weight'):
                weight = param
        assert weight is not None
        if scale is not None and with_scale:
            with torch.no_grad():
                return weight * scale[..., 0, :, None]
        else:
            return weight.data

    # todo: have option to move to/from GPU


class NNHyperoptAlgInterface(OptAlgInterface):
    def __init__(self, space: Optional[Union[str, Dict[str, Any]]] = None, n_hyperopt_steps: int = 50,
                 opt_method: str = 'hyperopt', **config):
        from hyperopt import hp
        default_config = config  # todo
        max_config = copy.copy(default_config)
        if space == 'default':
            space = {
                'lr': hp.loguniform('lr', np.log(2e-2), np.log(3e-1)),
                'num_emb_type': hp.choice('num_emb_type', ['none', 'pl', 'plr', 'pbld']),
                'add_front_scale': hp.choice('add_front_scale', [(0.6, True), (0.4, False)]),
                'p_drop': hp.choice('p_drop', [(0.3, 0.0), (0.5, 0.15), (0.2, 0.3)]),
                'wd': hp.choice('wd', [0.0, 0.02]),
                'plr_sigma': hp.loguniform('plr_sigma', np.log(0.05), np.log(0.5)),
                'act': hp.choice('act', ['relu', 'selu', 'mish']),
                'hidden_sizes': hp.choice('hidden_sizes', [(0.6, [256]*3), (0.2, [512]), (0.2, [64]*5)]),
                'ls_eps': hp.choice('ls_eps', [(0.3, 0.0), (0.7, 0.1)])
            }
            utils.update_dict(default_config, remove_keys=list(space.keys()))
        elif not isinstance(space, dict):
            print(f'Unkown hyperparameter space: {space}')

        config = utils.update_dict(default_config, config)
        opt_class = SMACOptimizer if opt_method == 'smac' else HyperoptOptimizer
        super().__init__(hyper_optimizer=opt_class(space=space, fixed_params=default_config,
                                                   n_hyperopt_steps=n_hyperopt_steps,
                                                   **config),
                         max_resource_config=utils.join_dicts(config),
                         **config)

    def create_alg_interface(self, n_sub_splits: int, **config) -> AlgInterface:
        return NNAlgInterface(**config)

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        required_resources = super().get_required_resources(ds, n_cv, n_refit, n_splits, split_seeds, n_train=n_train)

        # add n_steps * model_ram_gb to required resources, because these will be stored
        alg_interface = NNAlgInterface(**self.max_resource_config)
        model_ram_gb = alg_interface.get_model_ram_gb(ds, n_cv, n_refit, n_splits, split_seeds)
        required_resources.cpu_ram_gb += self.hyper_optimizer.get_n_hyperopt_steps() * model_ram_gb
        return required_resources


class RealMLPParamSampler:
    def __init__(self, is_classification: bool, hpo_space_name: str = 'default', **config):
        self.is_classification = is_classification
        self.hpo_space_name = hpo_space_name

    def sample_params(self, seed: int) -> Dict[str, Any]:
        assert self.hpo_space_name in ['default', 'clr', 'moresigma', 'moresigmadim', 'moresigmadimreg',
                                       'moresigmadimsize', 'moresigmadimlr']
        rng = np.random.default_rng(seed=seed)

        hidden_size_options = [[256] * 3, [64] * 5, [512]]

        params = {'num_emb_type': rng.choice(['none', 'pbld', 'pl', 'plr']),
                  'add_front_scale': rng.choice([True, False], p=[0.6, 0.4]),
                  'lr': np.exp(rng.uniform(np.log(2e-2), np.log(3e-1))),
                  'p_drop': rng.choice([0.0, 0.15, 0.3], p=[0.3, 0.5, 0.2]),
                  'wd': rng.choice([0.0, 2e-2]),
                  'plr_sigma': np.exp(rng.uniform(np.log(0.05), np.log(0.5))),
                  'act': rng.choice(['relu', 'selu', 'mish']),
                  'hidden_sizes': hidden_size_options[rng.choice([0, 1, 2], p=[0.6, 0.2, 0.2])]}

        if self.is_classification:
            params['ls_eps'] = rng.choice([0.0, 0.1], p=[0.3, 0.7])

        if self.hpo_space_name == 'clr':
            params['lr'] = np.exp(rng.uniform(np.log(2e-3), np.log(3e-1)))
            params['lr_sched'] = 'constant'
            params['use_early_stopping'] = True
            params['early_stopping_multiplicative_patience'] = 1
            params['early_stopping_additive_patience'] = 16
        elif self.hpo_space_name == 'moresigma':
            params['plr_sigma'] = np.exp(rng.uniform(np.log(1e-2), np.log(1e1)))
        elif self.hpo_space_name == 'moresigmadim':
            params['plr_sigma'] = np.exp(rng.uniform(np.log(1e-2), np.log(1e1)))
            params['plr_hidden_1'] = 2*round(np.exp(rng.uniform(np.log(1), np.log(32))))
            params['plr_hidden_2'] = round(np.exp(rng.uniform(np.log(2), np.log(64))))
        elif self.hpo_space_name == 'moresigmadimreg':
            params['plr_sigma'] = np.exp(rng.uniform(np.log(1e-2), np.log(1e1)))
            params['plr_hidden_1'] = 2*round(np.exp(rng.uniform(np.log(1), np.log(32))))
            params['plr_hidden_2'] = round(np.exp(rng.uniform(np.log(2), np.log(64))))
            params['p_drop'] = rng.choice([0.0, rng.uniform(0.0, 0.5)])
            params['wd'] = np.exp(rng.uniform(np.log(1e-5), np.log(4e-2)))
        elif self.hpo_space_name == 'moresigmadimsize':
            params['plr_sigma'] = np.exp(rng.uniform(np.log(1e-2), np.log(1e1)))
            params['plr_hidden_1'] = 2*round(np.exp(rng.uniform(np.log(1), np.log(32))))
            params['plr_hidden_2'] = round(np.exp(rng.uniform(np.log(2), np.log(64))))
            params['hidden_sizes'] = [rng.choice(np.arange(8, 513))] * rng.choice(np.arange(1, 6))
        elif self.hpo_space_name == 'moresigmadimlr':
            params['plr_sigma'] = np.exp(rng.uniform(np.log(1e-2), np.log(1e1)))
            params['plr_hidden_1'] = 2*round(np.exp(rng.uniform(np.log(1), np.log(32))))
            params['plr_hidden_2'] = round(np.exp(rng.uniform(np.log(2), np.log(64))))
            params['lr'] = np.exp(rng.uniform(np.log(5e-3), np.log(5e-1)))
        # print(f'{params=}')

        default_params = DefaultParams.RealMLP_TD_CLASS if self.is_classification else DefaultParams.RealMLP_TD_REG
        return utils.join_dicts(default_params, params)


class RandomParamsNNAlgInterface(SingleSplitAlgInterface):
    def __init__(self, model_idx: int, fit_params: Optional[List[Dict[str, Any]]] = None, **config):
        # model_idx is used for seeding along with the seed given in fit(),
        # so we can do HPO by combining multiple RandomParamsNNAlgInterface objects with different model_idx values
        super().__init__(fit_params=fit_params, **config)
        self.model_idx = model_idx
        self.alg_interface = None

    def get_refit_interface(self, n_refit: int, fit_params: Optional[List[Dict]] = None) -> 'AlgInterface':
        raise NotImplementedError('Refit is not fully implemented...')
        # return RandomParamsNNAlgInterface(model_idx=self.model_idx, fit_params=fit_params or self.fit_params,
        #                                   **self.config)

    def _create_sub_interface(self, ds: DictDataset, seed: int):
        # this is also set in get_required_resources, but okay
        if self.fit_params is None:
            hparam_seed = utils.combine_seeds(seed, self.model_idx)
            is_classification = not ds.tensor_infos['y'].is_cont()
            self.fit_params = [RealMLPParamSampler(is_classification, **self.config).sample_params(hparam_seed)]
        # todo: need epoch for refit
        return NNAlgInterface(fit_params=None, **utils.update_dict(self.config, self.fit_params[0]))

    def fit(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources,
            logger: Logger, tmp_folders: List[Optional[Path]], name: str) -> None:
        assert len(idxs_list) == 1
        self.alg_interface = self._create_sub_interface(ds, idxs_list[0].split_seed)
        logger.log(1, f'{self.fit_params=}')
        self.alg_interface.fit(ds, idxs_list, interface_resources, logger, tmp_folders, name)

    def predict(self, ds: DictDataset) -> torch.Tensor:
        return self.alg_interface.predict(ds)

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        assert len(split_seeds) == 1
        alg_interface = self._create_sub_interface(ds, split_seeds[0])
        return alg_interface.get_required_resources(ds, n_cv, n_refit, n_splits, split_seeds, n_train=n_train)


# class NNHyperoptAlgInterface(OptAlgInterface):
#     def __init__(self, space=None, n_hyperopt_steps: int = 50, **config):
#         from hyperopt import hp
#         default_config = {}
#         max_config = {}
#         if space is None:
#             space = {
#                 'num_emb_type': hp.choice(['none', 'pl-densenet', 'plr']),
#                 'add_front_scale': hp.choice([True, False]),
#                 'lr': hp.loguniform([2e-2, 1.5e-1]),
#                 'p_drop': hp.choice([0.0, 0.15, 0.3, 0.45]),
#                 'hidden_sizes': hp.choice([[256]*3, [512]]),
#                 'act': hp.choice(['selu', 'mish', 'relu']),
#                 'ls_eps': hp.choice([0.0, 1.0])
#             }
#         # todo: have conversion function?
#         config = utils.update_dict(default_config, config)
#         super().__init__(hyper_optimizer=HyperoptOptimizer(space=space, fixed_params=dict(),
#                                                            n_hyperopt_steps=n_hyperopt_steps,
#                                                            **config),
#                          max_resource_config=utils.join_dicts(config, max_config),
#                          **config)
#
#     def create_alg_interface(self, n_sub_splits: int, **config) -> AlgInterface:
#         return NNAlgInterface(**config)
