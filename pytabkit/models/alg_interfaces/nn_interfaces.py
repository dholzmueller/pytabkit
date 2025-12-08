import copy
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

import numpy as np
import torch

from pytabkit.models.training.nn_creator import get_realmlp_auto_batch_size

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
from pytabkit.models.training.lightning_modules import TabNNModule, postprocess_multiquantile
from pytabkit.models.training.logging import Logger
from pytabkit.models.alg_interfaces.alg_interfaces import AlgInterface, SingleSplitAlgInterface, OptAlgInterface
from pytabkit.models.alg_interfaces.base import SplitIdxs, InterfaceResources, RequiredResources


def get_lignting_accel_and_devices(device: str):
    if device == 'cpu':
        pl_accelerator = 'cpu'
        pl_devices = 'auto'
    elif device == 'mps':
        pl_accelerator = 'mps'
        pl_devices = 'auto'
    elif device == 'cuda':
        pl_accelerator = 'gpu'
        pl_devices = [0]
    elif device.startswith('cuda:'):
        pl_accelerator = 'gpu'
        pl_devices = [int(device[len('cuda:'):])]
    else:
        raise ValueError(f'Unknown device "{device}"')

    return pl_accelerator, pl_devices


class NNAlgInterface(AlgInterface):
    def __init__(self, fit_params: Optional[List[Dict[str, Any]]] = None, **config):
        super().__init__(fit_params=fit_params, **config)
        self.model: Optional[TabNNModule] = None
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

        # print(f'Starting NN fit')

        # have the option to change the seeds (for comparing NNs with different random seeds)
        random_seed_offset = self.config.get('random_seed_offset', 0)
        if random_seed_offset != 0:
            idxs_list = [SplitIdxs(train_idxs=idxs.train_idxs, val_idxs=idxs.val_idxs,
                                   test_idxs=idxs.test_idxs, split_seed=idxs.split_seed + random_seed_offset,
                                   sub_split_seeds=[seed + random_seed_offset for seed in idxs.sub_split_seeds],
                                   split_id=idxs.split_id) for idxs in idxs_list]
        if self.config.get('same_seed_for_sub_splits', False):
            idxs_list = [SplitIdxs(train_idxs=idxs.train_idxs, val_idxs=idxs.val_idxs,
                                   test_idxs=idxs.test_idxs, split_seed=idxs.split_seed,
                                   sub_split_seeds=[idxs.sub_split_seeds[0]] * len(idxs.sub_split_seeds),
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

        fit_params = self.fit_params
        if self.fit_params is None and 'stop_epoch' in self.config:
            fit_params = [dict(stop_epoch=self.config['stop_epoch'])] * len(idxs_list)

        n_epochs = self.config.get('n_epochs', 256)
        self.model = TabNNModule(**utils.join_dicts({'n_epochs': 256, 'logger': logger}, self.config),
                                 fit_params=fit_params)
        self.model.compile_model(ds, idxs_list, interface_resources)

        pl_accelerator, pl_devices = get_lignting_accel_and_devices(self.device)

        max_time = None if interface_resources.time_in_seconds is None else timedelta(
            seconds=interface_resources.time_in_seconds)

        self.min_trainer_kwargs = dict(
            max_time=max_time,
            accelerator=pl_accelerator,
            devices=pl_devices,
            max_epochs=n_epochs,
            enable_checkpointing=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            enable_model_summary=False,
            log_every_n_steps=1,
        )

        # don't save the trainer in self, otherwise it stores the dataset
        trainer = pl.Trainer(
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

        trainer.fit(
            model=self.model, train_dataloaders=self.model.train_dl, val_dataloaders=self.model.val_dl
        )

        if hasattr(self.model, 'fit_params'):
            self.fit_params = self.model.fit_params

        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32

        # remove all remaining references to GPU tensors, for some reason this can't be done in the model
        del self.model._trainer

        # self.model.to('cpu')  # to allow serialization without GPU issues, but doesn't work

        # print(f'Importances (sorted):', self.get_importances().sort()[0])

    def predict(self, ds: DictDataset) -> torch.Tensor:
        pred_dict = self.get_current_predict_params_dict()
        if 'val_metric_name' in pred_dict:
            self.model.restore_ckpt_for_val_metric_name(pred_dict['val_metric_name'])
        old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.model.to(self.device)
        ds = ds.to(self.device)
        ds_x, _ = ds.split_xy()

        pl_accelerator, pl_devices = get_lignting_accel_and_devices(self.device)

        # create new trainer so we don't have to pickle the full trainer that references the dataset somehow
        # update devices since the model device may have been moved since
        trainer = pl.Trainer(**(self.min_trainer_kwargs | dict(accelerator=pl_accelerator, devices=pl_devices)))
        y_pred = trainer.predict(model=self.model, dataloaders=self.model.get_predict_dataloader(ds_x))
        y_pred = cat_if_necessary(y_pred, dim=-2).to('cpu')  # concat along batch dimension
        y_pred = postprocess_multiquantile(y_pred, **self.config)  # postprocessing in case of multiquantile loss
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32

        # remove all remaining references to GPU tensors, for some reason this can't be done in the model
        del self.model._trainer

        return y_pred

    def get_available_predict_params(self) -> Dict[str, Dict[str, Any]]:
        val_metric_names = self.config.get('val_metric_names', None)
        if val_metric_names is None:
            return {'': dict()}
        else:
            return {f'_val-{val_metric_name}': dict(val_metric_name=val_metric_name) for val_metric_name in
                    val_metric_names}

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
        n_parallel = max(n_cv, n_refit) * n_splits * self.config.get('n_ens', 1)
        batch_size = self.config.get('batch_size', 256)
        if batch_size == 'auto':
            batch_size = get_realmlp_auto_batch_size(n_train)
        # print(f'{batch_size=}')
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
        init_ram_gb_full = n_forward * ds.n_samples * 8 / (1024 ** 3)
        init_ram_gb_max = 1.2  # todo: rough estimate, a bit larger than what is allowed in fit_transform_subsample()
        init_ram_gb = min(init_ram_gb_max, init_ram_gb_full)
        # init_ram_gb = 1.5

        # print(f'{ds_ram_gb=}, {pass_memory/(1024**3)=}, {param_memory/(1024**3)=}, {init_ram_gb=}')

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

    def to(self, device: str) -> None:
        # print(f'Move RealMLP model to device {device}')
        self.model.to(device)
        self.device = device

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
                'hidden_sizes': hp.choice('hidden_sizes', [(0.6, [256] * 3), (0.2, [512]), (0.2, [64] * 5)]),
                'ls_eps': hp.choice('ls_eps', [(0.3, 0.0), (0.7, 0.1)])
            }
            utils.update_dict(default_config, remove_keys=list(space.keys()))
        elif not isinstance(space, dict):
            print(f'Unknown hyperparameter space: {space}')

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
                                       'moresigmadimsize', 'moresigmadimlr', 'probclass', 'probclass-mlp', 'large',
                                       'alt1', 'alt2', 'alt3', 'alt4', 'alt5', 'alt6', 'alt7', 'alt8', 'alt9', 'alt10',
                                       'tabarena', 'tabarena-new', 'alt11', 'alt12', 'alt13', 'alt14', 'alt15', 'alt16',
                                       'alt17', 'alt18', 'alt19', 'alt20']
        rng = np.random.default_rng(seed=seed)

        if self.hpo_space_name == 'probclass-mlp':
            params = {'lr': np.exp(rng.uniform(np.log(1e-4), np.log(1e-2))),
                      'p_drop': rng.choice([0.0, 0.1, 0.2, 0.3]),
                      'wd': rng.choice([0.0, 1e-5, 1e-4, 1e-3])}
            default_params = DefaultParams.VANILLA_MLP_CLASS if self.is_classification else DefaultParams.VANILLA_MLP_REG
            return utils.join_dicts(default_params, params)

        hidden_size_options = [[256] * 3, [64] * 5, [512]]

        params = {'num_emb_type': rng.choice(['none', 'pbld', 'pl', 'plr']),
                  'add_front_scale': rng.choice([True, False], p=[0.6, 0.4]),
                  # convert to actual bool so it can be serialized
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
            params['plr_hidden_1'] = 2 * round(np.exp(rng.uniform(np.log(1), np.log(32))))
            params['plr_hidden_2'] = round(np.exp(rng.uniform(np.log(2), np.log(64))))
        elif self.hpo_space_name == 'moresigmadimreg':
            params['plr_sigma'] = np.exp(rng.uniform(np.log(1e-2), np.log(1e1)))
            params['plr_hidden_1'] = 2 * round(np.exp(rng.uniform(np.log(1), np.log(32))))
            params['plr_hidden_2'] = round(np.exp(rng.uniform(np.log(2), np.log(64))))
            params['p_drop'] = rng.choice([0.0, rng.uniform(0.0, 0.5)])
            params['wd'] = np.exp(rng.uniform(np.log(1e-5), np.log(4e-2)))
        elif self.hpo_space_name == 'moresigmadimsize':
            params['plr_sigma'] = np.exp(rng.uniform(np.log(1e-2), np.log(1e1)))
            params['plr_hidden_1'] = 2 * round(np.exp(rng.uniform(np.log(1), np.log(32))))
            params['plr_hidden_2'] = round(np.exp(rng.uniform(np.log(2), np.log(64))))
            params['hidden_sizes'] = [rng.choice(np.arange(8, 513))] * rng.choice(np.arange(1, 6))
        elif self.hpo_space_name == 'moresigmadimlr':
            params['plr_sigma'] = np.exp(rng.uniform(np.log(1e-2), np.log(1e1)))
            params['plr_hidden_1'] = 2 * round(np.exp(rng.uniform(np.log(1), np.log(32))))
            params['plr_hidden_2'] = round(np.exp(rng.uniform(np.log(2), np.log(64))))
            params['lr'] = np.exp(rng.uniform(np.log(5e-3), np.log(5e-1)))
        elif self.hpo_space_name == 'probclass':
            params['ls_eps'] = rng.choice([0.0, 0.1])
            params['wd'] = rng.choice([0.0, 2e-3, 2e-2])
        elif self.hpo_space_name == 'large':
            params = {'num_emb_type': rng.choice(['none', 'pbld', 'pl', 'plr']),
                      'add_front_scale': rng.choice([True, False], p=[0.6, 0.4]),
                      'n_hidden': round(np.exp(rng.uniform(np.log(64), np.log(512)))),
                      'n_layers': rng.integers(1, 5, endpoint=True),
                      'lr': np.exp(rng.uniform(np.log(1e-2), np.log(5e-1))),
                      'p_drop': rng.uniform(0.0, 0.6),
                      'wd': rng.choice([rng.uniform(0.0, 1e-3), np.exp(rng.uniform(np.log(1e-3), np.log(1e-1)))]),
                      'plr_sigma': np.exp(rng.uniform(np.log(1e-2), np.log(1e2))),
                      'act': rng.choice(['relu', 'selu', 'mish', 'silu', 'gelu']),
                      'use_parametric_act': rng.choice([False, True]),
                      'p_drop_sched': rng.choice(['flat_cos', 'constant']),
                      'wd_sched': rng.choice(['flat_cos', 'constant']),
                      'ls_eps': rng.choice([0.0, rng.uniform(0.0, 0.2)]),
                      'lr_sched': rng.choice(['coslog4', 'cos']),
                      'sq_mom': 1.0 - np.exp(rng.uniform(np.log(1e-3), np.log(1e-1))),
                      'plr_lr_factor': np.exp(rng.uniform(np.log(3e-2), np.log(3e-1))),
                      }

            params['hidden_sizes'] = [params['n_hidden']] * params['n_layers']
        elif self.hpo_space_name == 'alt1':
            params = {'num_emb_type': rng.choice(['none', 'pbld']),
                      'n_hidden': rng.choice([128, 256, 384]),
                      'n_layers': rng.integers(1, 3, endpoint=True),
                      'lr': np.exp(rng.uniform(np.log(2e-2), np.log(3e-1))),
                      'p_drop': rng.uniform(0.0, 0.5),
                      'wd': np.exp(rng.uniform(np.log(5e-3), np.log(5e-2))),
                      'plr_sigma': np.exp(rng.uniform(np.log(1e-2), np.log(1e1))),
                      'act': rng.choice(['selu', 'mish', 'silu']),
                      # 'use_parametric_act': rng.choice([False, True]),
                      # 'p_drop_sched': rng.choice(['flat_cos', 'constant']),
                      # 'wd_sched': rng.choice(['flat_cos', 'constant']),
                      'ls_eps': rng.choice([0.0, np.exp(rng.uniform(np.log(5e-3), np.log(5e-2)))]),
                      # 'lr_sched': rng.choice(['coslog4', 'cos']),
                      'sq_mom': 1.0 - np.exp(rng.uniform(np.log(5e-3), np.log(1e-1))),
                      'plr_lr_factor': np.exp(rng.uniform(np.log(3e-2), np.log(3e-1))),
                      'scale_lr_factor': np.exp(rng.uniform(np.log(2.0), np.log(10.0))),
                      'use_early_stopping': True,
                      'early_stopping_multiplicative_patience': 2,
                      'early_stopping_additive_patience': 20,
                      }

            params['hidden_sizes'] = [params['n_hidden']] * params['n_layers']

        elif self.hpo_space_name == 'alt2':
            # refined version of large
            params = {'num_emb_type': 'pbld',
                      'n_hidden': round(np.exp(rng.uniform(np.log(198), np.log(512)))),
                      'n_layers': rng.integers(1, 3, endpoint=True),
                      'lr': np.exp(rng.uniform(np.log(1e-2), np.log(5e-1))),
                      'p_drop': rng.uniform(0.06, 0.6),
                      'wd': np.exp(rng.uniform(np.log(6e-3), np.log(1e-1))),
                      'plr_sigma': np.exp(rng.uniform(np.log(1e-2), np.log(15))),
                      'act': rng.choice(['mish', 'silu']),
                      'wd_sched': rng.choice(['flat_cos', 'constant']),
                      'ls_eps': rng.choice([0.0, np.exp(rng.uniform(np.log(5e-3), np.log(5e-2)))]),
                      'sq_mom': 1.0 - np.exp(rng.uniform(np.log(1e-3), np.log(1e-1))),
                      'plr_lr_factor': np.exp(rng.uniform(np.log(3e-2), np.log(3e-1))),
                      'scale_lr_factor': np.exp(rng.uniform(np.log(2.0), np.log(10.0))),
                      'p_drop_sched': 'constant',
                      }
            params['hidden_sizes'] = [params['n_hidden']] * params['n_layers']

        elif self.hpo_space_name == 'alt3':
            # refined version of alt2 (better for 20 steps but worse for 50)
            params = {'num_emb_type': 'pbld',
                      'n_hidden': round(np.exp(rng.uniform(np.log(323), np.log(480)))),
                      'n_layers': rng.integers(1, 2, endpoint=True),
                      'lr': np.exp(rng.uniform(np.log(3e-2), np.log(5e-1))),
                      'p_drop': rng.uniform(0.1, 0.5),
                      'wd': np.exp(rng.uniform(np.log(6e-3), np.log(6e-2))),
                      'plr_sigma': np.exp(rng.uniform(np.log(1e-2), np.log(15))),
                      'act': 'mish',
                      'wd_sched': 'flat_cos',
                      'ls_eps': rng.choice([0.0, np.exp(rng.uniform(np.log(5e-3), np.log(2e-2)))]),
                      'sq_mom': 1.0 - np.exp(rng.uniform(np.log(1e-3), np.log(4e-2))),
                      'plr_lr_factor': np.exp(rng.uniform(np.log(1e-1), np.log(3e-1))),
                      'scale_lr_factor': np.exp(rng.uniform(np.log(2.5), np.log(7.5))),
                      'p_drop_sched': 'constant',
                      }
            params['hidden_sizes'] = [params['n_hidden']] * params['n_layers']

        elif self.hpo_space_name == 'alt4':
            # large space for regression
            params = {'num_emb_type': 'pbld',
                      'add_front_scale': rng.choice([True, False], p=[0.6, 0.4]),
                      'n_hidden': round(np.exp(rng.uniform(np.log(128), np.log(512)))),
                      'n_layers': rng.integers(1, 4, endpoint=True),
                      'lr': np.exp(rng.uniform(np.log(1e-2), np.log(5e-1))),
                      'p_drop': rng.uniform(0.0, 0.5),
                      'wd': np.exp(rng.uniform(np.log(1e-3), np.log(1e-1))),
                      'plr_sigma': np.exp(rng.uniform(np.log(1e-2), np.log(1e2))),
                      'act': rng.choice(['mish', 'silu', 'elu']),
                      'use_parametric_act': True,
                      'p_drop_sched': rng.choice(['flat_cos', 'constant']),
                      'wd_sched': rng.choice(['flat_cos', 'constant']),
                      'lr_sched': rng.choice(['coslog4', 'cos']),
                      'sq_mom': 1.0 - np.exp(rng.uniform(np.log(1e-3), np.log(1e-1))),
                      'plr_lr_factor': np.exp(rng.uniform(np.log(3e-2), np.log(3e-1))),
                      'scale_lr_factor': np.exp(rng.uniform(np.log(2.0), np.log(10.0))),
                      }

            params['hidden_sizes'] = [params['n_hidden']] * params['n_layers']
        elif self.hpo_space_name == 'alt5':
            # refined space for regression
            params = {'num_emb_type': 'pbld',
                      'add_front_scale': rng.choice([True, False], p=[0.6, 0.4]),
                      'n_hidden': round(np.exp(rng.uniform(np.log(128), np.log(512)))),
                      'n_layers': 4,
                      'lr': np.exp(rng.uniform(np.log(3e-2), np.log(1e-1))),
                      'p_drop': rng.uniform(0.0, 0.45),
                      'wd': np.exp(rng.uniform(np.log(1e-3), np.log(1e-1))),
                      'plr_sigma': np.exp(rng.uniform(np.log(1e-2), np.log(1e2))),
                      'act': 'mish',
                      'use_parametric_act': True,
                      'p_drop_sched': 'flat_cos',
                      'wd_sched': 'flat_cos',
                      'lr_sched': 'coslog4',
                      'sq_mom': 1.0 - np.exp(rng.uniform(np.log(3e-3), np.log(1e-1))),
                      'plr_lr_factor': np.exp(rng.uniform(np.log(5e-2), np.log(3e-1))),
                      'scale_lr_factor': np.exp(rng.uniform(np.log(2.0), np.log(7.5))),
                      }
            params['hidden_sizes'] = [params['n_hidden']] * params['n_layers']

        elif self.hpo_space_name == 'alt6':
            # regression, manually adjusted from alt5
            params = {'num_emb_type': 'pbld',
                      'add_front_scale': True,
                      'n_hidden': 256,
                      'n_layers': rng.choice([2, 3, 4]),
                      'lr': np.exp(rng.uniform(np.log(4e-2), np.log(2e-1))),
                      'p_drop': rng.uniform(0.0, 0.5),
                      'wd': np.exp(rng.uniform(np.log(1e-3), np.log(5e-2))),
                      'plr_sigma': np.exp(rng.uniform(np.log(1e-2), np.log(1e2))),
                      'act': 'mish',
                      'use_parametric_act': True,
                      'p_drop_sched': 'flat_cos',
                      'wd_sched': 'flat_cos',
                      'lr_sched': 'coslog4',
                      'sq_mom': 1.0 - np.exp(rng.uniform(np.log(3e-3), np.log(1e-1))),
                      'plr_lr_factor': np.exp(rng.uniform(np.log(5e-2), np.log(3e-1))),
                      'scale_lr_factor': np.exp(rng.uniform(np.log(2.0), np.log(7.5))),
                      }
            params['hidden_sizes'] = [params['n_hidden']] * params['n_layers']

        elif self.hpo_space_name == 'alt7':
            # refined version of alt2 (classification)
            params = {'num_emb_type': 'pbld',
                      'n_hidden': 256,
                      'n_layers': rng.integers(1, 4, endpoint=True),
                      'lr': np.exp(rng.uniform(np.log(1e-2), np.log(5e-1))),
                      'p_drop': rng.uniform(0.0, 0.6),
                      'wd': np.exp(rng.uniform(np.log(1e-3), np.log(1e-1))),
                      'plr_sigma': np.exp(rng.uniform(np.log(1e-2), np.log(30))),
                      'act': 'mish',
                      'wd_sched': rng.choice(['flat_cos', 'constant']),
                      'ls_eps': rng.choice([0.0, np.exp(rng.uniform(np.log(5e-3), np.log(2e-1)))]),
                      'sq_mom': 1.0 - np.exp(rng.uniform(np.log(1e-3), np.log(1e-1))),
                      'plr_lr_factor': np.exp(rng.uniform(np.log(3e-2), np.log(3e-1))),
                      'scale_lr_factor': np.exp(rng.uniform(np.log(2.0), np.log(10.0))),
                      'p_drop_sched': 'constant',
                      }
            params['hidden_sizes'] = [params['n_hidden']] * params['n_layers']

        elif self.hpo_space_name == 'alt8':
            # version of alt2 (classification) with some new hyperparameters
            params = {'num_emb_type': 'pbld',
                      'hidden_sizes': 'rectangular',
                      'hidden_width': 256,
                      'ls_eps_sched': 'coslog4',
                      'tfms': [['one_hot', 'median_center', 'robust_scale', 'smooth_clip', 'embedding'],
                               ['one_hot', 'mean_center', 'l2_normalize', 'embedding']][rng.choice([0, 1])],
                      'batch_size': [256, 'auto'][rng.choice([0, 1])],
                      'n_hidden_layers': rng.integers(1, 4, endpoint=True),
                      'first_layer_lr_factor': np.exp(rng.uniform(np.log(0.3), np.log(5.0))),
                      'lr': np.exp(rng.uniform(np.log(1e-2), np.log(5e-1))),
                      'p_drop': rng.uniform(0.06, 0.6),
                      'wd': np.exp(rng.uniform(np.log(6e-3), np.log(1e-1))),
                      'plr_sigma': np.exp(rng.uniform(np.log(1e-2), np.log(15))),
                      'act': rng.choice(['mish', 'silu']),
                      'wd_sched': rng.choice(['flat_cos', 'constant']),
                      'ls_eps': rng.choice([0.0, np.exp(rng.uniform(np.log(5e-3), np.log(1e-1)))]),
                      'sq_mom': 1.0 - np.exp(rng.uniform(np.log(1e-3), np.log(1e-1))),
                      'plr_lr_factor': np.exp(rng.uniform(np.log(3e-2), np.log(3e-1))),
                      'scale_lr_factor': np.exp(rng.uniform(np.log(2.0), np.log(10.0))),
                      'p_drop_sched': 'constant',
                      }
        elif self.hpo_space_name == 'alt9':
            # version of alt8 (classification) with reduced search spaces, and increased with space
            # removed batch_size tuning, tfms tuning
            params = {'num_emb_type': 'pbld',
                      'hidden_sizes': 'rectangular',
                      'hidden_width': rng.choice([256, 384, 512]),  # added
                      'ls_eps_sched': 'coslog4',
                      'n_hidden_layers': rng.integers(1, 3, endpoint=True),  # reduced
                      'first_layer_lr_factor': np.exp(rng.uniform(np.log(0.3), np.log(1.5))),  # reduced
                      'lr': np.exp(rng.uniform(np.log(1e-2), np.log(5e-1))),  # todo: could reduce this
                      'p_drop': rng.uniform(0.0, 0.5),  # reduced
                      'wd': np.exp(rng.uniform(np.log(5e-3), np.log(5e-2))),  # reduced
                      'plr_sigma': np.exp(rng.uniform(np.log(1e-2), np.log(15))),
                      'act': rng.choice(['mish', 'silu']),
                      'ls_eps': np.exp(rng.uniform(np.log(5e-3), np.log(1e-1))),
                      'sq_mom': 1.0 - np.exp(rng.uniform(np.log(5e-3), np.log(5e-2))),  # reduced
                      'plr_lr_factor': np.exp(rng.uniform(np.log(5e-2), np.log(3e-1))),  # reduced
                      'scale_lr_factor': np.exp(rng.uniform(np.log(2.0), np.log(10.0))),
                      'p_drop_sched': 'constant',
                      }
        elif self.hpo_space_name == 'alt10':
            # version of alt9, similar to tabrepo
            params = {'num_emb_type': 'pbld',
                      'hidden_sizes': 'rectangular',
                      'hidden_width': rng.choice([256, 384, 512]),
                      'ls_eps_sched': 'coslog4',
                      'act': 'mish',
                      'n_hidden_layers': rng.integers(1, 4, endpoint=True),
                      'first_layer_lr_factor': np.exp(rng.uniform(np.log(0.3), np.log(1.5))),
                      'lr': np.exp(rng.uniform(np.log(2e-2), np.log(3e-1))),
                      'p_drop': rng.uniform(0.0, 0.5),
                      'wd': np.exp(rng.uniform(np.log(1e-3), np.log(5e-2))),
                      'plr_sigma': np.exp(rng.uniform(np.log(1e-2), np.log(50))),
                      'ls_eps': np.exp(rng.uniform(np.log(5e-3), np.log(2e-1))),
                      'use_ls': rng.choice([False, True]),
                      'sq_mom': 1.0 - np.exp(rng.uniform(np.log(5e-3), np.log(5e-2))),
                      'plr_lr_factor': np.exp(rng.uniform(np.log(5e-2), np.log(3e-1))),
                      'scale_lr_factor': np.exp(rng.uniform(np.log(2.0), np.log(10.0))),
                      'p_drop_sched': 'flat_cos',
                      }
        elif self.hpo_space_name == 'tabarena':
            # common search space
            params = {
                'n_hidden_layers': rng.integers(2, 4, endpoint=True),
                'hidden_sizes': 'rectangular',
                'hidden_width': rng.choice([256, 384, 512]),
                'p_drop': rng.uniform(0.0, 0.5),
                'act': 'mish',
                'plr_sigma': np.exp(rng.uniform(np.log(1e-2), np.log(50))),
                'sq_mom': 1.0 - np.exp(rng.uniform(np.log(5e-3), np.log(5e-2))),
                'plr_lr_factor': np.exp(rng.uniform(np.log(5e-2), np.log(3e-1))),
                'scale_lr_factor': np.exp(rng.uniform(np.log(2.0), np.log(10.0))),
                'first_layer_lr_factor': np.exp(rng.uniform(np.log(0.3), np.log(1.5))),
                'ls_eps_sched': 'coslog4',
                'ls_eps': np.exp(rng.uniform(np.log(5e-3), np.log(1e-1))),
                'p_drop_sched': 'flat_cos',
                'lr': np.exp(rng.uniform(np.log(2e-2), np.log(3e-1))),
                'wd': np.exp(rng.uniform(np.log(1e-3), np.log(5e-2))),
                'use_ls': rng.choice([False, True]),  # use label smoothing (will be ignored for regression)
            }

            if rng.uniform(0.0, 1.0) > 0.5:
                # large configs
                params['plr_hidden_1'] = rng.choice([8, 16, 32, 64]).item()
                params['plr_hidden_2'] = rng.choice([8, 16, 32, 64]).item()
                params['n_epochs'] = rng.choice([256, 512]).item()
                params['use_early_stopping'] = True

                # set in the defaults of RealMLP in TabArena
                params['early_stopping_multiplicative_patience'] = 3
                params['early_stopping_additive_patience'] = 40
            else:
                # default values, used here to always set the same set of parameters
                params['plr_hidden_1'] = 16
                params['plr_hidden_2'] = 4
                params['n_epochs'] = 256
                params['use_early_stopping'] = False
        elif self.hpo_space_name == 'tabarena-new':
            # common search space
            params = {
                'n_hidden_layers': rng.integers(2, 4, endpoint=True),
                'hidden_sizes': 'rectangular',
                'hidden_width': rng.choice([256, 384, 512]),
                'p_drop': rng.uniform(0.0, 0.5),
                'act': 'mish',
                'plr_sigma': np.exp(rng.uniform(np.log(1e-2), np.log(50))),
                'sq_mom': 1.0 - np.exp(rng.uniform(np.log(5e-3), np.log(5e-2))),
                'plr_lr_factor': np.exp(rng.uniform(np.log(5e-2), np.log(3e-1))),
                'scale_lr_factor': np.exp(rng.uniform(np.log(2.0), np.log(10.0))),
                'first_layer_lr_factor': np.exp(rng.uniform(np.log(0.3), np.log(1.5))),
                'ls_eps_sched': 'coslog4',
                'ls_eps': np.exp(rng.uniform(np.log(5e-3), np.log(1e-1))),
                'p_drop_sched': 'flat_cos',
                'lr': np.exp(rng.uniform(np.log(2e-2), np.log(3e-1))),
                'wd': np.exp(rng.uniform(np.log(1e-3), np.log(5e-2))),
                'use_ls': rng.choice([False, True]),  # use label smoothing (will be ignored for regression)

                # added in tabarena-new compared to tabarena
                'max_one_hot_cat_size': int(np.floor(np.exp(rng.uniform(np.log(4.0), np.log(33.0)))).item()),
                'embedding_size': int(rng.choice([4, 8, 16])),
                'n_ens': 8,
                "ens_av_before_softmax": False,
            }

            if rng.uniform(0.0, 1.0) > 0.5:
                # large configs
                params['plr_hidden_1'] = rng.choice([8, 16, 32, 64]).item()
                params['plr_hidden_2'] = rng.choice([8, 16, 32, 64]).item()
                params['n_epochs'] = rng.choice([256, 512]).item()
                params['use_early_stopping'] = True

                # set in the defaults of RealMLP in TabArena
                params['early_stopping_multiplicative_patience'] = 3
                params['early_stopping_additive_patience'] = 40
            else:
                # default values, used here to always set the same set of parameters
                params['plr_hidden_1'] = 16
                params['plr_hidden_2'] = 4
                params['n_epochs'] = 256
                params['use_early_stopping'] = False
        elif self.hpo_space_name == 'alt11':
            # tabarena without the large configs
            params = {
                'n_hidden_layers': rng.integers(2, 4, endpoint=True),
                'hidden_sizes': 'rectangular',
                'hidden_width': rng.choice([256, 384, 512]),
                'p_drop': rng.uniform(0.0, 0.5),
                'act': 'mish',
                'plr_sigma': np.exp(rng.uniform(np.log(1e-2), np.log(50))),
                'sq_mom': 1.0 - np.exp(rng.uniform(np.log(5e-3), np.log(5e-2))),
                'plr_lr_factor': np.exp(rng.uniform(np.log(5e-2), np.log(3e-1))),
                'scale_lr_factor': np.exp(rng.uniform(np.log(2.0), np.log(10.0))),
                'first_layer_lr_factor': np.exp(rng.uniform(np.log(0.3), np.log(1.5))),
                'ls_eps_sched': 'coslog4',
                'ls_eps': np.exp(rng.uniform(np.log(5e-3), np.log(1e-1))),
                'p_drop_sched': 'flat_cos',
                'lr': np.exp(rng.uniform(np.log(2e-2), np.log(3e-1))),
                'wd': np.exp(rng.uniform(np.log(1e-3), np.log(5e-2))),
                'use_ls': rng.choice([False, True]),  # use label smoothing (will be ignored for regression)
            }
        elif self.hpo_space_name == 'alt12':
            # alt11 with n_hidden_layers=1 in the search space
            params = {
                'n_hidden_layers': rng.integers(1, 4, endpoint=True),
                'hidden_sizes': 'rectangular',
                'hidden_width': rng.choice([256, 384, 512]),
                'p_drop': rng.uniform(0.0, 0.5),
                'act': 'mish',
                'plr_sigma': np.exp(rng.uniform(np.log(1e-2), np.log(50))),
                'sq_mom': 1.0 - np.exp(rng.uniform(np.log(5e-3), np.log(5e-2))),
                'plr_lr_factor': np.exp(rng.uniform(np.log(5e-2), np.log(3e-1))),
                'scale_lr_factor': np.exp(rng.uniform(np.log(2.0), np.log(10.0))),
                'first_layer_lr_factor': np.exp(rng.uniform(np.log(0.3), np.log(1.5))),
                'ls_eps_sched': 'coslog4',
                'ls_eps': np.exp(rng.uniform(np.log(5e-3), np.log(1e-1))),
                'p_drop_sched': 'flat_cos',
                'lr': np.exp(rng.uniform(np.log(2e-2), np.log(3e-1))),
                'wd': np.exp(rng.uniform(np.log(1e-3), np.log(5e-2))),
                'use_ls': rng.choice([False, True]),  # use label smoothing (will be ignored for regression)
            }
        elif self.hpo_space_name == 'alt13':
            # alt11 with more categorical hyperparameters
            params = {
                'n_hidden_layers': rng.integers(2, 4, endpoint=True),
                'hidden_sizes': 'rectangular',
                'hidden_width': rng.choice([256, 384, 512]),
                'p_drop': rng.uniform(0.0, 0.5),
                'act': 'mish',
                'plr_sigma': np.exp(rng.uniform(np.log(1e-2), np.log(50))),
                'sq_mom': 1.0 - np.exp(rng.uniform(np.log(5e-3), np.log(5e-2))),
                'plr_lr_factor': np.exp(rng.uniform(np.log(5e-2), np.log(3e-1))),
                'scale_lr_factor': np.exp(rng.uniform(np.log(2.0), np.log(10.0))),
                'first_layer_lr_factor': np.exp(rng.uniform(np.log(0.3), np.log(1.5))),
                'ls_eps_sched': 'coslog4',
                'ls_eps': np.exp(rng.uniform(np.log(5e-3), np.log(1e-1))),
                'p_drop_sched': 'flat_cos',
                'lr': np.exp(rng.uniform(np.log(2e-2), np.log(3e-1))),
                'wd': np.exp(rng.uniform(np.log(1e-3), np.log(5e-2))),
                'use_ls': rng.choice([False, True]),  # use label smoothing (will be ignored for regression)
                'max_one_hot_cat_size': int(np.floor(np.exp(rng.uniform(np.log(4.0), np.log(33.0)))).item()),
                'embedding_size': int(rng.choice([4, 8, 16])),
            }
        elif self.hpo_space_name == 'alt14':
            # alt13 with weight_init_mode='normal'
            params = {
                'n_hidden_layers': rng.integers(2, 4, endpoint=True),
                'hidden_sizes': 'rectangular',
                'hidden_width': rng.choice([256, 384, 512]),
                'p_drop': rng.uniform(0.0, 0.5),
                'act': 'mish',
                'plr_sigma': np.exp(rng.uniform(np.log(1e-2), np.log(50))),
                'sq_mom': 1.0 - np.exp(rng.uniform(np.log(5e-3), np.log(5e-2))),
                'plr_lr_factor': np.exp(rng.uniform(np.log(5e-2), np.log(3e-1))),
                'scale_lr_factor': np.exp(rng.uniform(np.log(2.0), np.log(10.0))),
                'first_layer_lr_factor': np.exp(rng.uniform(np.log(0.3), np.log(1.5))),
                'ls_eps_sched': 'coslog4',
                'ls_eps': np.exp(rng.uniform(np.log(5e-3), np.log(1e-1))),
                'p_drop_sched': 'flat_cos',
                'lr': np.exp(rng.uniform(np.log(2e-2), np.log(3e-1))),
                'wd': np.exp(rng.uniform(np.log(1e-3), np.log(5e-2))),
                'use_ls': rng.choice([False, True]),  # use label smoothing (will be ignored for regression)
                'max_one_hot_cat_size': int(np.floor(np.exp(rng.uniform(np.log(4.0), np.log(33.0)))).item()),
                'embedding_size': int(rng.choice([4, 8, 16])),
                'weight_init_mode': 'normal',
            }
        elif self.hpo_space_name == 'alt15':
            # alt13 with tuning momentum (beta1)
            params = {
                'n_hidden_layers': rng.integers(2, 4, endpoint=True),
                'hidden_sizes': 'rectangular',
                'hidden_width': rng.choice([256, 384, 512]),
                'p_drop': rng.uniform(0.0, 0.5),
                'act': 'mish',
                'plr_sigma': np.exp(rng.uniform(np.log(1e-2), np.log(50))),
                'sq_mom': 1.0 - np.exp(rng.uniform(np.log(5e-3), np.log(5e-2))),
                'plr_lr_factor': np.exp(rng.uniform(np.log(5e-2), np.log(3e-1))),
                'scale_lr_factor': np.exp(rng.uniform(np.log(2.0), np.log(10.0))),
                'first_layer_lr_factor': np.exp(rng.uniform(np.log(0.3), np.log(1.5))),
                'ls_eps_sched': 'coslog4',
                'ls_eps': np.exp(rng.uniform(np.log(5e-3), np.log(1e-1))),
                'p_drop_sched': 'flat_cos',
                'lr': np.exp(rng.uniform(np.log(2e-2), np.log(3e-1))),
                'wd': np.exp(rng.uniform(np.log(1e-3), np.log(5e-2))),
                'use_ls': rng.choice([False, True]),  # use label smoothing (will be ignored for regression)
                'max_one_hot_cat_size': int(np.floor(np.exp(rng.uniform(np.log(4.0), np.log(33.0)))).item()),
                'embedding_size': int(rng.choice([4, 8, 16])),
                'mom': 1.0 - np.exp(rng.uniform(np.log(2e-2), np.log(3e-1))), # tune in [0.7, 0.98]
            }
        elif self.hpo_space_name == 'alt16':
            # alt13 with n_ens=2
            params = {
                'n_hidden_layers': rng.integers(2, 4, endpoint=True),
                'hidden_sizes': 'rectangular',
                'hidden_width': rng.choice([256, 384, 512]),
                'p_drop': rng.uniform(0.0, 0.5),
                'act': 'mish',
                'plr_sigma': np.exp(rng.uniform(np.log(1e-2), np.log(50))),
                'sq_mom': 1.0 - np.exp(rng.uniform(np.log(5e-3), np.log(5e-2))),
                'plr_lr_factor': np.exp(rng.uniform(np.log(5e-2), np.log(3e-1))),
                'scale_lr_factor': np.exp(rng.uniform(np.log(2.0), np.log(10.0))),
                'first_layer_lr_factor': np.exp(rng.uniform(np.log(0.3), np.log(1.5))),
                'ls_eps_sched': 'coslog4',
                'ls_eps': np.exp(rng.uniform(np.log(5e-3), np.log(1e-1))),
                'p_drop_sched': 'flat_cos',
                'lr': np.exp(rng.uniform(np.log(2e-2), np.log(3e-1))),
                'wd': np.exp(rng.uniform(np.log(1e-3), np.log(5e-2))),
                'use_ls': rng.choice([False, True]),  # use label smoothing (will be ignored for regression)
                'max_one_hot_cat_size': int(np.floor(np.exp(rng.uniform(np.log(4.0), np.log(33.0)))).item()),
                'embedding_size': int(rng.choice([4, 8, 16])),
                'n_ens': 2,
                'ens_av_before_softmax': True,
            }
        elif self.hpo_space_name == 'alt17':
            # alt13 with n_ens=4
            params = {
                'n_hidden_layers': rng.integers(2, 4, endpoint=True),
                'hidden_sizes': 'rectangular',
                'hidden_width': rng.choice([256, 384, 512]),
                'p_drop': rng.uniform(0.0, 0.5),
                'act': 'mish',
                'plr_sigma': np.exp(rng.uniform(np.log(1e-2), np.log(50))),
                'sq_mom': 1.0 - np.exp(rng.uniform(np.log(5e-3), np.log(5e-2))),
                'plr_lr_factor': np.exp(rng.uniform(np.log(5e-2), np.log(3e-1))),
                'scale_lr_factor': np.exp(rng.uniform(np.log(2.0), np.log(10.0))),
                'first_layer_lr_factor': np.exp(rng.uniform(np.log(0.3), np.log(1.5))),
                'ls_eps_sched': 'coslog4',
                'ls_eps': np.exp(rng.uniform(np.log(5e-3), np.log(1e-1))),
                'p_drop_sched': 'flat_cos',
                'lr': np.exp(rng.uniform(np.log(2e-2), np.log(3e-1))),
                'wd': np.exp(rng.uniform(np.log(1e-3), np.log(5e-2))),
                'use_ls': rng.choice([False, True]),  # use label smoothing (will be ignored for regression)
                'max_one_hot_cat_size': int(np.floor(np.exp(rng.uniform(np.log(4.0), np.log(33.0)))).item()),
                'embedding_size': int(rng.choice([4, 8, 16])),
                'n_ens': 4,
                'ens_av_before_softmax': True,
            }
        elif self.hpo_space_name == 'alt18':
            # alt17 but with averaging after softmax
            params = {
                'n_hidden_layers': rng.integers(2, 4, endpoint=True),
                'hidden_sizes': 'rectangular',
                'hidden_width': rng.choice([256, 384, 512]),
                'p_drop': rng.uniform(0.0, 0.5),
                'act': 'mish',
                'plr_sigma': np.exp(rng.uniform(np.log(1e-2), np.log(50))),
                'sq_mom': 1.0 - np.exp(rng.uniform(np.log(5e-3), np.log(5e-2))),
                'plr_lr_factor': np.exp(rng.uniform(np.log(5e-2), np.log(3e-1))),
                'scale_lr_factor': np.exp(rng.uniform(np.log(2.0), np.log(10.0))),
                'first_layer_lr_factor': np.exp(rng.uniform(np.log(0.3), np.log(1.5))),
                'ls_eps_sched': 'coslog4',
                'ls_eps': np.exp(rng.uniform(np.log(5e-3), np.log(1e-1))),
                'p_drop_sched': 'flat_cos',
                'lr': np.exp(rng.uniform(np.log(2e-2), np.log(3e-1))),
                'wd': np.exp(rng.uniform(np.log(1e-3), np.log(5e-2))),
                'use_ls': rng.choice([False, True]),  # use label smoothing (will be ignored for regression)
                'max_one_hot_cat_size': int(np.floor(np.exp(rng.uniform(np.log(4.0), np.log(33.0)))).item()),
                'embedding_size': int(rng.choice([4, 8, 16])),
                'n_ens': 4,
                'ens_av_before_softmax': False,
            }
        elif self.hpo_space_name == 'alt19':
            # alt13 with numerical preprocessing tuning
            tfms_list = [
                ['one_hot', 'median_center', 'robust_scale', 'smooth_clip', 'embedding'],
                ['one_hot', 'mean_center', 'l2_normalize', 'smooth_clip', 'embedding'],
            ]
            params = {
                'n_hidden_layers': rng.integers(2, 4, endpoint=True),
                'hidden_sizes': 'rectangular',
                'hidden_width': rng.choice([256, 384, 512]),
                'p_drop': rng.uniform(0.0, 0.5),
                'act': 'mish',
                'plr_sigma': np.exp(rng.uniform(np.log(1e-2), np.log(50))),
                'sq_mom': 1.0 - np.exp(rng.uniform(np.log(5e-3), np.log(5e-2))),
                'plr_lr_factor': np.exp(rng.uniform(np.log(5e-2), np.log(3e-1))),
                'scale_lr_factor': np.exp(rng.uniform(np.log(2.0), np.log(10.0))),
                'first_layer_lr_factor': np.exp(rng.uniform(np.log(0.3), np.log(1.5))),
                'ls_eps_sched': 'coslog4',
                'ls_eps': np.exp(rng.uniform(np.log(5e-3), np.log(1e-1))),
                'p_drop_sched': 'flat_cos',
                'lr': np.exp(rng.uniform(np.log(2e-2), np.log(3e-1))),
                'wd': np.exp(rng.uniform(np.log(1e-3), np.log(5e-2))),
                'use_ls': rng.choice([False, True]),  # use label smoothing (will be ignored for regression)
                'max_one_hot_cat_size': int(np.floor(np.exp(rng.uniform(np.log(4.0), np.log(33.0)))).item()),
                'embedding_size': int(rng.choice([4, 8, 16])),
                'tfms': tfms_list[int(rng.choice([0, 1]))],
                'smooth_clip_max_abs_value': np.exp(rng.uniform(np.log(1.0), np.log(10.0)))
            }
        elif self.hpo_space_name == 'alt20':
            # alt13 with numerical preprocessing tuning (but without the max_abs_value unlike alt19)
            tfms_list = [
                ['one_hot', 'median_center', 'robust_scale', 'smooth_clip', 'embedding'],
                ['one_hot', 'mean_center', 'l2_normalize', 'smooth_clip', 'embedding'],
            ]
            params = {
                'n_hidden_layers': rng.integers(2, 4, endpoint=True),
                'hidden_sizes': 'rectangular',
                'hidden_width': rng.choice([256, 384, 512]),
                'p_drop': rng.uniform(0.0, 0.5),
                'act': 'mish',
                'plr_sigma': np.exp(rng.uniform(np.log(1e-2), np.log(50))),
                'sq_mom': 1.0 - np.exp(rng.uniform(np.log(5e-3), np.log(5e-2))),
                'plr_lr_factor': np.exp(rng.uniform(np.log(5e-2), np.log(3e-1))),
                'scale_lr_factor': np.exp(rng.uniform(np.log(2.0), np.log(10.0))),
                'first_layer_lr_factor': np.exp(rng.uniform(np.log(0.3), np.log(1.5))),
                'ls_eps_sched': 'coslog4',
                'ls_eps': np.exp(rng.uniform(np.log(5e-3), np.log(1e-1))),
                'p_drop_sched': 'flat_cos',
                'lr': np.exp(rng.uniform(np.log(2e-2), np.log(3e-1))),
                'wd': np.exp(rng.uniform(np.log(1e-3), np.log(5e-2))),
                'use_ls': rng.choice([False, True]),  # use label smoothing (will be ignored for regression)
                'max_one_hot_cat_size': int(np.floor(np.exp(rng.uniform(np.log(4.0), np.log(33.0)))).item()),
                'embedding_size': int(rng.choice([4, 8, 16])),
                'tfms': tfms_list[int(rng.choice([0, 1]))],
            }

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
        params = utils.join_dicts(self.config, self.fit_params[0], self.config.get('override_params', dict()) or dict())
        # params = utils.update_dict(self.fit_params[0], self.config)
        if 'n_epochs' in self.config:
            params['n_epochs'] = self.config['n_epochs']
        self.fit_params[0] = params
        return NNAlgInterface(fit_params=None, **params)

    def fit(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources,
            logger: Logger, tmp_folders: List[Optional[Path]], name: str) -> None:
        assert len(idxs_list) == 1
        self.alg_interface = self._create_sub_interface(ds, idxs_list[0].split_seed)
        logger.log(1, f'{self.fit_params=}')
        self.alg_interface.fit(ds, idxs_list, interface_resources, logger, tmp_folders, name)
        self.fit_params[0]['sub_fit_params'] = self.alg_interface.fit_params[0]

    def predict(self, ds: DictDataset) -> torch.Tensor:
        self.alg_interface.set_current_predict_params(self.get_current_predict_params_name())
        return self.alg_interface.predict(ds)

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        assert len(split_seeds) == 1
        alg_interface = self._create_sub_interface(ds, split_seeds[0])
        return alg_interface.get_required_resources(ds, n_cv, n_refit, n_splits, split_seeds, n_train=n_train)

    def get_available_predict_params(self) -> Dict[str, Dict[str, Any]]:
        return NNAlgInterface(**self.config).get_available_predict_params()

    def to(self, device: str) -> None:
        self.alg_interface.to(device)

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
