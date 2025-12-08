import contextlib
import random
from pathlib import Path
from typing import Optional, List, Any, Tuple, Dict

import numpy as np
import torch

from pytabkit.models import utils
from pytabkit.models.alg_interfaces.alg_interfaces import SingleSplitAlgInterface, AlgInterface, \
    RandomParamsAlgInterface
from pytabkit.models.alg_interfaces.base import RequiredResources, SplitIdxs, InterfaceResources
from pytabkit.models.alg_interfaces.resource_computation import ResourcePredictor
from pytabkit.models.alg_interfaces.sub_split_interfaces import SingleSplitWrapperAlgInterface
from pytabkit.models.data.data import DictDataset
from pytabkit.models.nn_models.base import Fitter
from pytabkit.models.nn_models.models import PreprocessingFactory
from pytabkit.models.torch_utils import get_available_memory_gb
from pytabkit.models.training.logging import Logger



class xRFMSubSplitInterface(SingleSplitAlgInterface):
    def __init__(self, fit_params: Optional[List[Dict[str, Any]]] = None, **config):
        super().__init__(fit_params=fit_params, **config)

    def get_refit_interface(self, n_refit: int, fit_params: Optional[List[Dict]] = None) -> 'AlgInterface':
        raise NotImplementedError()

    def fit(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources,
            logger: Logger, tmp_folders: List[Optional[Path]], name: str) -> Optional[
        List[List[List[Tuple[Dict, float]]]]]:
        assert len(idxs_list) == 1
        assert idxs_list[0].n_trainval_splits == 1

        torch.set_float32_matmul_precision('highest')

        seed = idxs_list[0].sub_split_seeds[0]
        # print(f'Setting seed: {seed}')
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        n_train = idxs_list[0].n_train
        n_classes = ds.get_n_classes()
        device = interface_resources.gpu_devices[0] if len(interface_resources.gpu_devices) >= 1 else 'cpu'

        self.n_classes_ = n_classes
        self.device_ = device

        # create preprocessing factory
        factory = self.config.get('factory', None)
        if 'tfms' not in self.config:
            self.config['tfms'] = ['mean_center', 'l2_normalize', 'one_hot']
        if factory is None:
            print("factory is None, creating factory")
            factory = PreprocessingFactory(**self.config)

        if idxs_list[0].val_idxs is None:
            raise ValueError(f'Training without validation set is currently not implemented')

        ds_train = ds.get_sub_dataset(idxs_list[0].train_idxs[0])
        ds_val = ds.get_sub_dataset(idxs_list[0].val_idxs[0])

        num_numerical = ds_train.tensor_infos['x_cont'].get_n_features()
        raw_cat_sizes = ds_train.tensor_infos['x_cat'].get_cat_sizes()
        if isinstance(raw_cat_sizes, torch.Tensor):
            raw_cat_sizes = raw_cat_sizes.tolist()
        else:
            raw_cat_sizes = [int(size) for size in raw_cat_sizes]

        if 'factory' in self.config or 'one_hot' not in self.config['tfms']:
            cat_sizes = []  # don't apply fast_categorical stuff
        else:
            use_missing_zero = self.config.get('use_missing_zero', True)
            use_binary_drop = self.config.get('use_1d_binary_onehot', True)
            cat_sizes = []
            for size in raw_cat_sizes:
                adjusted = size - 1 if use_missing_zero else size
                if adjusted == 2 and use_binary_drop:
                    adjusted = 1
                cat_sizes.append(adjusted)

        # transform according to factory
        fitter: Fitter = factory.create(ds.tensor_infos)
        self.tfm_, ds_train = fitter.fit_transform(ds_train)
        ds_val = self.tfm_(ds_val)

        # print("Expected shape from ds_train: ", ds_train.tensors['x_cont'].shape)
        numerical_indices, categorical_indices, categorical_vectors = None, None, None
        if 'one_hot' in self.config['tfms']:
            # Simpler categorical_info construction (no standardization):
            # - Treat one-hots for categories with <=100 levels as numerical features
            # - Only provide identity vectors for categories with >100 levels
            numerical_block = torch.arange(num_numerical)
            categorical_indices = []
            categorical_vectors = []
            numerical_indices_parts = []
            idx = num_numerical
            for cat_size in cat_sizes:
                cat_idxs = torch.arange(idx, idx + cat_size)
                if cat_size > 100:
                    categorical_indices.append(cat_idxs)
                    categorical_vectors.append(torch.eye(cat_size))
                else:
                    numerical_indices_parts.append(cat_idxs)
                idx += cat_size
            if len(numerical_indices_parts) > 0:
                numerical_indices = torch.cat([numerical_block] + numerical_indices_parts)
            else:
                numerical_indices = numerical_block
        
        # assume categoricals are encoded
        x_train = ds_train.tensors['x_cont'].to(device)
        x_val = ds_val.tensors['x_cont'].to(device)
        y_train = ds_train.tensors['y'].to(device)
        y_val = ds_val.tensors['y'].to(device)

        if self.n_classes_ == 0:  # regression
            assert ds.tensor_infos['y'].get_n_features() == 1
            self.y_mean_ = y_train.mean().item()
            self.y_std_ = y_train.std(correction=0).item()

            y_train = (y_train - self.y_mean_) / (self.y_std_ + 1e-30)
            y_val = (y_val - self.y_mean_) / (self.y_std_ + 1e-30)
        else:
            y_train = y_train.long()
            y_val = y_val.long()

        bandwidth = self.config.get('bandwidth', 10)
        p_interp = self.config.get('p_interp', 0.0)
        exponent = self.config.get('exponent', 1.0)
        reg = self.config.get('reg', 1e-3)
        iters = self.config.get('rfm_iters', 5)
        diag = self.config.get('diag', True)
        min_subset_size = self.config.get('max_leaf_samples', self.config.get('min_subset_size', 60_000))
        early_stop_rfm = self.config.get('early_stop_rfm', True)
        early_stop_multiplier = self.config.get('early_stop_multiplier', 1.1)
        classification_mode = self.config.get('classification_mode', 'prevalence')
        fast_categorical = self.config.get('fast_categorical', True)
        M_batch_size = self.config.get('M_batch_size', 'auto')
        overlap_fraction = self.config.get('overlap_fraction', 0.1)
        use_temperature_tuning = self.config.get('use_temperature_tuning', True)
        temp_tuning_space = self.config.get('temp_tuning_space', None)

        bandwidth_mode = self.config.get('bandwidth_mode', 'constant')
        kernel_type = self.config.get('kernel_type', 'l2')
        split_method = self.config.get('split_method', 'top_vector_agop_on_subset')
        if bandwidth_mode in ['constant', 'adaptive']:
            pass
        elif bandwidth_mode == 'sqrtd':
            bandwidth *= np.sqrt(x_train.shape[0])
        else:
            raise ValueError()

        if M_batch_size == 'auto':
            if kernel_type in ['gen_laplace', 'l1-laplace', 'lpq-laplace', 'l1', 'lpq', 'lpq_kermac']:
                # heuristic for storing a (n_train, M_batch_size, n_features) tensor in memory
                # 4 bytes per float
                full_tensor_size_per_elem_gb = (4 * n_train * ds_train.tensor_infos['x_cont'].get_n_features()) / (
                        1024 ** 3)
                full_tensor_size_per_elem_gb *= 12  # just a heuristic
                M_batch_size = max(1, min(8192, round(get_available_memory_gb(device) / full_tensor_size_per_elem_gb)))
                # M_batch_size = 512 if n_train <= 10_000 else (256 if n_train <= 20_000 else 64)
            else:
                M_batch_size = 8192

        print(f'{kernel_type=}, {M_batch_size=}')

        model_params, fit_params = {}, {}
        model_params['kernel'] = kernel_type
        model_params['bandwidth'] = bandwidth
        model_params['exponent'] = exponent
        model_params['norm_p'] = exponent + (2-exponent)*p_interp
        model_params['bandwidth_mode'] = bandwidth_mode
        model_params['diag'] = diag
        model_params['fast_categorical'] = fast_categorical
        fit_params['reg'] = reg
        fit_params['iters'] = iters
        fit_params['verbose'] = True
        fit_params['early_stop_rfm'] = early_stop_rfm
        fit_params['early_stop_multiplier'] = early_stop_multiplier
        fit_params['M_batch_size'] = M_batch_size

        if self.n_classes_ == 2:
            fit_params['solver'] = self.config.get('binary_solver', 'solve')
        else:
            fit_params['solver'] = 'solve'

        rfm_params = {'model': model_params, 'fit': fit_params}

        if 'one_hot' in self.config['tfms']:
            # Provide identity vectors only for high-cardinality categoricals; treat others as numerical
            categorical_info = {
                'numerical_indices': numerical_indices.to(device),
                'categorical_indices': [i.to(device) for i in categorical_indices],
                'categorical_vectors': [v.to(device) for v in categorical_vectors],
            }
        else:
            # treat cats like numerical features
            categorical_info = None
        
        classification = self.n_classes_ > 0

        val_metric_name = self.config.get('val_metric_name', 'class_error' if classification else 'mse')
        metric_name_to_metric_class = {
            '1-auroc-ovr': 'auc',
            'class_error': 'accuracy',
            'mse': 'mse',
            'rmse': 'rmse',
            'logloss': 'logloss',
            'cross_entropy': 'logloss',
            'brier': 'mse',
        }
        tuning_metric = metric_name_to_metric_class[val_metric_name]

        from xrfm import xRFM
        self.model_ = xRFM(rfm_params, device=device, min_subset_size=min_subset_size, 
                             tuning_metric=tuning_metric,
                             categorical_info=categorical_info, 
                             classification_mode=classification_mode,
                             split_method=split_method, overlap_fraction=overlap_fraction,
                           use_temperature_tuning=use_temperature_tuning, temp_tuning_space=temp_tuning_space)

        self.model_.fit(x_train, y_train, x_val, y_val)

        return None

    def predict(self, ds: DictDataset) -> torch.Tensor:
        ds = self.tfm_(ds).to(self.device_)

        x_cont = ds.tensors['x_cont']

        if self.n_classes_ > 0:
            with torch.cuda.device(self.device_) if self.device_.startswith('cuda') else contextlib.nullcontext():
                y_pred = torch.from_numpy(self.model_.predict_proba(x_cont)).to(self.device_)
            y_pred = torch.log(y_pred)
        else:
            with torch.cuda.device(self.device_) if self.device_.startswith('cuda') else contextlib.nullcontext():
                y_pred = torch.from_numpy(self.model_.predict(x_cont)).to(self.device_)
            y_pred = y_pred * self.y_std_ + self.y_mean_

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
        # gpu_ram_params = {'': 0.2, 'ds_onehot_size_gb': 5.0, 'n_train': 6e-5, 'n_features': 2e-3,
        #                   'n_train*n_train': 20.0 / (1024 ** 3), 'n_train*n_features': 20.0 / (1024 ** 3)}
        gpu_ram_params = {'': 0.0, 'ds_onehot_size_gb': 0.0, 'n_train': 0.0, 'n_features': 0.0,
                          'n_train*n_train': 0.0, 'n_train*n_features': 0.0}
        rc = ResourcePredictor(config=updated_config, time_params=time_params, gpu_ram_params=gpu_ram_params,
                               cpu_ram_params=ram_params, n_gpus=1, gpu_usage=1)
        
        # print("rc.get_required_resources(ds, n_train=n_train)")
        # rr = rc.get_required_resources(ds, n_train=n_train)
        # print("rr.n_threads = ", rr.n_threads)
        # print("rr.cpu_ram_gb = ", rr.cpu_ram_gb)
        # print("rr.n_gpus = ", rr.n_gpus)
        # print("rr.gpu_usage = ", rr.gpu_usage)
        # print("rr.gpu_ram_gb = ", rr.gpu_ram_gb)
        # print("rr.time_s = ", rr.time_s)
        # exit()
        return rc.get_required_resources(ds, n_train=n_train)
        # return RequiredResources(time_s=10.0, n_threads=16, cpu_ram_gb=50, n_gpus=0)


def sample_xrfm_params(seed: int, hpo_space_name: str = 'default'):
    rng = np.random.default_rng(seed)

    if hpo_space_name == 'default':
        # similar or identical to the search space used in TabArena
        # (but here we also tune the categorical preprocessing)
        num_tfms_list = [['mean_center', 'l2_normalize']]
        num_tfms = num_tfms_list[rng.integers(len(num_tfms_list))]
        cat_tfms_list = [['ordinal_encoding'], ['one_hot']]
        cat_tfms = cat_tfms_list[rng.integers(len(cat_tfms_list))]
        params = {
            'bandwidth': np.exp(rng.uniform(np.log(0.5), np.log(200.0))),
            'reg': np.exp(rng.uniform(np.log(1e-6), np.log(10.))),
            'exponent': rng.uniform(0.7, 1.4),
            'p_interp': rng.uniform(0., 0.8),
            'tfms': num_tfms + cat_tfms,
            'diag': rng.choice([False, True]),
            'kernel_type': rng.choice(['lpq', 'l2'], p=[0.8, 0.2]),
            # don't set these here so they can be overridden
            # they're the default values anyway
            # 'bandwidth_mode': rng.choice(['constant']),
            # 'min_subset_size': 60_000,
            # 'rfm_iters': 5,
            # 'classification_mode': 'prevalence',
            # 'binary_solver': 'solve',
            # 'early_stop_rfm': True,
            # 'early_stop_multiplier': 1.1, # early stop if val metric > esm * best val metric (for loss)
            # 'split_method': 'top_vector_agop_on_subset',
        }
    elif hpo_space_name == 'only_l2':
        num_tfms_list = [['mean_center', 'l2_normalize']]
        num_tfms = num_tfms_list[rng.integers(len(num_tfms_list))]
        cat_tfms_list = [['ordinal_encoding'], ['one_hot']]
        cat_tfms = cat_tfms_list[rng.integers(len(cat_tfms_list))]
        params = {
            'bandwidth': np.exp(rng.uniform(np.log(0.5), np.log(200.0))),
            'reg': np.exp(rng.uniform(np.log(1e-6), np.log(10.))),
            'exponent': rng.uniform(0.7, 1.4),
            'tfms': num_tfms + cat_tfms,
            'diag': rng.choice([False, True]),
            # don't set these here so they can be overridden
            # 'bandwidth_mode': rng.choice(['constant']),
            # 'kernel_type': 'l2',
            # 'min_subset_size': 60_000,
            # 'rfm_iters': 5,
            # 'classification_mode': 'prevalence',
            # 'binary_solver': 'solve',
            # 'early_stop_rfm': True,
            # 'early_stop_multiplier': 1.1, # early stop if val metric > esm * best val metric (for loss)
            # 'split_method': 'top_vector_agop_on_subset',
        }
    elif hpo_space_name == 'paper-large':
        # used on meta-test in the paper
        num_tfms_list = [['mean_center', 'l2_normalize']]
        num_tfms = num_tfms_list[rng.integers(len(num_tfms_list))]
        cat_tfms_list = [['ordinal_encoding'], ['one_hot']]
        cat_tfms = cat_tfms_list[rng.integers(len(cat_tfms_list))]

        params = {
            'bandwidth_mode': rng.choice(['constant', 'adaptive']),
            'bandwidth': np.exp(rng.uniform(np.log(0.4), np.log(80.0))),
            'reg': np.exp(rng.uniform(np.log(1e-5), np.log(50.))),
            'exponent': rng.uniform(0.7, 1.3),
            'p_interp': rng.uniform(0., 0.8),
            'tfms': num_tfms + cat_tfms,
            'diag': rng.choice([False, True]),
            'kernel_type': rng.choice(['lpq_kermac', 'l2'], p=[0.8, 0.2]),
            # 'max_leaf_samples': 60_000,  # don't put it here, it's the default anyway and can be overridden
            'rfm_iters': 5,
            'classification_mode': 'zero_one',
            'binary_solver': 'solve',  # todo: adjust general solver?
            'early_stop_rfm': True,
            'early_stop_multiplier': 1.1,  # early stop if val metric > esm * best val metric (for loss)
            'split_method': 'top_vector_agop_on_subset',
            'overlap_fraction': 0.0,
            'use_temperature_tuning': False,
        }
    elif hpo_space_name == 'paper-large-pca':
        # like paper-large, but with pca splitting
        num_tfms_list = [['mean_center', 'l2_normalize']]
        num_tfms = num_tfms_list[rng.integers(len(num_tfms_list))]
        cat_tfms_list = [['ordinal_encoding'], ['one_hot']]
        cat_tfms = cat_tfms_list[rng.integers(len(cat_tfms_list))]

        params = {
            'bandwidth_mode': rng.choice(['constant', 'adaptive']),
            'bandwidth': np.exp(rng.uniform(np.log(0.4), np.log(80.0))),
            'reg': np.exp(rng.uniform(np.log(1e-5), np.log(50.))),
            'exponent': rng.uniform(0.7, 1.3),
            'p_interp': rng.uniform(0., 0.8),
            'tfms': num_tfms + cat_tfms,
            'diag': rng.choice([False, True]),
            'kernel_type': rng.choice(['lpq_kermac', 'l2'], p=[0.8, 0.2]),
            # 'max_leaf_samples': 60_000,
            'rfm_iters': 5,  # don't put it here, it's the default anyway and can be overridden
            'classification_mode': 'zero_one',
            'binary_solver': 'solve',  # todo: adjust general solver?
            'early_stop_rfm': True,
            'early_stop_multiplier': 1.1,  # early stop if val metric > esm * best val metric (for loss)
            'split_method': 'pca',  # changed compared
            'overlap_fraction': 0.0,
            'use_temperature_tuning': False,
        }
    elif hpo_space_name == 'large-soft':
        # used on meta-test in the paper
        num_tfms_list = [['mean_center', 'l2_normalize']]
        num_tfms = num_tfms_list[rng.integers(len(num_tfms_list))]
        cat_tfms_list = [['ordinal_encoding'], ['one_hot']]
        cat_tfms = cat_tfms_list[rng.integers(len(cat_tfms_list))]

        params = {
            'bandwidth_mode': rng.choice(['constant', 'adaptive']),
            'bandwidth': np.exp(rng.uniform(np.log(0.4), np.log(80.0))),
            'reg': np.exp(rng.uniform(np.log(1e-5), np.log(50.))),
            'exponent': rng.uniform(0.7, 1.3),
            'p_interp': rng.uniform(0., 0.8),
            'tfms': num_tfms + cat_tfms,
            'diag': rng.choice([False, True]),
            'kernel_type': rng.choice(['lpq_kermac', 'l2'], p=[0.8, 0.2]),
            # 'max_leaf_samples': 60_000,  # don't put it here, it's the default anyway and can be overridden
            'rfm_iters': 5,
            'classification_mode': 'zero_one',
            'binary_solver': 'solve',
            'early_stop_rfm': True,
            'early_stop_multiplier': 1.1,  # early stop if val metric > esm * best val metric (for loss)
            'split_method': 'top_vector_agop_on_subset',
            # 'overlap_fraction': 0.0,
            # 'use_temperature_tuning': False,
        }
    elif hpo_space_name == 'large-soft-pca':
        # used on meta-test in the paper
        num_tfms_list = [['mean_center', 'l2_normalize']]
        num_tfms = num_tfms_list[rng.integers(len(num_tfms_list))]
        cat_tfms_list = [['ordinal_encoding'], ['one_hot']]
        cat_tfms = cat_tfms_list[rng.integers(len(cat_tfms_list))]

        params = {
            'bandwidth_mode': rng.choice(['constant', 'adaptive']),
            'bandwidth': np.exp(rng.uniform(np.log(0.4), np.log(80.0))),
            'reg': np.exp(rng.uniform(np.log(1e-5), np.log(50.))),
            'exponent': rng.uniform(0.7, 1.3),
            'p_interp': rng.uniform(0., 0.8),
            'tfms': num_tfms + cat_tfms,
            'diag': rng.choice([False, True]),
            'kernel_type': rng.choice(['lpq_kermac', 'l2'], p=[0.8, 0.2]),
            # 'max_leaf_samples': 60_000,  # don't put it here, it's the default anyway and can be overridden
            'rfm_iters': 5,
            'classification_mode': 'zero_one',
            'binary_solver': 'solve',
            'early_stop_rfm': True,
            'early_stop_multiplier': 1.1,  # early stop if val metric > esm * best val metric (for loss)
            'split_method': 'pca',
            # 'overlap_fraction': 0.0,
            # 'use_temperature_tuning': False,
        }
    elif hpo_space_name == 'large-temptune':
        # used on meta-test in the paper
        num_tfms_list = [['mean_center', 'l2_normalize']]
        num_tfms = num_tfms_list[rng.integers(len(num_tfms_list))]
        cat_tfms_list = [['ordinal_encoding'], ['one_hot']]
        cat_tfms = cat_tfms_list[rng.integers(len(cat_tfms_list))]

        params = {
            'bandwidth_mode': rng.choice(['constant', 'adaptive']),
            'bandwidth': np.exp(rng.uniform(np.log(0.4), np.log(80.0))),
            'reg': np.exp(rng.uniform(np.log(1e-5), np.log(50.))),
            'exponent': rng.uniform(0.7, 1.3),
            'p_interp': rng.uniform(0., 0.8),
            'tfms': num_tfms + cat_tfms,
            'diag': rng.choice([False, True]),
            'kernel_type': rng.choice(['lpq_kermac', 'l2'], p=[0.8, 0.2]),
            # 'max_leaf_samples': 60_000,  # don't put it here, it's the default anyway and can be overridden
            'rfm_iters': 5,
            'classification_mode': 'zero_one',
            'binary_solver': 'solve',
            'early_stop_rfm': True,
            'early_stop_multiplier': 1.1,  # early stop if val metric > esm * best val metric (for loss)
            'split_method': 'top_vector_agop_on_subset',
            'overlap_fraction': 0.0,
            # 'use_temperature_tuning': False,
            'temp_tuning_space': [0.0] + list(np.logspace(np.log10(0.025), np.log10(4.5), num=15))
        }
    elif hpo_space_name == 'large-temptune-pca':
        # used on meta-test in the paper
        num_tfms_list = [['mean_center', 'l2_normalize']]
        num_tfms = num_tfms_list[rng.integers(len(num_tfms_list))]
        cat_tfms_list = [['ordinal_encoding'], ['one_hot']]
        cat_tfms = cat_tfms_list[rng.integers(len(cat_tfms_list))]

        params = {
            'bandwidth_mode': rng.choice(['constant', 'adaptive']),
            'bandwidth': np.exp(rng.uniform(np.log(0.4), np.log(80.0))),
            'reg': np.exp(rng.uniform(np.log(1e-5), np.log(50.))),
            'exponent': rng.uniform(0.7, 1.3),
            'p_interp': rng.uniform(0., 0.8),
            'tfms': num_tfms + cat_tfms,
            'diag': rng.choice([False, True]),
            'kernel_type': rng.choice(['lpq_kermac', 'l2'], p=[0.8, 0.2]),
            # 'max_leaf_samples': 60_000,  # don't put it here, it's the default anyway and can be overridden
            'rfm_iters': 5,
            'classification_mode': 'zero_one',
            'binary_solver': 'solve',
            'early_stop_rfm': True,
            'early_stop_multiplier': 1.1,  # early stop if val metric > esm * best val metric (for loss)
            'split_method': 'pca',
            'overlap_fraction': 0.0,
            # 'use_temperature_tuning': False,
            'temp_tuning_space': [0.0] + list(np.logspace(np.log10(0.025), np.log10(4.5), num=15))
        }
    elif hpo_space_name == 'large-temptune-rf':
        # used on meta-test in the paper
        num_tfms_list = [['mean_center', 'l2_normalize']]
        num_tfms = num_tfms_list[rng.integers(len(num_tfms_list))]
        cat_tfms_list = [['ordinal_encoding'], ['one_hot']]
        cat_tfms = cat_tfms_list[rng.integers(len(cat_tfms_list))]

        params = {
            'bandwidth_mode': rng.choice(['constant', 'adaptive']),
            'bandwidth': np.exp(rng.uniform(np.log(0.4), np.log(80.0))),
            'reg': np.exp(rng.uniform(np.log(1e-5), np.log(50.))),
            'exponent': rng.uniform(0.7, 1.3),
            'p_interp': rng.uniform(0., 0.8),
            'tfms': num_tfms + cat_tfms,
            'diag': rng.choice([False, True]),
            'kernel_type': rng.choice(['lpq_kermac', 'l2'], p=[0.8, 0.2]),
            # 'max_leaf_samples': 60_000,  # don't put it here, it's the default anyway and can be overridden
            'rfm_iters': 5,
            'classification_mode': 'zero_one',
            'binary_solver': 'solve',
            'early_stop_rfm': True,
            'early_stop_multiplier': 1.1,  # early stop if val metric > esm * best val metric (for loss)
            'split_method': 'rf_criterion',
            'overlap_fraction': 0.0,
            # 'use_temperature_tuning': False,
            'temp_tuning_space': [0.0] + list(np.logspace(np.log10(0.025), np.log10(4.5), num=15))
        }
    else:
        raise ValueError(f'Unknown {hpo_space_name=}')

    return params


class RandomParamsxRFMAlgInterface(RandomParamsAlgInterface):
    def _sample_params(self, is_classification: bool, seed: int, n_train: int):
        return sample_xrfm_params(seed, self.config.get('hpo_space_name', 'default'))

    def _create_interface_from_config(self, n_tv_splits: int, **config):
        return SingleSplitWrapperAlgInterface([xRFMSubSplitInterface(**config) for i in range(n_tv_splits)])
