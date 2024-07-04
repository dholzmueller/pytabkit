from typing import List, Any, Optional, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from skorch.dataset import Dataset
from skorch.helper import predefined_split

from pytabkit.models.alg_interfaces.resource_computation import ResourcePredictor
from pytabkit.models import utils
from pytabkit.models.alg_interfaces.alg_interfaces import AlgInterface, SingleSplitAlgInterface
from pytabkit.models.alg_interfaces.sub_split_interfaces import SklearnSubSplitInterface, SingleSplitWrapperAlgInterface
from pytabkit.models.alg_interfaces.base import SplitIdxs, InterfaceResources, RequiredResources
from pytabkit.models.data.data import DictDataset
from pytabkit.models.training.logging import Logger
from pytabkit.models.nn_models.rtdl_resnet import create_mlp_classifier_skorch, create_mlp_regressor_skorch, \
    create_resnet_classifier_skorch, create_resnet_regressor_skorch
from pytabkit.models.training.metrics import insert_missing_class_columns


class SkorchSubSplitInterface(SklearnSubSplitInterface):
    def _fit_sklearn(self, x_df: pd.DataFrame, y: np.ndarray, val_idxs: np.ndarray,
                    cat_col_names: Optional[List[str]] = None):
        # set number of classes
        if self.n_classes > 0: #classification
            self.model.set_n_classes(self.n_classes)
        # get transformed_target from config
        transformed_target = self.config.get("transformed_target", False)
        if transformed_target:
            #do TransformedTargetRegressor by hand (because setting the
            # validation set in skorch conflicts with TransformedTargetRegressor)
            self.transformer = StandardScaler()
            y = self.transformer.fit_transform(y.reshape(-1, 1))
        else:
            self.transformer = None
        n_samples = len(x_df)
        train_mask = np.ones(shape=(n_samples,), dtype=np.bool_)
        train_mask[val_idxs] = False
        # create val_ds for skorch (see FAQ)
        # Note that this break TransformedTargetRegressor, which is why we do it by hand
        x_train =  np.array(x_df.iloc[train_mask, :], dtype=np.float32)
        x_val = np.array(x_df.iloc[~train_mask, :], dtype=np.float32)
        y_train = y[train_mask]
        y_val = y[~train_mask] if self.n_classes else y[~train_mask].reshape(-1, 1)
        self.categorical_indicator = None
        if cat_col_names is not None and len(cat_col_names) > 0:
            self.categorical_indicator = np.array([name in cat_col_names for name in x_df.columns])
            self.model.set_categorical_indicator(self.categorical_indicator)
            # we do OrdinalEncoder one more time to be sure that there are no "holes"
            # in the categories
            # missing values were encoded as zero, we need to make them missing again
            self.replace_zero_by_nans = SimpleImputer(missing_values=0.,
                                    strategy="constant",
                                    fill_value=np.nan)
            x_train[:, self.categorical_indicator] = self.replace_zero_by_nans.fit_transform(x_train[:, self.categorical_indicator])
            self.ord_enc = OrdinalEncoder(dtype=np.float32, handle_unknown='use_encoded_value', unknown_value=-1,
                                                encoded_missing_value=-1)
            x_train[:, self.categorical_indicator] = self.ord_enc.fit_transform(x_train[:, self.categorical_indicator])
            x_val[:, self.categorical_indicator] = self.replace_zero_by_nans.transform(x_val[:, self.categorical_indicator])
            x_val[:, self.categorical_indicator] = self.ord_enc.transform(x_val[:, self.categorical_indicator])
        val_ds = Dataset(x_val, y_val)
        self.model.set_params(train_split=predefined_split(val_ds))

        self.model.fit(x_train, y_train)

    def predict(self, ds: DictDataset) -> torch.Tensor:
        # adapted from SklearnSubSplitLearner
        # should return tensor of shape len(ds) x output_shape
        if self.tfm is not None:
            ds = self.tfm.forward_ds(ds)

        x_df = ds.without_labels().to_df()
        x_array = np.array(x_df, dtype=np.float32)  # added

        if self.categorical_indicator is not None:
            x_array[:, self.categorical_indicator] = self.replace_zero_by_nans.transform(x_array[:, self.categorical_indicator])
            x_array[:, self.categorical_indicator] = self.ord_enc.transform(x_array[:, self.categorical_indicator])

        # skorch doesn't support pandas dataframe

        if self.n_classes > 0:
            # classification
            y_pred = np.log(self.model.predict_proba(x_array) + 1e-30)
        else:
            # regression
            y_pred = self.model.predict(x_array)
            if len(y_pred.shape) == 1:
                y_pred = y_pred[:, None]

        y_pred = torch.as_tensor(y_pred, dtype=torch.float32)
        # guard against missing classes in the training set
        # (GBDT interfaces don't need this because they get passed n_classes as a parameter)
        y_pred = insert_missing_class_columns(y_pred, self.train_ds)
        # added
        if self.transformer is not None:
            y_pred = self.transformer.inverse_transform(y_pred.reshape(-1, 1))
            # transform to tensor
            y_pred = torch.from_numpy(y_pred)
        return y_pred[None]  # add vectorized dimension


class RTDL_MLPSubSplitInterface(SkorchSubSplitInterface):
    def _create_sklearn_model(self, seed: int, n_threads: int, gpu_devices: List[str]) -> Any:
        # the random state is handled by SklearnSubSplitLearner.fit() which sets
        # numpy and torch seeds based on self.random_state
        # which is all we need for skorch, so
        # we don't need to use seed here
        params_config = [
            ("lr_scheduler", None),
            ("module__n_layers", None),
            ("module__d_layers", None),
            ("module__d_first_layer", None),
            ("module__d_last_layer", None),
            ("module__activation", None),
            ("module__dropout", None),
            ("lr", None),
            ("optimizer", None),
            ("module__d_embedding", None),
            ("batch_size", None),
            ("max_epochs", None),
            ("use_checkpoints", None),
            ("es_patience", None),
            ("lr_patience", None),
            ("verbose", None),
            ("checkpoint_dir", "tmp_folder"),
            ("val_metric_name", None),
        ]
        params = utils.extract_params(self.config, params_config)
        params['device'] = 'cpu' if len(gpu_devices) == 0 else gpu_devices[0]
        if 'checkpoint_dir' not in params or params['checkpoint_dir'] is None:
            params['checkpoint_dir'] = './rtdl_checkpoints'
        if self.n_classes > 0:
            return create_mlp_classifier_skorch(**params)
        else:
            return create_mlp_regressor_skorch(**params)

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int]) -> RequiredResources:
        assert n_cv == 1
        assert n_refit == 0
        assert n_splits == 1
        updated_config = utils.join_dicts(dict(n_estimators=100, max_n_threads=2), self.config)
        time_params = {'': 10, 'ds_onehot_size_gb': 10.0, 'n_samples': 8e-5}
        ram_params = {'': 0.15, 'ds_onehot_size_gb': 1.5}
        gpu_ram_params = {'': 0.4, 'n_features': 1e-4}
        rc = ResourcePredictor(config=updated_config, time_params=time_params, gpu_ram_params=gpu_ram_params,
                               cpu_ram_params=ram_params, n_gpus=1, gpu_usage=0.02)#, gpu_ram_params)
        return rc.get_required_resources(ds)


class ResnetSubSplitInterface(SkorchSubSplitInterface):
    def _create_sklearn_model(self, seed: int, n_threads: int, gpu_devices: List[str]) -> Any:
        # the random state is handled by SklearnSubSplitLearner.fit() which sets
        # numpy and torch seeds based on self.random_state
        # which is all we need for skorch, so
        # we don't need to use seed here
        params_config = [
            ("lr_scheduler", None),
            ("module__activation", None),
            ("module__normalization", None),
            ("module__n_layers", None),
            ("module__d", None),
            ("module__d_hidden_factor", None),
            ("module__hidden_dropout", None),
            ("module__residual_dropout", None),
            ("lr", None),
            ("optimizer__weight_decay", None),
            ("optimizer", None),
            ("module__d_embedding", None),
            ("batch_size", None),
            ("max_epochs", None),
            ("use_checkpoints", None),
            ("es_patience", None),
            ("lr_patience", None),
            ("verbose", None),
            ("checkpoint_dir", "tmp_folder"),
            ("val_metric_name", None),
        ]
        params = utils.extract_params(self.config, params_config)
        params['device'] = 'cpu' if len(gpu_devices) == 0 else gpu_devices[0]
        if 'checkpoint_dir' not in params or params['checkpoint_dir'] is None:
            params['checkpoint_dir'] = './rtdl_checkpoints'
        if self.n_classes > 0:
            return create_resnet_classifier_skorch(**params)
        else:
            return create_resnet_regressor_skorch(**params)

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int]) -> RequiredResources:
        assert n_cv == 1
        assert n_refit == 0
        assert n_splits == 1
        updated_config = utils.join_dicts(dict(n_estimators=100, max_n_threads=2), self.config)
        time_params = {'': 10, 'ds_onehot_size_gb': 10.0, 'n_samples': 8e-5}
        ram_params = {'': 0.15, 'ds_onehot_size_gb': 1.5}
        gpu_ram_params = {'': 0.4, 'n_features': 1e-4}
        rc = ResourcePredictor(config=updated_config, time_params=time_params, gpu_ram_params=gpu_ram_params,
                               cpu_ram_params=ram_params, n_gpus=1, gpu_usage=0.02)  #, gpu_ram_params)
        return rc.get_required_resources(ds)


def choose_batch_size_rtdl(train_size) -> int:
    # set batch_size depending on the number of samples
    # as in the rtdl paper
    # if train_size < 10_000:
    #     return 128  # taken from tabr paper, not used in our paper due to a bug
    if train_size < 30_000:
        return 256
    elif train_size < 100_000:
        return 512
    else:
        return 1024


class RTDL_MLP_ParamSampler:
    def __init__(self, is_classification: bool, train_size: int):
        self.is_classification = is_classification
        self.train_size = train_size

    def sample_params(self, seed: int) -> Dict[str, Any]:
        rng = np.random.default_rng(seed=seed)
        # cutoff to change hp space for large datasets
        # as in rtdl
        # the cutoff is between 70K and 300K
        cutoff_train_size_rtdl = 100_000
        is_large_dataset = self.train_size > cutoff_train_size_rtdl

        params = {
            # reduced d_layers
            "module__n_layers": rng.choice(np.arange(1, 12)) if is_large_dataset \
                else rng.choice(np.arange(1, 9)),
            "module__d_layers": rng.choice(np.arange(1, 1025)) if is_large_dataset \
                    else rng.choice(np.arange(1, 513)),
            # "Note that the size of the first and the last layers are tuned and set separately, while the size for
            #“in-between” layers is the same for all of them." from rtdl paper
            "module__d_first_layer": rng.choice(np.arange(1, 1025)) if is_large_dataset \
                    else rng.choice(np.arange(1, 513)),
            "module__d_last_layer": rng.choice(np.arange(1, 1025)) if is_large_dataset \
                    else rng.choice(np.arange(1, 513)),
            "module__dropout": rng.choice([rng.uniform(0, 0.5)] + [0.]),
            "lr": np.exp(rng.uniform(np.log(1e-5), np.log(1e-2))),
            "optimizer__weight_decay": rng.choice(
                [np.exp(rng.uniform(np.log(1e-6), np.log(1e-3)))] + [0.]
                ),
            "module__d_embedding": rng.choice(np.arange(8, 32)), # we go lower (than 64) 
                            #because we have smaller datasets with categorical features
            "batch_size": choose_batch_size_rtdl(self.train_size),
            "lr_scheduler": False,
            "optimizer": "adamw",
            "max_epochs": 400,
            "use_checkpoints": True,
            "es_patience": 16,
            'verbose': 0,
            'tfms': ['quantile_tabr'],
        }

        if self.is_classification:
            params["transformed_target"] = False
        else:
            params["transformed_target"] = True

        return params
    
    
class RTDL_ResNet_ParamSampler:
    def __init__(self, is_classification: bool, train_size: int):
        self.is_classification = is_classification
        self.train_size = train_size

    def sample_params(self, seed: int) -> Dict[str, Any]:
        rng = np.random.default_rng(seed=seed)
        # cutoff to change hp space for large datasets
        # as in rtdl
        # the cutoff is between 70K and 300K
        cutoff_train_size_rtdl = 100_000
        is_large_dataset = self.train_size > cutoff_train_size_rtdl

        params = {
            "module__n_layers": rng.choice(np.arange(1, 17)) if is_large_dataset \
                  else rng.choice(np.arange(1, 9)),
            "module__d": rng.choice(np.arange(64, 1025)) if is_large_dataset \
                    else rng.choice(np.arange(64, 513)),
            "module__d_hidden_factor": rng.choice(np.arange(1, 5)),
            "module__hidden_dropout": rng.uniform(0.0, 0.5),
            "module__residual_dropout": rng.choice([rng.uniform(0, 0.5)] + [0.]),
            "lr": np.exp(rng.uniform(np.log(1e-5), np.log(1e-2))),
            "optimizer__weight_decay": rng.choice(
                [np.exp(rng.uniform(np.log(1e-6), np.log(1e-3)))] + [0.]
                ),
            "module__d_embedding": rng.choice(np.arange(8, 32)), # we go lower (than 64) 
                            #because we have smaller datasets with categorical features
            "batch_size": choose_batch_size_rtdl(self.train_size),
            "module__activation": "relu",
            "module__normalization": "batchnorm",
            "lr_scheduler": False,
            "optimizer": "adamw",
            "max_epochs": 400,
            "use_checkpoints": True,
            "es_patience": 16,
            'verbose': 0,
            'tfms': ['quantile_tabr'],
        }

        if self.is_classification:
            params["transformed_target"] = False
        else:
            params["transformed_target"] = True


        return params
    

class RandomParamsResnetAlgInterface(SingleSplitAlgInterface):
    def __init__(self, model_idx: int, fit_params: Optional[List[Dict[str, Any]]] = None, **config):
        super().__init__(fit_params=fit_params, **config)
        self.model_idx = model_idx
        self.alg_interface = None

    def get_refit_interface(self, n_refit: int, fit_params: Optional[List[Dict]] = None) -> 'AlgInterface':
        return RandomParamsResnetAlgInterface(model_idx=self.model_idx, fit_params=fit_params or self.fit_params,
                                          **self.config)

    def _create_sub_interface(self, ds: DictDataset, seed: int):
        # this is also set in get_required_resources, but okay
        if self.fit_params is None:
            hparam_seed = utils.combine_seeds(seed, self.model_idx)
            is_classification = not ds.tensor_infos['y'].is_cont()
            train_size = ds.n_samples #TODO: get just train?
            self.fit_params = [RTDL_ResNet_ParamSampler(is_classification, 
                                                        train_size).sample_params(hparam_seed)]
        return SingleSplitWrapperAlgInterface([ResnetSubSplitInterface(**utils.update_dict(self.config, self.fit_params[0]))])

    def fit(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources,
            logger: Logger, tmp_folders: List[Optional[Path]], name: str) -> None:
        assert len(idxs_list) == 1
        self.alg_interface = self._create_sub_interface(ds, idxs_list[0].split_seed)
        self.alg_interface.fit(ds, idxs_list, interface_resources, logger, tmp_folders, name)

    def predict(self, ds: DictDataset) -> torch.Tensor:
        return self.alg_interface.predict(ds)

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int]) -> RequiredResources:
        assert len(split_seeds) == 1
        alg_interface = self._create_sub_interface(ds, split_seeds[0])
        return alg_interface.get_required_resources(ds, n_cv, n_refit, n_splits, split_seeds)
    
    
class RandomParamsRTDLMLPAlgInterface(SingleSplitAlgInterface):
    def __init__(self, model_idx: int, fit_params: Optional[List[Dict[str, Any]]] = None, **config):
        super().__init__(fit_params=fit_params, **config)
        self.model_idx = model_idx
        self.alg_interface = None

    def get_refit_interface(self, n_refit: int, fit_params: Optional[List[Dict]] = None) -> 'AlgInterface':
        return RandomParamsRTDLMLPAlgInterface(model_idx=self.model_idx, fit_params=fit_params or self.fit_params,
                                          **self.config)

    def _create_sub_interface(self, ds: DictDataset, seed: int):
        # this is also set in get_required_resources, but okay
        if self.fit_params is None:
            hparam_seed = utils.combine_seeds(seed, self.model_idx)
            is_classification = not ds.tensor_infos['y'].is_cont()
            train_size = ds.n_samples #TODO: get just train?
            self.fit_params = [RTDL_MLP_ParamSampler(is_classification, 
                                                        train_size).sample_params(hparam_seed)]
        return SingleSplitWrapperAlgInterface([RTDL_MLPSubSplitInterface(**utils.update_dict(self.config, self.fit_params[0]))])

    def fit(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources,
            logger: Logger, tmp_folders: List[Optional[Path]], name: str) -> None:
        assert len(idxs_list) == 1
        self.alg_interface = self._create_sub_interface(ds, idxs_list[0].split_seed)
        self.alg_interface.fit(ds, idxs_list, interface_resources, logger, tmp_folders, name)

    def predict(self, ds: DictDataset) -> torch.Tensor:
        return self.alg_interface.predict(ds)

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int]) -> RequiredResources:
        assert len(split_seeds) == 1
        alg_interface = self._create_sub_interface(ds, split_seeds[0])
        return alg_interface.get_required_resources(ds, n_cv, n_refit, n_splits, split_seeds)