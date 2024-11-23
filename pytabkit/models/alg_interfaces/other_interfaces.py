import os
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, \
    GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from pytabkit.models.alg_interfaces.alg_interfaces import RandomParamsAlgInterface
from pytabkit.models.alg_interfaces.resource_computation import ResourcePredictor
from pytabkit.models.alg_interfaces.base import RequiredResources
from pytabkit.models.alg_interfaces.sub_split_interfaces import SklearnSubSplitInterface, SingleSplitWrapperAlgInterface
from pytabkit.models import utils
from pytabkit.models.data.data import DictDataset


class RFSubSplitInterface(SklearnSubSplitInterface):
    def _create_sklearn_model(self, seed: int, n_threads: int, gpu_devices: List[str]) -> Any:
        params_config = [('n_estimators', None),
                         ('criterion', None),
                         ('max_depth', None),
                         ('min_samples_split', None),
                         ('max_features', None),
                         ('min_samples_leaf', None),
                         ('bootstrap', None),
                         ('min_impurity_decrease', None),
                         ('min_weight_fraction_leaf', None),
                         ('n_jobs', ['n_jobs', 'n_threads'], n_threads),
                         ('verbose', ['verbose', 'verbosity'])]

        params = utils.extract_params(self.config, params_config)
        if self.n_classes > 0:
            return RandomForestClassifier(random_state=seed, **params)
        else:
            train_metric_name = self.config.get('train_metric_name', None)
            if train_metric_name == 'mse':
                params['criterion'] = 'squared_error'  # is the default anyway
            elif train_metric_name == 'mae':
                params['criterion'] = 'absolute_error'
            elif train_metric_name is not None:
                raise ValueError(f'Train metric "{train_metric_name}" is currently not supported!')
            return RandomForestRegressor(random_state=seed, **params)

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        assert n_cv == 1
        assert n_refit == 0
        assert n_splits == 1
        updated_config = utils.join_dicts(dict(n_estimators=100), self.config)
        time_params = {'': 0.5, 'ds_size_gb': 10.0, '1/n_threads*n_samples*n_estimators*n_tree_repeats': 4e-8}
        ram_params = {'': 0.5, 'ds_size_gb': 3.0, 'n_samples*n_estimators*n_tree_repeats': 3e-9}
        rc = ResourcePredictor(config=updated_config, time_params=time_params,
                               cpu_ram_params=ram_params)
        return rc.get_required_resources(ds)


class RandomParamsRFAlgInterface(RandomParamsAlgInterface):
    def _sample_params(self, is_classification: bool, seed: int, n_train: int):
        rng = np.random.default_rng(seed)
        # adapted from Grinsztajn et al. (2022)
        space = {
            'n_estimators': 250,
            'max_depth': rng.choice([None, 2, 3, 4], p=[0.7, 0.1, 0.1, 0.1]),
            'criterion': rng.choice(['gini', 'entropy']) if is_classification
                            else rng.choice(['squared_error', 'absolute_error']),
            'max_features': rng.choice(['sqrt', 'sqrt', 'log2', None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
            'min_samples_split': rng.choice([2, 3], p=[0.95, 0.05]),
            'min_samples_leaf': round(np.exp(rng.uniform(np.log(1.5), np.log(50.5)))),
            'bootstrap': rng.choice([True, False]),
            'min_impurity_decrease': rng.choice([0.0, 0.01, 0.02, 0.05], p=[0.85, 0.05, 0.05, 0.05]),
            'tfms': ['one_hot'],
        }
        return space

    def _create_interface_from_config(self, n_tv_splits: int, **config):
        return SingleSplitWrapperAlgInterface([RFSubSplitInterface(**config) for i in range(n_tv_splits)])


class GBTSubSplitInterface(SklearnSubSplitInterface):
    def _create_sklearn_model(self, seed: int, n_threads: int, gpu_devices: List[str]) -> Any:
        params_config = [('n_estimators', None),
                         ('learning_rate', None),
                         ('subsample', None),
                         ('max_depth', None),
                         ('verbose', ['verbose', 'verbosity'])]

        params = utils.extract_params(self.config, params_config)
        if self.n_classes > 0:
            return GradientBoostingClassifier(random_state=seed, **params)
        else:
            train_metric_name = self.config.get('train_metric_name', 'mse')
            if train_metric_name == 'mse':
                pass  # is the default anyway
            elif train_metric_name.startswith('pinball('):
                quantile = float(train_metric_name[len('pinball('):-1])
                params['loss'] = f'quantile'
                params['alpha'] = quantile
            elif train_metric_name == 'mae':
                params['loss'] = 'absolute_error'
            else:
                raise ValueError(f'Train metric "{train_metric_name}" is currently not supported!')
            return GradientBoostingRegressor(random_state=seed, **params)

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        assert n_cv == 1
        assert n_refit == 0
        assert n_splits == 1
        updated_config = utils.join_dicts(dict(n_estimators=100), self.config)
        time_params = {'': 0.5, 'ds_size_gb': 10.0, '1/n_threads*n_samples*n_estimators*n_tree_repeats': 4e-8}
        ram_params = {'': 0.5, 'ds_size_gb': 3.0, 'n_samples*n_estimators*n_tree_repeats': 3e-9}
        rc = ResourcePredictor(config=updated_config, time_params=time_params,
                               cpu_ram_params=ram_params)
        return rc.get_required_resources(ds)


class SklearnMLPSubSplitInterface(SklearnSubSplitInterface):
    def _create_sklearn_model(self, seed: int, n_threads: int, gpu_devices: List[str]) -> Any:
        params_config = []  # todo: add parameters
        params = utils.extract_params(self.config, params_config)
        if self.n_classes > 0:
            return MLPClassifier(random_state=seed, **params)
        else:
            reg = MLPRegressor(random_state=seed, **params)
            return TransformedTargetRegressor(regressor=reg, transformer=StandardScaler())

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        assert n_cv == 1
        assert n_refit == 0
        assert n_splits == 1
        updated_config = utils.join_dicts(dict(n_estimators=100), self.config)
        time_params = {'': 0.5, 'ds_onehot_size_gb': 10.0, '1/n_threads*n_samples': 4e-5}
        ram_params = {'': 0.5, 'ds_onehot_size_gb': 5.0}
        rc = ResourcePredictor(config=updated_config, time_params=time_params,
                               cpu_ram_params=ram_params)
        return rc.get_required_resources(ds)


class KANSubSplitInterface(SklearnSubSplitInterface):
    def _create_sklearn_model(self, seed: int, n_threads: int, gpu_devices: List[str]) -> Any:
        import imodelsx.kan
        params_config = []  # todo: add parameters
        params = utils.extract_params(self.config, params_config)
        params['device'] = 'cpu' if len(gpu_devices) == 0 else gpu_devices[0]
        if self.n_classes > 0:
            return imodelsx.kan.KANClassifier(**params)
        else:
            reg = imodelsx.kan.KANRegressor(**params)
            return TransformedTargetRegressor(regressor=reg, transformer=StandardScaler())

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        assert n_cv == 1
        assert n_refit == 0
        assert n_splits == 1
        updated_config = utils.join_dicts(dict(n_estimators=100, max_n_threads=2), self.config)
        time_params = {'': 10, 'ds_onehot_size_gb': 10.0, 'n_samples': 8e-5}
        ram_params = {'': 0.15, 'ds_onehot_size_gb': 1.5}
        gpu_ram_params = {'': 0.4, 'n_features': 1e-4}
        rc = ResourcePredictor(config=updated_config, time_params=time_params, gpu_ram_params=gpu_ram_params,
                               cpu_ram_params=ram_params, n_gpus=1, gpu_usage=0.02)  # , gpu_ram_params)
        return rc.get_required_resources(ds)

    def _fit_sklearn(self, x_df: pd.DataFrame, y: np.ndarray, val_idxs: np.ndarray,
                     cat_col_names: Optional[List[str]] = None):
        # by default, we ignore the validation set since most sklearn methods do not support it
        n_samples = len(x_df)
        train_mask = np.ones(shape=(n_samples,), dtype=np.bool_)
        train_mask[val_idxs] = False
        # give train+valid to KAN since it does its own train+valid split
        # (even though that one uses 20% valid instead of 25%)
        # x_df = x_df.iloc[train_mask, :]
        x_np = x_df.to_numpy()
        # y = y[train_mask]
        if cat_col_names is not None and len(cat_col_names) > 0:
            self.model.fit(x_np, y, **{self._get_cat_indexes_arg_name(): cat_col_names})
        else:
            self.model.fit(x_np, y)

    def _predict_sklearn(self, x_df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(x_df.to_numpy())

    def _predict_proba_sklearn(self, x_df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(x_df.to_numpy())


class GrandeWrapper:
    """
    Wrapper class for GRANDE that allows to pass cat_features in fit() instead of the constructor.
    """
    def __init__(self, **config):
        self.config = config

    def fit(self, X, y, X_val, y_val, cat_features: Optional[List[str]] = None):
        # params_config = []  # todo: add parameters
        # params = utils.extract_params(self.config, params_config)
        params = {
            'depth': 5,  # tree depth
            'n_estimators': 2048,  # number of estimators / trees

            'learning_rate_weights': 0.005,  # learning rate for leaf weights
            'learning_rate_index': 0.01,  # learning rate for split indices
            'learning_rate_values': 0.01,  # learning rate for split values
            'learning_rate_leaf': 0.01,  # learning rate for leafs (logits)

            'optimizer': 'adam',  # optimizer
            'cosine_decay_steps': 0,  # decay steps for lr schedule (CosineDecayRestarts)

            # loss function (default 'crossentropy' for binary & multi-class classification and 'mse' for regression)
            'focal_loss': False,  # use focal loss {True, False}
            'temperature': 0.0,  # temperature for stochastic re-weighted GD (0.0, 1.0)

            'from_logits': True,  # use logits for weighting {True, False}
            'use_class_weights': True,  # use class weights for training {True, False}

            'dropout': 0.0,
            # dropout rate (here, dropout randomly disables individual estimators of the ensemble during training)

            'selected_variables': 0.8,  # feature subset percentage (0.0, 1.0)
            'data_subset_fraction': 1.0,  # data subset percentage (0.0, 1.0)
        }

        args = {
            'epochs': 1,  # number of epochs for training
            'early_stopping_epochs': 25,  # patience for early stopping (best weights are restored)
            'batch_size': 64,  # batch size for training
            'random_seed': 42,
            'verbose': 1,
        }

        if issubclass(y.dtype.type, np.floating):
            print(f'regression')
            self.is_regression_ = True
            params['loss'] = 'mse'
            args['objective'] = 'regression'
        elif len(np.unique(y)) <= 2:
            self.is_regression_ = False
            params['loss'] = 'crossentropy'
            args['objective'] = 'binary'
        else:
            self.is_regression_ = False
            params['loss'] = 'crossentropy'
            args['objective'] = 'classification'

        if cat_features is not None:
            args['cat_idx'] = [X.columns.get_loc(name) for name in cat_features]
        else:
            args['cat_idx'] = []

        device = self.config.get('device', 'cpu')
        if device.startswith('cuda'):
            gpu_idx_str = device[len('cuda:'):]
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx_str
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


        from GRANDE import GRANDE

        self.model_ = GRANDE(params=params, args=args)
        self.model_.fit(X.copy(), y, X_val.copy(), y_val)

    def predict_proba(self, X):
        return self.model_.predict(X)

    def predict(self, X):
        y_pred = self.model_.predict(X)
        if not self.is_regression_:
            return np.argmax(y_pred, axis=1)
        else:
            return y_pred


class GrandeSubSplitInterface(SklearnSubSplitInterface):
    def _create_sklearn_model(self, seed: int, n_threads: int, gpu_devices: List[str]) -> Any:
        model = GrandeWrapper(**self.config, device='cpu' if len(gpu_devices) == 0 else gpu_devices[0])
        # if self.n_classes == 0:  # doesn't work with validation sets anyway
        #     model = TransformedTargetRegressor(regressor=model, transformer=StandardScaler())
        return model

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        assert n_cv == 1
        assert n_refit == 0
        assert n_splits == 1
        updated_config = utils.join_dicts(dict(n_estimators=100), self.config)
        time_params = {'': 0.5, 'ds_onehot_size_gb': 10.0, '1/n_threads*n_samples': 4e-5}
        ram_params = {'': 0.5, 'ds_onehot_size_gb': 5.0}
        rc = ResourcePredictor(config=updated_config, time_params=time_params,
                               cpu_ram_params=ram_params)
        return rc.get_required_resources(ds)

    def _fit_sklearn(self, x_df: pd.DataFrame, y: np.ndarray, val_idxs: np.ndarray,
                     cat_col_names: Optional[List[str]] = None):
        # by default, we ignore the validation set since most sklearn methods do not support it
        n_samples = len(x_df)
        train_mask = np.ones(shape=(n_samples,), dtype=np.bool_)
        train_mask[val_idxs] = False
        x_val_df = x_df.iloc[~train_mask, :]
        y_val_df = y[~train_mask]
        x_df = x_df.iloc[train_mask, :]
        y = y[train_mask]
        if cat_col_names is not None and len(cat_col_names) > 0:
            self.model.fit(x_df, y, x_val_df, y_val_df, cat_features=cat_col_names)
        else:
            self.model.fit(x_df, y, x_val_df, y_val_df)

