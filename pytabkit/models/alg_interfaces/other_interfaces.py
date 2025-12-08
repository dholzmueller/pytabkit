import os
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, \
    GradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor
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
                         ('max_leaf_nodes', None),
                         ('max_samples', None),
                         ('n_jobs', ['n_jobs', 'n_threads'], n_threads),
                         ('verbose', ['verbose', 'verbosity'])]

        params = utils.extract_params(self.config, params_config)
        if not params.get('bootstrap', True) and 'max_samples' in params:
            del params['max_samples']
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
            reg = RandomForestRegressor(random_state=seed, **params)
            if self.config.get('standardize_target', False):
                reg = TransformedTargetRegressor(reg, transformer=StandardScaler())
            return reg

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
        hpo_space_name = self.config.get('hpo_space_name', 'grinsztajn')
        if hpo_space_name == 'grinsztajn':
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
        elif hpo_space_name == 'large-v1':
            space = {
                'n_estimators': 300,
                # this wasn't used in the experiments
                # 'max_leaf_nodes': round(np.exp(rng.uniform(np.log(500), np.log(100_000)))),
                'max_depth': rng.choice([None, 2, 3, 4, 6, 8, 12, 16]),
                'criterion': rng.choice(['gini', 'entropy']) if is_classification
                else rng.choice(['squared_error', 'absolute_error']),
                'max_features': rng.choice(['sqrt', 'log2', None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(32.0)))),
                'min_samples_leaf': round(np.exp(rng.uniform(np.log(0.6), np.log(128.0)))),
                'bootstrap': rng.choice([True, False]),
                'min_impurity_decrease': rng.choice([0.0, np.exp(rng.uniform(np.log(1e-3), np.log(5e-1)))]),
                'tfms': [['one_hot'], ['ordinal_encoding']][rng.integers(0, 1, endpoint=True)],
            }
        elif hpo_space_name == 'large-v2':
            # large-v1 but reduced max_depth, criterion, min_samples_leaf, min_impurity_decrease
            # added max_leaf_nodes back in
            space = {
                'n_estimators': 300,
                'max_leaf_nodes': round(np.exp(rng.uniform(np.log(500), np.log(100_000)))),
                'max_depth': rng.choice([None, 12, 16]),
                'criterion': rng.choice(['entropy']) if is_classification
                else rng.choice(['squared_error', 'absolute_error']),
                'max_features': rng.choice(['sqrt', 'log2', None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(8.0)))),
                'min_samples_leaf': 1,
                'bootstrap': rng.choice([True, False]),
                'min_impurity_decrease': rng.choice([0.0, np.exp(rng.uniform(np.log(1e-3), np.log(5e-3)))]),
                'tfms': [['one_hot'], ['ordinal_encoding']][rng.integers(0, 1, endpoint=True)],
            }
        elif hpo_space_name == 'large-v3':
            # large-v2 but not tuning min_impurity_decrease, reduced max_depth, reduced min_samples_split,
            # only 100 estimators
            space = {
                'n_estimators': 100,
                'max_leaf_nodes': round(np.exp(rng.uniform(np.log(500), np.log(100_000)))),
                'max_depth': rng.choice([None, 16]),
                'criterion': rng.choice(['entropy']) if is_classification
                else rng.choice(['squared_error', 'absolute_error']),
                'max_features': rng.choice(['sqrt', 'log2', None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(4.5)))),
                'min_samples_leaf': 1,
                'bootstrap': rng.choice([True, False]),
                'tfms': [['one_hot'], ['ordinal_encoding']][rng.integers(0, 1, endpoint=True)],
            }
        elif hpo_space_name == 'large-v4':
            # large-v2 but only ordinal encoding
            space = {
                'n_estimators': 300,
                'max_leaf_nodes': round(np.exp(rng.uniform(np.log(500), np.log(100_000)))),
                'max_depth': rng.choice([None, 12, 16]),
                'criterion': rng.choice(['entropy']) if is_classification
                else rng.choice(['squared_error', 'absolute_error']),
                'max_features': rng.choice(['sqrt', 'log2', None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(8.0)))),
                'min_samples_leaf': 1,
                'bootstrap': rng.choice([True, False]),
                'min_impurity_decrease': rng.choice([0.0, np.exp(rng.uniform(np.log(1e-3), np.log(5e-3)))]),
                'tfms': ['ordinal_encoding'],
            }
        elif hpo_space_name == 'large-v5':
            # large-v3 but with 300 estimators and only ordinal encoding
            space = {
                'n_estimators': 300,
                'max_leaf_nodes': round(np.exp(rng.uniform(np.log(500), np.log(100_000)))),
                'max_depth': rng.choice([None, 16]),
                'criterion': rng.choice(['entropy']) if is_classification
                else rng.choice(['squared_error', 'absolute_error']),
                'max_features': rng.choice(['sqrt', 'log2', None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(4.5)))),
                'min_samples_leaf': 1,
                'bootstrap': rng.choice([True, False]),
                'tfms': ['ordinal_encoding'],
            }
        elif hpo_space_name == 'large-v6':
            # large-v4 but only bootstrap=True
            space = {
                'n_estimators': 300,
                'max_leaf_nodes': round(np.exp(rng.uniform(np.log(500), np.log(100_000)))),
                'max_depth': rng.choice([None, 12, 16]),
                'criterion': rng.choice(['entropy']) if is_classification
                else rng.choice(['squared_error', 'absolute_error']),
                'max_features': rng.choice(['sqrt', 'log2', None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(8.0)))),
                'min_samples_leaf': 1,
                'min_impurity_decrease': rng.choice([0.0, np.exp(rng.uniform(np.log(1e-3), np.log(5e-3)))]),
                'tfms': ['ordinal_encoding'],
            }
        elif hpo_space_name == 'large-v7':
            # large-v6 but not tuning max_leaf_nodes
            space = {
                'n_estimators': 300,
                'max_depth': rng.choice([None, 12, 16]),
                'criterion': rng.choice(['entropy']) if is_classification
                else rng.choice(['squared_error', 'absolute_error']),
                'max_features': rng.choice(['sqrt', 'log2', None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(8.0)))),
                'min_samples_leaf': 1,
                'min_impurity_decrease': rng.choice([0.0, np.exp(rng.uniform(np.log(1e-3), np.log(5e-3)))]),
                'tfms': ['ordinal_encoding'],
            }
        elif hpo_space_name == 'large-v8':
            # large-v4 but not tuning max_leaf_nodes, not allowing absolute_error
            space = {
                'n_estimators': 300,
                'max_depth': rng.choice([None, 12, 16]),
                'criterion': rng.choice(['entropy']) if is_classification
                else rng.choice(['squared_error']),
                'max_features': rng.choice(['sqrt', 'log2', None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(8.0)))),
                'min_samples_leaf': 1,
                'bootstrap': rng.choice([True, False]),
                'min_impurity_decrease': rng.choice([0.0, np.exp(rng.uniform(np.log(1e-3), np.log(5e-3)))]),
                'tfms': ['ordinal_encoding'],
            }
        elif hpo_space_name == 'large-v9':
            # large-v8 but tuning max_leaf_nodes again
            space = {
                'n_estimators': 300,
                'max_leaf_nodes': round(np.exp(rng.uniform(np.log(500), np.log(100_000)))),
                'max_depth': rng.choice([None, 12, 16]),
                'criterion': rng.choice(['entropy']) if is_classification
                else rng.choice(['squared_error']),
                'max_features': rng.choice(['sqrt', 'log2', None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(8.0)))),
                'min_samples_leaf': 1,
                'bootstrap': rng.choice([True, False]),
                'min_impurity_decrease': rng.choice([0.0, np.exp(rng.uniform(np.log(1e-3), np.log(5e-3)))]),
                'tfms': ['ordinal_encoding'],
            }
        elif hpo_space_name == 'large-v10':
            # large-v9 but not tuning min_impurity_decrease
            space = {
                'n_estimators': 300,
                'max_leaf_nodes': round(np.exp(rng.uniform(np.log(500), np.log(100_000)))),
                'max_depth': rng.choice([None, 12, 16]),
                'criterion': rng.choice(['entropy']) if is_classification
                else rng.choice(['squared_error']),
                'max_features': rng.choice(['sqrt', 'log2', None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(8.0)))),
                'min_samples_leaf': 1,
                'bootstrap': rng.choice([True, False]),
                'tfms': ['ordinal_encoding'],
            }
        elif hpo_space_name == 'large-v11':
            # large-v9 but tuning one-hot encoding
            space = {
                'n_estimators': 300,
                'max_leaf_nodes': round(np.exp(rng.uniform(np.log(500), np.log(100_000)))),
                'max_depth': rng.choice([None, 12, 16]),
                'criterion': rng.choice(['entropy']) if is_classification
                else rng.choice(['squared_error']),
                'max_features': rng.choice(['sqrt', 'log2', None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(8.0)))),
                'min_samples_leaf': 1,
                'min_impurity_decrease': rng.choice([0.0, np.exp(rng.uniform(np.log(1e-3), np.log(5e-3)))]),
                'bootstrap': rng.choice([True, False]),
                'tfms': [['one_hot'], ['ordinal_encoding']][rng.integers(0, 1, endpoint=True)],
            }
        elif hpo_space_name == 'large-v12':
            # very large space like large-v1 but a bit different
            # only 50 estimators -> use with bagging
            space = {
                'n_estimators': 50,
                'max_depth': rng.choice([6, 8, 12, 16, 20]),
                'criterion': 'entropy' if is_classification else 'squared_error',
                'max_features': rng.choice(['sqrt', 'log2', 0.2, 0.4, 0.6, 0.8, None]),
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(32.0)))),
                'min_samples_leaf': round(np.exp(rng.uniform(np.log(0.6), np.log(64.0)))),
                'bootstrap': rng.choice([True, False]),
                'min_impurity_decrease': rng.choice([0.0, np.exp(rng.uniform(np.log(1e-3), np.log(1e-1)))]),
                # 'max_samples': rng.uniform(0.4, 1.0), # this was accidentally not used
                'tfms': ['ordinal_encoding'],
            }
        elif hpo_space_name == 'large-v12':
            # very large space like large-v1 but a bit different
            # only 50 estimators -> use with bagging
            space = {
                'n_estimators': 50,
                'max_depth': rng.choice([6, 8, 12, 16, 20]),
                'criterion': 'entropy' if is_classification else 'squared_error',
                'max_features': rng.choice(['sqrt', 'log2', 0.2, 0.4, 0.6, 0.8, None]),
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(32.0)))),
                'min_samples_leaf': round(np.exp(rng.uniform(np.log(0.6), np.log(64.0)))),
                'bootstrap': rng.choice([True, False]),
                'min_impurity_decrease': rng.choice([0.0, np.exp(rng.uniform(np.log(1e-3), np.log(1e-1)))]),
                # 'max_samples': rng.uniform(0.4, 1.0), # this was accidentally not used
                'tfms': ['ordinal_encoding'],
            }
        elif hpo_space_name == 'large-v13':
            # reduced version on large-v12 based on talent-reg-small
            space = {
                'n_estimators': 50,
                'criterion': 'entropy' if is_classification else 'squared_error',
                'max_depth': rng.choice([16, 20]),
                'max_features': rng.choice([0.4, 0.6, 0.8, None]),
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(4.5)))),
                'min_samples_leaf': 1,
                'bootstrap': rng.choice([True, False]),
                'min_impurity_decrease': rng.choice([0.0, np.exp(rng.uniform(np.log(1e-3), np.log(1e-1)))]),
                # 'max_samples': rng.uniform(0.4, 1.0), # this was accidentally not used
                'tfms': ['ordinal_encoding'],
            }
        elif hpo_space_name == 'large-v14':
            # reduced version of large-v13 based on talent-reg-small
            # changed max_features, removed max_depth, changed min_impurity_decrease
            # removed tuning max_samples since it doesn't seem to do much?
            # this doesn't perform very well (target was not standardized for regression)
            space = {
                'n_estimators': 50,
                'criterion': 'entropy' if is_classification else 'squared_error',
                'max_features': rng.uniform(0.2, 0.9),
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(4.5)))),
                'min_samples_leaf': 1,
                'bootstrap': rng.choice([True, False]),
                'min_impurity_decrease': np.exp(rng.uniform(np.log(1e-5), np.log(1e-2))),
                'tfms': ['ordinal_encoding'],
            }
        elif hpo_space_name == 'large-v15':
            # large-v14 but with standardized target
            # better
            space = {
                'n_estimators': 50,
                'criterion': 'entropy' if is_classification else 'squared_error',
                'max_features': rng.uniform(0.2, 0.9),
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(4.5)))),
                'min_samples_leaf': 1,
                'bootstrap': rng.choice([True, False]),
                'min_impurity_decrease': np.exp(rng.uniform(np.log(1e-5), np.log(1e-2))),
                'tfms': ['ordinal_encoding'],
                'standardize_target': True,
            }
        elif hpo_space_name == 'large-v16':
            # large-v15 but don't tune min_impurity_decrease. Also go back to old max_features
            space = {
                'n_estimators': 50,
                'max_features': rng.choice([0.4, 0.6, 0.8, None]),
                'criterion': 'entropy' if is_classification else 'squared_error',
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(4.5)))),
                'min_samples_leaf': 1,
                'bootstrap': rng.choice([True, False]),
                'tfms': ['ordinal_encoding'],
                'standardize_target': True,
            }
        elif hpo_space_name == 'large-v17':
            # large-v16 but with tuning max_samples (wasn't used)
            space = {
                'n_estimators': 50,
                'max_features': rng.choice([0.4, 0.6, 0.8, None]),
                'criterion': 'entropy' if is_classification else 'squared_error',
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(4.5)))),
                'min_samples_leaf': 1,
                'bootstrap': rng.choice([True, False]),
                # 'max_samples': rng.uniform(0.4, 1.0), # this was accidentally not used
                'tfms': ['ordinal_encoding'],
                'standardize_target': True,
            }
        elif hpo_space_name == 'large-v18':
            # large-v16 but with max_depth limit  (equivalent to large-v13 without tuning min_impurity_decrease)
            space = {
                'n_estimators': 50,
                'max_features': rng.choice([0.4, 0.6, 0.8, None]),
                'max_depth': rng.choice([16, 20]),
                'criterion': 'entropy' if is_classification else 'squared_error',
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(4.5)))),
                'min_samples_leaf': 1,
                'bootstrap': rng.choice([True, False]),
                # 'max_samples': rng.uniform(0.4, 1.0), # this was accidentally not used
                'tfms': ['ordinal_encoding'],
                'standardize_target': True,
            }
        elif hpo_space_name == 'large-v19':
            # large-v18 but with tuning max_samples
            space = {
                'n_estimators': 50,
                'max_features': rng.choice([0.4, 0.6, 0.8, None]),
                'max_depth': rng.choice([16, 20]),
                'criterion': 'entropy' if is_classification else 'squared_error',
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(4.5)))),
                'min_samples_leaf': 1,
                'bootstrap': rng.choice([True, False]),
                'max_samples': rng.uniform(0.4, 1.0),
                'tfms': ['ordinal_encoding'],
                'standardize_target': True,
            }
        elif hpo_space_name == 'large-v20':
            # large-v19 but with tuning min_impurity_decrease, with 300 estimator, a few more max_depth options
            space = {
                'n_estimators': 300,
                'max_features': rng.choice([0.4, 0.6, 0.8, None]),
                'max_depth': rng.choice([12, 16, 20, None]),
                'criterion': 'entropy' if is_classification else 'squared_error',
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(4.5)))),
                'min_samples_leaf': 1,
                'bootstrap': rng.choice([True, False]),
                'max_samples': rng.uniform(0.4, 1.0),
                'min_impurity_decrease': rng.choice([0.0, np.exp(rng.uniform(np.log(1e-4), np.log(5e-3)))]),
                'tfms': ['ordinal_encoding'],
                'standardize_target': True,
            }
        elif hpo_space_name == 'large-v21':
            # large-v20 but with different max_depth, min_impurity_decrease, and 50 estimators
            space = {
                'n_estimators': 50,
                'max_features': rng.choice([0.4, 0.6, 0.8, None]),
                'max_depth': rng.choice([16, 20, None]),
                'criterion': 'entropy' if is_classification else 'squared_error',
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(4.5)))),
                'min_samples_leaf': 1,
                'bootstrap': rng.choice([True, False]),
                'max_samples': rng.uniform(0.4, 1.0),
                'min_impurity_decrease': rng.choice([0.0, np.exp(rng.uniform(np.log(1e-4), np.log(1e-3)))]),
                'tfms': ['ordinal_encoding'],
                'standardize_target': True,
            }
        elif hpo_space_name == 'large-v22':
            # large-v21 but without bootstrap=False
            space = {
                'n_estimators': 50,
                'max_features': rng.choice([0.4, 0.6, 0.8, None]),
                'max_depth': rng.choice([16, 20, None]),
                'criterion': 'entropy' if is_classification else 'squared_error',
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(4.5)))),
                'min_samples_leaf': 1,
                'max_samples': rng.uniform(0.4, 1.0),
                'min_impurity_decrease': rng.choice([0.0, np.exp(rng.uniform(np.log(1e-4), np.log(1e-3)))]),
                'tfms': ['ordinal_encoding'],
                'standardize_target': True,
            }
        elif hpo_space_name == 'large-v23':
            # large-v21 but with 100 estimators
            space = {
                'n_estimators': 100,
                'max_features': rng.choice([0.4, 0.6, 0.8, None]),
                'max_depth': rng.choice([16, 20, None]),
                'criterion': 'entropy' if is_classification else 'squared_error',
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(4.5)))),
                'min_samples_leaf': 1,
                'bootstrap': rng.choice([True, False]),
                'max_samples': rng.uniform(0.4, 1.0),
                'min_impurity_decrease': rng.choice([0.0, np.exp(rng.uniform(np.log(1e-4), np.log(1e-3)))]),
                'tfms': ['ordinal_encoding'],
                'standardize_target': True,
            }
        elif hpo_space_name == 'large-v24':
            # large-v21 but without tuning min_impurity_decrease
            space = {
                'n_estimators': 50,
                'max_features': rng.choice([0.4, 0.6, 0.8, None]),
                'max_depth': rng.choice([16, 20, None]),
                'criterion': 'entropy' if is_classification else 'squared_error',
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(4.5)))),
                'min_samples_leaf': 1,
                'bootstrap': rng.choice([True, False]),
                'max_samples': rng.uniform(0.4, 1.0),
                'tfms': ['ordinal_encoding'],
                'standardize_target': True,
            }
        elif hpo_space_name == 'large-v25':
            # large-v21 but with different min_impurity_decrease space
            space = {
                'n_estimators': 50,
                'max_features': rng.choice([0.4, 0.6, 0.8, None]),
                'max_depth': rng.choice([16, 20, None]),
                'criterion': 'entropy' if is_classification else 'squared_error',
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(4.5)))),
                'min_samples_leaf': 1,
                'bootstrap': rng.choice([True, False]),
                'max_samples': rng.uniform(0.4, 1.0),
                'min_impurity_decrease': np.exp(rng.uniform(np.log(1e-5), np.log(1e-3))),
                'tfms': ['ordinal_encoding'],
                'standardize_target': True,
            }
        elif hpo_space_name == 'large-v26':
            # large-v25 but with tuning min_samples_leaf
            space = {
                'n_estimators': 50,
                'max_features': rng.choice([0.4, 0.6, 0.8, None]),
                'max_depth': rng.choice([16, 20, None]),
                'criterion': 'entropy' if is_classification else 'squared_error',
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(4.5)))),
                'min_samples_leaf': round(np.exp(rng.uniform(np.log(1.5), np.log(4.5)))),
                'bootstrap': rng.choice([True, False]),
                'max_samples': rng.uniform(0.4, 1.0),
                'min_impurity_decrease': np.exp(rng.uniform(np.log(1e-5), np.log(1e-3))),
                'tfms': ['ordinal_encoding'],
                'standardize_target': True,
            }
        elif hpo_space_name == 'large-v27':
            # inspired from XT but with both bootstrap options
            space = {
                'n_estimators': 50,
                'max_features': ['sqrt', 0.5, 0.75, 1.0][rng.integers(4)],
                'criterion': 'entropy' if is_classification else 'squared_error',
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(16.0)))),
                'bootstrap': rng.choice([True, False]),
                'max_samples': rng.uniform(0.4, 1.0),
                'min_impurity_decrease': rng.choice([0.0, np.exp(rng.uniform(np.log(1e-5), np.log(1e-3)))]),
                'tfms': ['ordinal_encoding'],
                'standardize_target': True,
            }
        elif hpo_space_name == 'tabrepo1':
            space = {
                'n_estimators': 300,
                'max_leaf_nodes': rng.integers(5000, 50000, endpoint=True),
                'min_samples_leaf': rng.choice([1, 2, 3, 4, 5, 10, 20, 40, 80]),
                'max_features': ['sqrt', 'log2', 0.5, 0.75, 1.0][rng.integers(5)],
                'tfms': ['one_hot'],
            }
        elif hpo_space_name == 'tabrepo1-ordinal':
            space = {
                'n_estimators': 300,
                'max_leaf_nodes': rng.integers(5000, 50000, endpoint=True),
                'min_samples_leaf': rng.choice([1, 2, 3, 4, 5, 10, 20, 40, 80]),
                'max_features': ['sqrt', 'log2', 0.5, 0.75, 1.0][rng.integers(5)],
                'tfms': ['ordinal_encoding'],  # failed to fix it
            }
        else:
            raise ValueError()
        return space

    def _create_interface_from_config(self, n_tv_splits: int, **config):
        return SingleSplitWrapperAlgInterface([RFSubSplitInterface(**config) for i in range(n_tv_splits)])


class ExtraTreesSubSplitInterface(SklearnSubSplitInterface):
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
                         ('max_leaf_nodes', None),
                         ('max_samples', None),
                         ('n_jobs', ['n_jobs', 'n_threads'], n_threads),
                         ('verbose', ['verbose', 'verbosity'])]

        params = utils.extract_params(self.config, params_config)
        if not params.get('bootstrap', True) and 'max_samples' in params:
            del params['max_samples']
        if self.n_classes > 0:
            return ExtraTreesClassifier(random_state=seed, **params)
        else:
            train_metric_name = self.config.get('train_metric_name', None)
            if train_metric_name == 'mse':
                params['criterion'] = 'squared_error'  # is the default anyway
            elif train_metric_name == 'mae':
                params['criterion'] = 'absolute_error'
            elif train_metric_name is not None:
                raise ValueError(f'Train metric "{train_metric_name}" is currently not supported!')
            reg = ExtraTreesRegressor(random_state=seed, **params)
            if self.config.get('standardize_target', False):
                reg = TransformedTargetRegressor(reg, transformer=StandardScaler())
            return reg

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


class RandomParamsExtraTreesAlgInterface(RandomParamsAlgInterface):
    def _sample_params(self, is_classification: bool, seed: int, n_train: int):
        rng = np.random.default_rng(seed)
        hpo_space_name = self.config['hpo_space_name']
        if hpo_space_name == 'large-v1':
            space = {
                'n_estimators': 50,
                'max_leaf_nodes': round(np.exp(rng.uniform(np.log(500), np.log(100_000)))),
                'max_depth': rng.choice([None, 8, 12, 16]),
                'criterion': rng.choice(['gini', 'entropy']) if is_classification
                else 'squared_error',
                'max_features': rng.choice(['sqrt', 'log2', None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(16.0)))),
                'min_samples_leaf': round(np.exp(rng.uniform(np.log(0.6), np.log(8.0)))),
                'max_samples': float(rng.uniform(0.4, 1.0)),
                'bootstrap': rng.choice([True, False]),
                'min_impurity_decrease': np.exp(rng.uniform(np.log(1e-5), np.log(1e-2))),
                'tfms': ['ordinal_encoding'],
                'standardize_target': True,
            }
        elif hpo_space_name == 'large-v2':
            # large-v1 shrunken
            space = {
                'n_estimators': 50,
                'max_features': rng.choice([0.4, 0.6, 0.8, None]),
                'criterion': 'entropy' if is_classification else 'squared_error',
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(16.0)))),
                'min_samples_leaf': round(np.exp(rng.uniform(np.log(0.6), np.log(4.5)))),
                'bootstrap': rng.choice([True, False]),
                'max_samples': rng.uniform(0.4, 1.0),
                'min_impurity_decrease': np.exp(rng.uniform(np.log(1e-5), np.log(1e-3))),
                'tfms': ['ordinal_encoding'],
                'standardize_target': True,
            }
        elif hpo_space_name == 'large-v3':
            # large-v2 shrunken
            # very good for classification
            # tuning of max_features may be unnecessary, default might work just as well
            # maybe could go even larger with min_samples_split
            space = {
                'n_estimators': 50,
                'max_features': rng.choice([0.4, 0.6, 0.8, None]),
                'criterion': 'entropy' if is_classification else 'squared_error',
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(16.0)))),
                'min_samples_leaf': 1,
                'bootstrap': False,
                # 'max_samples': rng.uniform(0.4, 1.0),  # irrelevant without bootstrap
                'min_impurity_decrease': np.exp(rng.uniform(np.log(1e-5), np.log(1e-3))),
                # could decrease upper bound to 5e-4
                'tfms': ['ordinal_encoding'],
                'standardize_target': True,
            }
        elif hpo_space_name == 'large-v4':
            # large space for regression tests
            space = {
                'n_estimators': 50,
                'max_leaf_nodes': round(np.exp(rng.uniform(np.log(500), np.log(100_000)))),
                'criterion': 'entropy' if is_classification else 'squared_error',
                'max_features': rng.choice([0.4, 0.6, 0.8, None]),
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(32.0)))),
                'min_samples_leaf': 1,
                'max_samples': float(rng.uniform(0.4, 1.0)),
                'bootstrap': rng.choice([True, False]),
                'min_impurity_decrease': np.exp(rng.uniform(np.log(1e-5), np.log(1e-2))),
                'tfms': ['ordinal_encoding'],
                'standardize_target': True,
            }
        elif hpo_space_name == 'large-v5':
            # shrunken version of large-v4 for regression
            # min_impurity_decrease could be shrunk more
            space = {
                'n_estimators': 50,
                'criterion': 'entropy' if is_classification else 'squared_error',
                'max_features': float(rng.uniform(0.5, 1.0)),
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(32.0)))),
                'min_samples_leaf': 1,
                # 'max_samples': float(rng.uniform(0.4, 1.0)),    # irrelevant without bootstrap
                'bootstrap': False,
                'min_impurity_decrease': np.exp(rng.uniform(np.log(1e-6), np.log(5e-4))),
                'tfms': ['ordinal_encoding'],
                'standardize_target': True,
            }
        elif hpo_space_name == 'large-v6':
            # large-v5 without tuning min_impurity_decrease
            space = {
                'n_estimators': 50,
                'criterion': 'entropy' if is_classification else 'squared_error',
                'max_features': float(rng.uniform(0.5, 1.0)),
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(32.0)))),
                'min_samples_leaf': 1,
                # 'max_samples': float(rng.uniform(0.4, 1.0)),    # irrelevant without bootstrap
                'bootstrap': False,
                'tfms': ['ordinal_encoding'],
                'standardize_target': True,
            }
        elif hpo_space_name == 'large-v7':
            # large-v6 with tuning max_leaf_nodes
            # doesn't help
            space = {
                'n_estimators': 50,
                'criterion': 'entropy' if is_classification else 'squared_error',
                'max_leaf_nodes': rng.integers(5000, 50000, endpoint=True),
                'max_features': float(rng.uniform(0.5, 1.0)),
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(32.0)))),
                'min_samples_leaf': 1,
                # 'max_samples': float(rng.uniform(0.4, 1.0)),    # irrelevant without bootstrap
                'bootstrap': False,
                'tfms': ['ordinal_encoding'],
                'standardize_target': True,
            }
        elif hpo_space_name == 'large-v8':
            # large-v6 but with different tuning space for max_features
            space = {
                'n_estimators': 50,
                'criterion': 'entropy' if is_classification else 'squared_error',
                'max_features': ['sqrt', 'log2', 0.5, 0.75, 1.0][rng.integers(5)],
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(32.0)))),
                'min_samples_leaf': 1,
                'bootstrap': False,
                'tfms': ['ordinal_encoding'],
                'standardize_target': True,
            }
        elif hpo_space_name == 'large-v9':
            # large-v8 but tuning min_samples_leaf
            space = {
                'n_estimators': 50,
                'criterion': 'entropy' if is_classification else 'squared_error',
                'max_features': ['sqrt', 'log2', 0.5, 0.75, 1.0][rng.integers(5)],
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(32.0)))),
                'min_samples_leaf': round(np.exp(rng.uniform(np.log(1.5), np.log(8.5)))),
                'bootstrap': False,
                'tfms': ['ordinal_encoding'],
                'standardize_target': True,
            }
        elif hpo_space_name == 'large-v10':
            # large-v9 but without tuning min_samples_split
            space = {
                'n_estimators': 50,
                'criterion': 'entropy' if is_classification else 'squared_error',
                'max_features': ['sqrt', 'log2', 0.5, 0.75, 1.0][rng.integers(5)],
                'min_samples_split': 2,
                'min_samples_leaf': round(np.exp(rng.uniform(np.log(1.5), np.log(8.5)))),
                'bootstrap': False,
                'tfms': ['ordinal_encoding'],
                'standardize_target': True,
            }
        elif hpo_space_name == 'large-v11':
            # large-v10 but with fixed tuning space for min_samples_leaf
            space = {
                'n_estimators': 50,
                'criterion': 'entropy' if is_classification else 'squared_error',
                'max_features': ['sqrt', 'log2', 0.5, 0.75, 1.0][rng.integers(5)],
                'min_samples_split': 2,
                'min_samples_leaf': round(np.exp(rng.uniform(np.log(0.5), np.log(8.5)))),
                'bootstrap': False,
                'tfms': ['ordinal_encoding'],
                'standardize_target': True,
            }
        elif hpo_space_name == 'large-v12':
            # large-v9 but with fixed tuning space for min_samples_leaf
            space = {
                'n_estimators': 50,
                'criterion': 'entropy' if is_classification else 'squared_error',
                'max_features': ['sqrt', 'log2', 0.5, 0.75, 1.0][rng.integers(5)],
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(32.0)))),
                'min_samples_leaf': round(np.exp(rng.uniform(np.log(0.5), np.log(8.5)))),
                'bootstrap': False,
                'tfms': ['ordinal_encoding'],
                'standardize_target': True,
            }
        elif hpo_space_name == 'large-v13':
            # large-v3 with different max_features space
            space = {
                'n_estimators': 50,
                'max_features': ['sqrt', 'log2', 0.5, 0.75, 1.0][rng.integers(5)],
                'criterion': 'entropy' if is_classification else 'squared_error',
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(16.0)))),
                'min_samples_leaf': 1,
                'bootstrap': False,
                # 'max_samples': rng.uniform(0.4, 1.0),  # irrelevant without bootstrap
                'min_impurity_decrease': np.exp(rng.uniform(np.log(1e-5), np.log(1e-3))),
                # could decrease upper bound to 5e-4
                'tfms': ['ordinal_encoding'],
                'standardize_target': True,
            }
        elif hpo_space_name == 'large-v14':
            # large-v3 with different max_features space
            space = {
                'n_estimators': 50,
                'max_features': ['sqrt', 0.5, 0.75, 1.0][rng.integers(4)],
                'criterion': 'entropy' if is_classification else 'squared_error',
                'min_samples_split': round(np.exp(rng.uniform(np.log(1.5), np.log(16.0)))),
                'min_samples_leaf': 1,
                'bootstrap': False,
                # 'max_samples': rng.uniform(0.4, 1.0),  # irrelevant without bootstrap
                'min_impurity_decrease': np.exp(rng.uniform(np.log(1e-5), np.log(1e-3))),
                # could decrease upper bound to 5e-4
                'tfms': ['ordinal_encoding'],
                'standardize_target': True,
            }
        elif hpo_space_name == 'tabrepo1-mod':
            space = {
                'n_estimators': 50,
                # not completely sure if tabrepo1 uses entropy
                'criterion': 'entropy' if is_classification else 'squared_error',
                'max_leaf_nodes': rng.integers(5000, 50000, endpoint=True),
                'min_samples_leaf': rng.choice([1, 2, 3, 4, 5, 10, 20, 40, 80]),
                'max_features': ['sqrt', 'log2', 0.5, 0.75, 1.0][rng.integers(5)],
                'tfms': ['ordinal_encoding'],
            }
        else:
            raise ValueError()
        return space

    def _create_interface_from_config(self, n_tv_splits: int, **config):
        return SingleSplitWrapperAlgInterface([ExtraTreesSubSplitInterface(**config) for i in range(n_tv_splits)])


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


class KNNSubSplitInterface(SklearnSubSplitInterface):
    def _create_sklearn_model(self, seed: int, n_threads: int, gpu_devices: List[str]) -> Any:
        params_config = [('n_neighbors', None),
                         ('weights', None),
                         ('p', None),
                         ('n_jobs', ['n_jobs', 'n_threads'], n_threads)]

        params = utils.extract_params(self.config, params_config)
        if self.n_classes > 0:
            from sklearn.neighbors import KNeighborsClassifier
            return KNeighborsClassifier(**params)
        else:
            from sklearn.neighbors import KNeighborsRegressor
            return KNeighborsRegressor(**params)

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


class RandomParamsKNNAlgInterface(RandomParamsAlgInterface):
    def _sample_params(self, is_classification: bool, seed: int, n_train: int):
        rng = np.random.default_rng(seed)
        hpo_space_name = self.config['hpo_space_name']
        if hpo_space_name == 'v1':
            space = {
                'n_neighbors': int(np.exp(rng.uniform(np.log(1.0), np.log(101.0)))),
                'weights': rng.choice(['uniform', 'distance']),
                # 'p': np.exp(rng.uniform(np.log(0.2), np.log(8.0))),  # values outside of 1 and 2 can be very slow
                'p': rng.choice([1, 2]),
                'tfms': ['mean_center', 'l2_normalize', 'one_hot'],
            }
        elif hpo_space_name == 'tabrepo1':
            space = {
                'n_neighbors': rng.choice([3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 20, 30, 40, 50]),
                'weights': rng.choice(['uniform', 'distance']),
                'p': rng.choice([1, 2]),
                'tfms': ['mean_center', 'l2_normalize', 'one_hot'],
            }
        else:
            raise ValueError()
        return space

    def _create_interface_from_config(self, n_tv_splits: int, **config):
        return SingleSplitWrapperAlgInterface([KNNSubSplitInterface(**config) for i in range(n_tv_splits)])


class LinearModelSubSplitInterface(SklearnSubSplitInterface):
    def _create_sklearn_model(self, seed: int, n_threads: int, gpu_devices: List[str]) -> Any:
        params_config = [
            # ('l1_ratio', None),
            ('fit_intercept', None),
            # ('n_jobs', ['n_jobs', 'n_threads'], n_threads)
        ]

        penalty = self.config.get('penalty', 'l2')
        n_jobs = self.config.get('n_jobs', self.config.get('n_threads', None))

        params = utils.extract_params(self.config, params_config)
        l1_ratio = self.config.get('l1_ratio', 0.5)

        C = self.config.get('C', 1.0)
        if self.n_classes > 0:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(random_state=seed, penalty=penalty,
                                      solver='lbfgs' if penalty == 'l2' else 'saga',
                                      C=C, l1_ratio=l1_ratio if penalty == 'elasticnet' else None,
                                      n_jobs=n_jobs, **params)
            # return LogisticRegression(random_state=seed, penalty='l2', solver='newton-cholesky', C=C, **params)
        else:
            alpha = self.config.get('alpha', 1 / C)
            from sklearn.linear_model import Ridge, Lasso, ElasticNet
            if penalty == 'l2':
                return Ridge(random_state=seed, alpha=alpha, **params)
            elif penalty == 'l1':
                return Lasso(random_state=seed, alpha=alpha, **params)
            elif penalty == 'elasticnet':
                return ElasticNet(random_state=seed, alpha=alpha, l1_ratio=l1_ratio, **params)
            else:
                raise ValueError()
            # from sklearn.linear_model import ElasticNet
            # return ElasticNet(random_state=seed, alpha=alpha, **params)

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        assert n_cv == 1
        assert n_refit == 0
        assert n_splits == 1
        updated_config = utils.join_dicts(dict(n_estimators=100, n_threads=1), self.config)
        time_params = {'': 0.5, 'ds_size_gb': 10.0}
        ram_params = {'': 0.5, 'ds_size_gb': 3.0}
        rc = ResourcePredictor(config=updated_config, time_params=time_params,
                               cpu_ram_params=ram_params)
        return rc.get_required_resources(ds)


class RandomParamsLinearModelAlgInterface(RandomParamsAlgInterface):
    def _sample_params(self, is_classification: bool, seed: int, n_train: int):
        rng = np.random.default_rng(seed)
        hpo_space_name = self.config['hpo_space_name']
        if hpo_space_name == 'v1':
            space = {
                'penalty': rng.choice(['l1', 'l2', 'elasticnet']),
                'l1_ratio': rng.uniform(0.01, 1.0),
                'C': np.exp(rng.uniform(np.log(1e-2), np.log(1e7))),
                'tfms': ['mean_center', 'l2_normalize', 'one_hot'],
            }
        elif hpo_space_name == 'v2':
            # smaller version of v1
            space = {
                'penalty': rng.choice(['l1', 'l2', 'elasticnet']),
                'l1_ratio': rng.uniform(0.01, 0.8),
                'C': np.exp(rng.uniform(np.log(1e-1), np.log(1e5))),
                'tfms': ['mean_center', 'l2_normalize', 'one_hot'],
            }
        elif hpo_space_name == 'v3':
            # smaller version of v1
            space = {
                'penalty': rng.choice(['l1', 'l2', 'elasticnet']),
                'l1_ratio': rng.uniform(0.01, 0.5),
                'C': np.exp(rng.uniform(np.log(1e-1), np.log(1e4))),
                'tfms': ['mean_center', 'l2_normalize', 'one_hot'],
            }
        elif hpo_space_name == 'v4':
            # smaller version of v1
            space = {
                'penalty': rng.choice(['l1', 'l2']),
                'C': np.exp(rng.uniform(np.log(1e-1), np.log(1e5))),
                'tfms': ['mean_center', 'l2_normalize', 'one_hot'],
            }
        elif hpo_space_name == 'tabrepo1':
            space = {
                'penalty': rng.choice(['l1', 'l2']),
                'C': np.exp(rng.uniform(np.log(1e-1), np.log(1e3))),
                'tfms': ['mean_center', 'l2_normalize', 'one_hot'],
            }
        elif hpo_space_name == 'tabrepo1-rssc3':
            space = {
                'penalty': rng.choice(['l1', 'l2']),
                'C': np.exp(rng.uniform(np.log(1e-1), np.log(1e3))),
                'tfms': ['median_center', 'robust_scale', 'smooth_clip', 'one_hot'],
                'smooth_clip_max_abs_value': 3,
            }
        elif hpo_space_name == 'tabrepo1-rssc5':
            space = {
                'penalty': rng.choice(['l1', 'l2']),
                'C': np.exp(rng.uniform(np.log(1e-1), np.log(1e3))),
                'tfms': ['median_center', 'robust_scale', 'smooth_clip', 'one_hot'],
                'smooth_clip_max_abs_value': 5,
            }
        elif hpo_space_name == 'tabrepo1-rssc10':
            space = {
                'penalty': rng.choice(['l1', 'l2']),
                'C': np.exp(rng.uniform(np.log(1e-1), np.log(1e3))),
                'tfms': ['median_center', 'robust_scale', 'smooth_clip', 'one_hot'],
                'smooth_clip_max_abs_value': 10,
            }
        else:
            raise ValueError()
        return space

    def _create_interface_from_config(self, n_tv_splits: int, **config):
        return SingleSplitWrapperAlgInterface([LinearModelSubSplitInterface(**config) for i in range(n_tv_splits)])


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
            'learning_rate_leaf': 0.01,  # learning rate for leaves (logits)

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


class TabPFN2SubSplitInterface(SklearnSubSplitInterface):
    def _create_sklearn_model(self, seed: int, n_threads: int, gpu_devices: List[str]) -> Any:
        params_config = [
            # ('n_jobs', ['n_jobs', 'n_threads'], n_threads),
            ('softmax_temperature', None),
            ('average_before_softmax', None),
            ('inference_precision', None),
            ('fit_mode', None),
            ('model_path', None),
        ]

        params = utils.extract_params(self.config, params_config)
        if self.config.get('use_float32', False):
            params['inference_precision'] = torch.float32
        # print(f'{gpu_devices=}')
        if self.n_classes > 0:
            from tabpfn import TabPFNClassifier
            return TabPFNClassifier(random_state=seed,
                                    device=gpu_devices[0] if len(gpu_devices) > 0 else 'cpu',
                                    # device='cuda' if len(gpu_devices) > 0 else 'cpu',
                                    ignore_pretraining_limits=True, **params)
        else:
            from tabpfn import TabPFNRegressor
            return TabPFNRegressor(random_state=seed,
                                   device=gpu_devices[0] if len(gpu_devices) > 0 else 'cpu',
                                   # device='cuda' if len(gpu_devices) > 0 else 'cpu',
                                   ignore_pretraining_limits=True, **params)

    def _fit_sklearn(self, x_df: pd.DataFrame, y: np.ndarray, val_idxs: np.ndarray,
                     cat_col_names: Optional[List[str]] = None):
        # by default, we ignore the validation set since most sklearn methods do not support it
        if not self.config.get('fit_on_valid', False):
            n_samples = len(x_df)
            train_mask = np.ones(shape=(n_samples,), dtype=np.bool_)
            train_mask[val_idxs] = False
            x_df = x_df.iloc[train_mask, :]
            y = y[train_mask]
        # don't provide a categorical indicator, it should work like this as well
        self.model.fit(x_df, y)

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        assert n_cv == 1
        assert n_refit == 0
        assert n_splits == 1
        updated_config = utils.join_dicts(dict(n_estimators=100), self.config)
        time_params = {'': 0.5, 'ds_size_gb': 10.0, '1/n_threads*n_samples*n_estimators*n_tree_repeats': 4e-8}
        ram_params = {'': 0.5, 'ds_size_gb': 3.0, 'n_samples*n_estimators*n_tree_repeats': 3e-9}
        rc = ResourcePredictor(config=updated_config, time_params=time_params,
                               cpu_ram_params=ram_params, n_gpus=1, gpu_usage=1.0, gpu_ram_params={'': 10.0})
        return rc.get_required_resources(ds)


class TabICLSubSplitInterface(SklearnSubSplitInterface):
    def _create_sklearn_model(self, seed: int, n_threads: int, gpu_devices: List[str]) -> Any:
        params_config = [
            # ('n_jobs', ['n_jobs', 'n_threads'], n_threads),
            ('n_estimators', None),
            ('softmax_temperature', None),
            ('average_logits', None),
            ('use_amp', None),
            ('batch_size', None),
            ('model_path', None),
            ('allow_auto_download', None),
            ('norm_methods', None)
        ]

        params = utils.extract_params(self.config, params_config)
        if self.config.get('use_float32', False):
            params['inference_precision'] = torch.float32
        # print(f'{gpu_devices=}')
        if self.n_classes > 0:
            if self.config.get('use_tabiclex', False):
                from tabiclex import TabICLClassifier
            else:
                from tabicl import TabICLClassifier
            return TabICLClassifier(random_state=seed,
                                    device=gpu_devices[0] if len(gpu_devices) > 0 else 'cpu',
                                    **params)
        else:
            raise ValueError(f'TabICL for regression does not exist')

    def _fit_sklearn(self, x_df: pd.DataFrame, y: np.ndarray, val_idxs: np.ndarray,
                     cat_col_names: Optional[List[str]] = None):
        # by default, we ignore the validation set since most sklearn methods do not support it
        if not self.config.get('fit_on_valid', False):
            n_samples = len(x_df)
            train_mask = np.ones(shape=(n_samples,), dtype=np.bool_)
            train_mask[val_idxs] = False
            x_df = x_df.iloc[train_mask, :]
            y = y[train_mask]
        x_df = x_df.copy()
        if self.config.get('add_fingerprint_feature', False):
            x_df['__fingerprint_feature'] = np.random.randn(len(x_df))
        if self.config.get('mirror_numerical_features', False):
            self.float_cols_ = x_df.select_dtypes(include=['float']).columns
            print(f'{len(self.float_cols_)=}')
            # Generate random signs (+1 or -1) for each column
            self.signs_ = np.random.choice([-1, 1], size=len(self.float_cols_))
            # Multiply each float column by its random sign
            x_df.loc[:, self.float_cols_] = x_df.loc[:, self.float_cols_] * self.signs_
        # don't provide a categorical indicator, it should work like this as well
        self.model.fit(x_df, y)

    def _predict_sklearn(self, x_df: pd.DataFrame) -> np.ndarray:
        x_df = x_df.copy()
        if self.config.get('add_fingerprint_feature', False):
            x_df['__fingerprint_feature'] = np.random.randn(len(x_df))
        if self.config.get('mirror_numerical_features', False):
            x_df.loc[:, self.float_cols_] = x_df.loc[:, self.float_cols_] * self.signs_
        return super()._predict_sklearn(x_df)

    def _predict_proba_sklearn(self, x_df: pd.DataFrame) -> np.ndarray:
        x_df = x_df.copy()
        if self.config.get('add_fingerprint_feature', False):
            x_df['__fingerprint_feature'] = np.random.randn(len(x_df))
        if self.config.get('mirror_numerical_features', False):
            x_df.loc[:, self.float_cols_] = x_df.loc[:, self.float_cols_] * self.signs_
        return super()._predict_proba_sklearn(x_df)

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        assert n_cv == 1
        assert n_refit == 0
        assert n_splits == 1
        updated_config = utils.join_dicts(dict(n_estimators=100), self.config)
        time_params = {'': 0.5, 'ds_size_gb': 10.0, '1/n_threads*n_samples*n_estimators*n_tree_repeats': 4e-8}
        ram_params = {'': 0.5, 'ds_size_gb': 3.0, 'n_samples*n_estimators*n_tree_repeats': 3e-9}
        rc = ResourcePredictor(config=updated_config, time_params=time_params,
                               cpu_ram_params=ram_params, n_gpus=1, gpu_usage=1.0, gpu_ram_params={'': 10.0})
        return rc.get_required_resources(ds)


