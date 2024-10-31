import copy
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import torch

from pytabkit.models.alg_interfaces.resource_computation import ResourcePredictor
from pytabkit.models.alg_interfaces.resource_params import ResourceParams
from pytabkit.models import utils
from pytabkit.models.alg_interfaces.base import RequiredResources
from pytabkit.models.alg_interfaces.sub_split_interfaces import TreeBasedSubSplitInterface, SingleSplitWrapperAlgInterface, \
    SklearnSubSplitInterface
from pytabkit.models.data.data import DictDataset
from pytabkit.models.hyper_opt.hyper_optimizers import HyperoptOptimizer
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor


from pytabkit.models.alg_interfaces.alg_interfaces import OptAlgInterface, AlgInterface, RandomParamsAlgInterface
from pytabkit.models.training.metrics import Metrics


class XGBCustomMetric:
    def __init__(self, metric_name: str, is_classification: bool, is_higher_better: bool = False):
        self.metric_name = metric_name
        self.is_classification = is_classification
        self.is_higher_better = is_higher_better

    def __call__(self, y_pred: np.ndarray, dtrain: xgb.DMatrix):
        y = torch.as_tensor(dtrain.get_label(), dtype=torch.long if self.is_classification else torch.float32)
        if len(y.shape) == 1:
            y = y[:, None]

        # print(f'{y_pred.shape=}, {eval_data.get_label().shape=}')
        y_pred = torch.as_tensor(y_pred, dtype=torch.float32)
        if len(y_pred.shape) == 1:
            if self.is_classification:
                if y_pred.shape[0] == y.shape[0]:
                    # binary classification, transform into both class probabilities
                    y_pred = torch.stack([1. - y_pred, y_pred], dim=-1)
                else:
                    # bugged multiclass classification in LightGBM, need to reshape
                    # print(y_pred[:7])
                    y_pred = y_pred.view(-1, y.shape[0]).t().contiguous()
                    # print(y_pred[0, :].sum())
            else:
                y_pred = y_pred[:, None]

        if self.is_classification:
            # go from probabilities to logits
            y_pred = torch.log(y_pred + 1e-30)

        # print(f'{y_pred.shape=}, {y.shape=}')

        # print(f'{y_pred=}, {y=}')

        eval_result = Metrics.apply(y_pred, y, metric_name=self.metric_name)

        # print(f'loss: {eval_result.item():g}')

        return self.metric_name, eval_result.item()


class XGBSklearnSubSplitInterface(SklearnSubSplitInterface):
    def _create_sklearn_model(self, seed: int, n_threads: int, gpu_devices: List[str]) -> Any:
        params_config = [('n_estimators', None),
                         ('verbosity', None),
                         ('max_depth', None),
                         ('eta', ['lr', 'learning_rate', 'eta']),
                         ('subsample', None),
                         ('colsample_bytree', None),
                         ('colsample_bylevel', None),
                         ('colsample_bynode', None),
                         ('alpha', ['alpha', 'reg_alpha']),
                         ('lambda', ['lambda', 'reg_lambda']),
                         ('gamma', ['gamma', 'reg_gamma']),
                         ('tree_method', None),
                         ('min_child_weight', None),
                         ('max_delta_step', None),
                         ('max_cat_to_onehot', ['max_cat_to_onehot', 'max_onehot_cat_size', 'one_hot_max_size'], None),
                         ('num_parallel_tree', None),
                         ('max_bin', None),
                         ('nthread', ['nthread', 'n_threads'], n_threads),
                         ]

        params = utils.extract_params(self.config, params_config)
        if self.n_classes > 0:
            return XGBClassifier(random_state=seed, **params)
        else:
            return XGBRegressor(random_state=seed, **params)

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        assert n_cv == 1
        assert n_refit == 0
        assert n_splits == 1
        updated_config = utils.join_dicts(dict(n_estimators=1000, max_depth=6), self.config)
        rc = ResourcePredictor(config=updated_config, time_params=ResourceParams.xgb_class_time,
                               cpu_ram_params=ResourceParams.xgb_class_ram)
        return rc.get_required_resources(ds)


class XGBSubSplitInterface(TreeBasedSubSplitInterface):
    # for RF: https://xgboost.readthedocs.io/en/latest/tutorials/rf.html
    def _get_params(self):
        # n_estimators is not set in params but directly in bst.fit() below
        params_config = [('verbosity', None, 0),
                         ('max_depth', None, 6),
                         ('eta', ['lr', 'learning_rate', 'eta'], 0.3),
                         ('subsample', None, 1.0),
                         ('colsample_bytree', None, 1.0),
                         ('colsample_bylevel', None, 1.0),
                         ('colsample_bynode', None, 1.0),
                         ('alpha', ['alpha', 'reg_alpha'], 0.0),
                         ('lambda', ['lambda', 'reg_lambda'], 1.0),
                         ('gamma', ['gamma', 'reg_gamma'], 0.0),
                         ('tree_method', None, 'auto'),
                         ('min_child_weight', None),
                         ('max_delta_step', None),
                         ('max_cat_to_onehot', ['max_cat_to_onehot', 'max_onehot_cat_size', 'one_hot_max_size'], None),
                         ('num_parallel_tree', None),
                         ('max_bin', None),
                         ('multi_strategy', None)
                         ]

        params = utils.extract_params(self.config, params_config)
        if self.config.get('use_gpu', False):
            params['tree_method'] = 'gpu_hist'
        return params

    def get_refit_interface(self, n_refit: int, fit_params: Optional[List[Dict]] = None) -> 'AlgInterface':
        assert n_refit == 1
        return XGBSubSplitInterface(fit_params=fit_params or self.fit_params, **self.config)

    # adapted from https://github.com/catboost/benchmarks/blob/master/quality_benchmarks/xgboost_experiment.py
    def _preprocess_params(self, params: Dict[str, Any], n_classes: int) -> Dict[str, Any]:
        params = copy.deepcopy(params)
        if n_classes == 0:
            train_metric_name = self.config.get('train_metric_name', 'mse')
            # val_metric_name = self.config.get('val_metric_name', 'rmse')
            if train_metric_name == 'mse':
                params['objective'] = 'reg:squarederror'
                # params['eval_metric'] = 'rmse'
            elif train_metric_name.startswith('pinball('):
                quantile = float(train_metric_name[len('pinball('):-1])
                params['objective'] = 'reg:quantileerror'
                params['quantile_alpha'] = quantile
            else:
                raise ValueError(f'Train metric "{train_metric_name}" is currently not supported!')
            # params.update({'objective': 'reg:squarederror', 'eval_metric': 'rmse'})
        elif n_classes == 2:
            params.update({'objective': 'binary:logistic'})
        elif n_classes > 2:
            params.update({'objective': 'multi:softprob', 'num_class': n_classes})

        if n_classes <= 2 and 'multi_strategy' in params:
            del params['multi_strategy']

        # could use gpu using
        # param['gpu_id'] = 0
        # param['tree_method'] = 'gpu_hist'

        params['max_depth'] = int(params['max_depth'])
        return params

    def _convert_ds(self, ds: DictDataset) -> Any:
        label = None if 'y' not in ds.tensors else ds.tensors['y'].cpu().numpy()
        has_cat = 'x_cat' in ds.tensor_infos and ds.tensor_infos['x_cat'].get_n_features() > 0
        x_df = ds.without_labels().to_df()
        return xgb.DMatrix(x_df, label, enable_categorical=has_cat)

    def _fit(self, train_ds: DictDataset, val_ds: Optional[DictDataset], params: Dict[str, Any], seed: int,
             n_threads: int, val_metric_name: Optional[str] = None,
             tmp_folder: Optional[Path] = None) -> Tuple[Any, Optional[List[float]]]:
        # print(f'Fitting XGBoost')
        n_classes = train_ds.tensor_infos['y'].get_cat_sizes()[0].item()
        params = self._preprocess_params(params, n_classes)
        params.update({'seed': seed, 'nthread': n_threads})
        evals = [] if val_ds is None else [(self._convert_ds(val_ds), 'val')]
        evals_result = {}

        feval = None
        eval_metric_name = None

        if val_ds is not None:
            if val_metric_name is None:
                val_metric_name = 'class_error' if n_classes > 0 else 'rmse'

            if val_metric_name == 'class_error':
                eval_metric_name = 'error' if n_classes == 2 else 'merror'
            elif val_metric_name == 'cross_entropy':
                eval_metric_name = 'logloss' if n_classes == 2 else 'mlogloss'
            elif val_metric_name == 'rmse':
                eval_metric_name = 'rmse'
            elif val_metric_name == 'mae':
                eval_metric_name = 'mae'
            else:
                eval_metric_name = val_metric_name
                feval = XGBCustomMetric(val_metric_name, is_classification=n_classes > 0)

            if feval is None:
                params['eval_metric'] = eval_metric_name
            else:
                params['disable_default_eval_metric'] = True

        extra_train_params = {}
        if val_ds is not None and 'early_stopping_rounds' in self.config:
            extra_train_params['early_stopping_rounds'] = self.config['early_stopping_rounds']

        n_estimators = self.config.get('n_estimators', 1000)
        if 'n_estimators' in params:
            # can happen for refit because fit_params are directly joined into params
            n_estimators = int(params['n_estimators'])

        bst = xgb.train(params, self._convert_ds(train_ds), evals=evals, evals_result=evals_result, custom_metric=feval,
                        num_boost_round=n_estimators, verbose_eval=False,
                        **extra_train_params)

        if val_ds is not None:
            val_errors = evals_result['val'][eval_metric_name]
        else:
            val_errors = None
        return bst, val_errors

    def _predict(self, bst: xgb.Booster, ds: DictDataset, n_classes: int, other_params: Dict[str, Any]) -> torch.Tensor:
        # print(f'XGB _predict() with {other_params=}')
        iteration_range = (0, 0) if other_params is None else (0, int(other_params['n_estimators']))
        y_pred = torch.as_tensor(bst.predict(self._convert_ds(ds), iteration_range=iteration_range), dtype=torch.float32)
        if n_classes == 0:
            y_pred = y_pred.unsqueeze(-1)
        elif n_classes == 2:
            y_pred = torch.stack([1. - y_pred, y_pred], dim=-1)

        if n_classes >= 2:
            y_pred = torch.log(y_pred + 1e-30)
        # print(f'min: {torch.min(y_pred).item():g}, max: {torch.max(y_pred).item():g}')
        return y_pred

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        assert n_cv == 1
        assert n_refit == 0
        assert n_splits == 1
        updated_config = utils.join_dicts(dict(n_estimators=1000, max_depth=6, max_n_threads=8), self.config)
        rc = ResourcePredictor(config=updated_config, time_params=ResourceParams.xgb_class_time,
                               cpu_ram_params=ResourceParams.xgb_class_ram)
        return rc.get_required_resources(ds)


class XGBHyperoptAlgInterface(OptAlgInterface):
    def __init__(self, space=None, n_hyperopt_steps: int = 50, **config):
        from hyperopt import hp
        default_config = {}
        max_config = dict()
        if space == 'catboost_quality_benchmarks':
            # space from catboost quality benchmarks
            # https://github.com/catboost/benchmarks/blob/master/quality_benchmarks/xgboost_experiment.py
            # the parameter names in the space are for the alg interface, not directly for the GBDT interface!
            space = {
                'eta': hp.loguniform('eta', -7, 0),
                'max_depth': hp.quniform('max_depth', 2, 10, 1),
                'subsample': hp.uniform('subsample', 0.5, 1),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
                'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1),
                'min_child_weight': hp.loguniform('min_child_weight', -16, 5),
                'reg_alpha': hp.choice('alpha', [0, hp.loguniform('alpha_positive', -16, 2)]),
                'reg_lambda': hp.choice('lambda', [0, hp.loguniform('lambda_positive', -16, 2)]),
                'reg_gamma': hp.choice('gamma', [0, hp.loguniform('gamma_positive', -16, 2)])
            }
            default_config = dict(n_estimators=5000)
            max_config['max_depth'] = 10
        elif space == 'NODE' or space == 'popov':
            # space from NODE paper:
            # Popov, Morozov, and Babenko, Neural oblivious decision ensembles for deep learning on tabular data
            # the parameter names in the space are for the alg interface, not directly for the GBDT interface!
            # same as catboost_quality_benchmarks except with smaller n_estimators
            space = {
                'eta': hp.loguniform('eta', -7, 0),
                'max_depth': hp.quniform('max_depth', 2, 10, 1),
                'subsample': hp.uniform('subsample', 0.5, 1),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
                'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1),
                'min_child_weight': hp.loguniform('min_child_weight', -16, 5),
                'reg_alpha': hp.choice('alpha', [0, hp.loguniform('alpha_positive', -16, 2)]),
                'reg_lambda': hp.choice('lambda', [0, hp.loguniform('lambda_positive', -16, 2)]),
                'reg_gamma': hp.choice('gamma', [0, hp.loguniform('gamma_positive', -16, 2)])
            }
            default_config = dict(n_estimators=2048)
            max_config['max_depth'] = 10
        elif space == 'shwartz-ziv':
            # from Shwartz-Ziv and Armon, Tabular data: Deep learning is not all you need
            # the TabPFN-Paper uses the same configuration
            space = {
                'n_estimators': hp.quniform('n_estimators', 100, 4000, 1),
                'eta': hp.loguniform('eta', -7, 0),
                'max_depth': hp.quniform('max_depth', 1, 10, 1),
                'subsample': hp.uniform('subsample', 0.2, 1),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 1),
                'colsample_bylevel': hp.uniform('colsample_bylevel', 0.2, 1),
                'min_child_weight': hp.loguniform('min_child_weight', -16, 5),
                'reg_alpha': hp.choice('alpha', [0, hp.loguniform('alpha_positive', -16, 2)]),
                'reg_lambda': hp.choice('lambda', [0, hp.loguniform('lambda_positive', -16, 2)]),
                'reg_gamma': hp.choice('gamma', [0, hp.loguniform('gamma_positive', -16, 2)])
            }
            max_config['max_depth'] = 10
        elif space == 'kadra':
            # from Kadra, Lindauer, Hutter, and Grabocka, Well-tuned Simple Nets Excel on Tabular Datasets
            space = {
                'n_estimators': hp.quniform('n_estimators', 1, 1000, 1),
                'eta': hp.loguniform('eta', np.log(1e-3), 0),
                'max_depth': hp.quniform('max_depth', 1, 20, 1),
                'subsample': hp.uniform('subsample', 0.01, 1),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
                'colsample_bynode': hp.uniform('colsample_bynode', 0.1, 1),
                'colsample_bylevel': hp.uniform('colsample_bylevel', 0.1, 1),
                'min_child_weight': hp.loguniform('min_child_weight', np.log(0.1), np.log(20.0)),
                'max_delta_step': hp.quniform('max_delta_step', 0, 10, 1),
                'reg_alpha': hp.loguniform('alpha', np.log(1e-10), 0),
                'reg_lambda': hp.loguniform('lambda', np.log(1e-10), 0),
                'reg_gamma': hp.loguniform('gamma', np.log(1e-10), 0)
            }
            max_config['max_depth'] = 20
        elif space == 'grinsztajn':
            # from Grinsztajn, Oyallon, Varoquaux,
            # Why do tree-based models still outperform deep learning on typical tabular data?
            #  they have early-stopping-rounds=20
            #  they also use XGBClassifier / XGBRegressor from scikit-learn
            #  they also start the random searches with the default hyperparameters of the model
            # see https://github.com/LeoGrin/tabular-benchmark/blob/main/src/configs/model_configs/xgb_config.py
            space = {
                'eta': hp.loguniform('eta', np.log(1e-5), np.log(0.7)),
                'max_depth': hp.quniform('max_depth', 1, 11, 1),
                'min_child_weight': hp.qloguniform('min_child_weight', 0.0, np.log(100.0), 1),
                'subsample': hp.uniform('subsample', 0.5, 1),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
                'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1),
                'reg_alpha': hp.loguniform('alpha', np.log(1e-8), np.log(1e-2)),
                'reg_lambda': hp.loguniform('lambda', np.log(1.0), np.log(4.0)),
                'reg_gamma': hp.loguniform('gamma', np.log(1e-8), np.log(7.0))
            }
            default_config = dict(n_estimators=1000)
            max_config['max_depth'] = 11
        elif space == 'gorishniy':
            # from Gorishniy, Rubachev, Khrulkov, Babenko, Revisiting Deep Learning Models for Tabular Data
            # they also have booster = "gbtree" (default), early-stopping-rounds=50,
            #  n_hyperopt_steps=100
            space = {
                'eta': hp.loguniform('eta', np.log(1e-5), np.log(1.0)),
                'max_depth': hp.quniform('max_depth', 3, 10, 1),
                'min_child_weight': hp.qloguniform('min_child_weight', np.log(1e-8), np.log(1e5), 1),
                'subsample': hp.uniform('subsample', 0.5, 1),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
                'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1),
                'reg_alpha': hp.choice('alpha', [0, hp.loguniform('alpha_positive', np.log(1e-8), np.log(1e2))]),
                'reg_lambda': hp.choice('lambda', [0, hp.loguniform('lambda_positive', np.log(1e-8), np.log(1e2))]),
                'reg_gamma': hp.choice('gamma', [0, hp.loguniform('gamma_positive', np.log(1e-8), np.log(1e2))])
            }
            default_config = dict(n_estimators=2000)
            max_config['max_depth'] = 10
        elif space == 'custom-v1':
            space = {
                'eta': hp.loguniform('eta', np.log(2e-3), np.log(0.5)),
                'max_depth': hp.quniform('max_depth', 1, 10, 1),
                'min_child_weight': hp.qloguniform('min_child_weight', np.log(1e-5), np.log(100.0), 1),
                'subsample': hp.uniform('subsample', 0.4, 1),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
                'colsample_bylevel': hp.uniform('colsample_bylevel', 0.5, 1),
                'reg_alpha': hp.loguniform('alpha', np.log(1e-8), np.log(1.0)),
                'reg_lambda': hp.loguniform('lambda', np.log(1e-8), np.log(4.0)),
                'reg_gamma': hp.loguniform('gamma', np.log(1e-8), np.log(7.0))
            }
            default_config = dict(n_estimators=1000)
            max_config['max_depth'] = 11

        config = utils.update_dict(default_config, config)
        super().__init__(hyper_optimizer=HyperoptOptimizer(space=space, fixed_params=dict(),
                                                           n_hyperopt_steps=n_hyperopt_steps,
                                                           **config),
                         max_resource_config=utils.join_dicts(config, max_config),
                         **config)

    def create_alg_interface(self, n_sub_splits: int, **config) -> AlgInterface:
        return SingleSplitWrapperAlgInterface([XGBSubSplitInterface(**config) for i in range(n_sub_splits)])


class RandomParamsXGBAlgInterface(RandomParamsAlgInterface):
    def _sample_params(self, is_classification: bool, seed: int, n_train: int):
        rng = np.random.default_rng(seed)
        # adapted from Grinsztajn et al. (2022)
        space = {
            'eta': np.exp(rng.uniform(np.log(1e-5), np.log(0.7))),
            'max_depth': rng.integers(1, 11, endpoint=True),
            'min_child_weight': round(np.exp(rng.uniform(0.0, np.log(100.0)))),
            'subsample': rng.uniform(0.5, 1),
            'colsample_bytree': rng.uniform(0.5, 1),
            'colsample_bylevel': rng.uniform(0.5, 1),
            'reg_alpha': np.exp(rng.uniform(np.log(1e-8), np.log(1e-2))),
            'reg_lambda': np.exp(rng.uniform(np.log(1.0), np.log(4.0))),
            'reg_gamma': np.exp(rng.uniform(np.log(1e-8), np.log(7.0)))
        }
        return space

    def _create_interface_from_config(self, n_tv_splits: int, **config):
        return SingleSplitWrapperAlgInterface([XGBSubSplitInterface(**config) for i in range(n_tv_splits)])
