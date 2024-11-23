import copy
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, List

import numpy as np
import torch

from lightgbm import record_evaluation, LGBMClassifier, LGBMRegressor

from pytabkit.models.alg_interfaces.resource_computation import ResourcePredictor
from pytabkit.models.alg_interfaces.resource_params import ResourceParams
from pytabkit.models import utils
from pytabkit.models.alg_interfaces.alg_interfaces import OptAlgInterface, \
    AlgInterface, RandomParamsAlgInterface
from pytabkit.models.alg_interfaces.base import RequiredResources
from pytabkit.models.alg_interfaces.sub_split_interfaces import TreeBasedSubSplitInterface, SingleSplitWrapperAlgInterface, \
    SklearnSubSplitInterface
from pytabkit.models.data.data import DictDataset
from pytabkit.models.hyper_opt.hyper_optimizers import HyperoptOptimizer, SMACOptimizer
import lightgbm as lgbm
import warnings

from pytabkit.models.training.metrics import Metrics


class LGBMCustomMetric:
    def __init__(self, metric_name: str, is_classification: bool, is_higher_better: bool = False):
        self.metric_name = metric_name
        self.is_classification = is_classification
        self.is_higher_better = is_higher_better

    def __call__(self, y_pred: np.ndarray, eval_data: lgbm.Dataset):
        y = torch.as_tensor(eval_data.get_label(), dtype=torch.long if self.is_classification else torch.float32)
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
                    # bugged multiclass classification, need to reshape
                    # print(y_pred[:7])
                    y_pred = y_pred.view(-1, y.shape[0]).t().contiguous()
                    # print(y_pred[0, :].sum())
            else:
                y_pred = y_pred[:, None]

        if self.is_classification:
            # go from probabilities to logits
            y_pred = torch.log(y_pred + 1e-30)

        eval_result = Metrics.apply(y_pred, y, metric_name=self.metric_name)

        print(f'LightGBM metric value: {self.metric_name} = {eval_result.item():g}')

        return self.metric_name, eval_result, self.is_higher_better


class LGBMSklearnSubSplitInterface(SklearnSubSplitInterface):
    def _get_cat_indexes_arg_name(self) -> str:
        return 'categorical_feature'

    def _create_sklearn_model(self, seed: int, n_threads: int, gpu_devices: List[str]) -> Any:
        params_config = [('n_estimators', None),
                         ('max_depth', None),
                         ('verbosity', None),
                         ('learning_rate', ['lr', 'learning_rate', 'eta']),
                         ('subsample', ['subsample', 'bagging_fraction']),
                         ('colsample_bytree', ['colsample_bytree', 'feature_fraction']),
                         ('bagging_freq', None),
                         ('min_data_in_leaf', None),
                         ('min_sum_hessian_in_leaf', ['min_sum_hessian_in_leaf', 'min_child_weight']),
                         ('lambda_l1', ['lambda_l1', 'alpha', 'reg_alpha']),
                         ('lambda_l2', ['lambda_l2', 'lambda', 'reg_lambda']),
                         ('num_leaves', None),
                         ('min_child_weight', None),
                         ('boosting_type', None),
                         ('max_bin', None),
                         ('cat_smooth', None),
                         ('cat_l2', None),
                         ('n_jobs', ['n_jobs', 'n_threads'], n_threads),
                         ]

        params = utils.extract_params(self.config, params_config)
        if self.n_classes > 0:
            return LGBMClassifier(random_state=seed, **params)
        else:
            return LGBMRegressor(random_state=seed, **params)

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        assert n_cv == 1
        assert n_refit == 0
        assert n_splits == 1
        updated_config = utils.join_dicts(dict(n_estimators=1000, num_leaves=31), self.config)
        rc = ResourcePredictor(config=updated_config, time_params=ResourceParams.lgbm_class_time,
                               cpu_ram_params=ResourceParams.lgbm_class_ram)
        return rc.get_required_resources(ds)


class LGBMSubSplitInterface(TreeBasedSubSplitInterface):
    def _get_params(self):
        params_config = [('n_estimators', None, 1000),
                         ('max_depth', None),
                         ('verbosity', None, -1),
                         ('learning_rate', ['lr', 'learning_rate', 'eta'], 0.1),
                         ('subsample', ['subsample', 'bagging_fraction'], 1.0),
                         ('colsample_bytree', ['colsample_bytree', 'feature_fraction'], 1.0),
                         ('bagging_freq', None, 1),  # 1 is not the default in the interface but 0 could be misleading
                         ('min_data_in_leaf', None, 20),
                         ('min_sum_hessian_in_leaf', ['min_sum_hessian_in_leaf', 'min_child_weight'], 1e-3),
                         ('lambda_l1', ['lambda_l1', 'alpha', 'reg_alpha'], 0.0),
                         ('lambda_l2', ['lambda_l2', 'lambda', 'reg_lambda'], 0.0),
                         ('num_leaves', None, 31),
                         ('boosting', ['boosting', 'boosting_type'], None),
                         ('max_bin', None),
                         ('cat_smooth', None),
                         ('cat_l2', None),
                         ('early_stopping_round', ['early_stopping_round', 'early_stopping_rounds'], None),
                         ]

        params = utils.extract_params(self.config, params_config)
        return params

    def get_refit_interface(self, n_refit: int, fit_params: Optional[List[Dict]] = None) -> 'AlgInterface':
        assert n_refit == 1
        return LGBMSubSplitInterface(fit_params=fit_params or self.fit_params, **self.config)

    # adapted from https://github.com/catboost/benchmarks/blob/master/quality_benchmarks/lightgbm_experiment.py
    def _preprocess_params(self, params: Dict[str, Any], n_classes: int) -> Dict[str, Any]:
        params = copy.deepcopy(params)
        if n_classes == 0:
            train_metric_name = self.config.get('train_metric_name', 'mse')
            if train_metric_name == 'mse':
                params.update({'objective': 'mean_squared_error'})
            elif train_metric_name.startswith('pinball('):
                quantile = float(train_metric_name[len('pinball('):-1])
                params.update({'objective': 'quantile', 'alpha': quantile})
            else:
                raise ValueError(f'Train metric "{train_metric_name}" is currently not supported!')
        elif n_classes <= 2:
            params.update({'objective': 'binary'})
        elif n_classes > 2:
            params.update({'objective': 'multiclass', 'num_class': n_classes})

        if 'num_leaves' in params:
            params['num_leaves'] = max(int(params['num_leaves']), 2)
        if 'min_data_in_leaf' in params:
            params['min_data_in_leaf'] = int(params['min_data_in_leaf'])
        return params

    def _convert_ds(self, ds: DictDataset) -> Any:
        x_cont = ds.tensors['x_cont'].cpu().numpy()
        label = None if 'y' not in ds.tensors else ds.tensors['y'].cpu().numpy()
        if label is not None and label.shape[1] == 1:
            label = label[:, 0]
        has_cat = 'x_cat' in ds.tensor_infos and ds.tensor_infos['x_cat'].get_n_features() > 0
        if not has_cat:
            # no categorical columns
            return lgbm.Dataset(x_cont, label=label, categorical_feature=[])

        x_df = ds.without_labels().to_df()
        cat_features = x_df.select_dtypes(include='category').columns.tolist()
        return lgbm.Dataset(x_df, label, categorical_feature=cat_features)

    def _fit(self, train_ds: DictDataset, val_ds: Optional[DictDataset], params: Dict[str, Any], seed: int,
             n_threads: int, val_metric_name: Optional[str] = None,
             tmp_folder: Optional[Path] = None) -> Tuple[Any, Optional[List[float]]]:
        # print(f'Fitting LightGBM')
        n_classes = train_ds.tensor_infos['y'].get_cat_sizes()[0].item()
        params = self._preprocess_params(params, n_classes)
        params.update({
            'data_random_seed': 1 + seed,
            'feature_fraction_seed': 2 + seed,
            'bagging_seed': 3 + seed,
            'drop_seed': 4 + seed,
            'objective_seed': 5 + seed,
            'extra_seed': 6 + seed,
            'num_threads': n_threads
        })

        eval_metric = None
        eval_name = None
        feval = None

        if val_ds is not None:
            if val_metric_name is None:
                val_metric_name = 'class_error' if n_classes > 0 else 'rmse'

            if val_metric_name == 'class_error':
                eval_metric = 'binary_error' if n_classes <= 2 else 'multi_error'
            elif val_metric_name == 'cross_entropy':
                eval_metric = 'binary_logloss' if n_classes <= 2 else 'multi_logloss'
            elif val_metric_name == 'rmse':
                eval_metric = 'rmse'
            elif val_metric_name == 'mae':
                eval_metric = 'mae'
            else:
                eval_name = val_metric_name
                feval = LGBMCustomMetric(val_metric_name, is_classification=n_classes > 0)

            if eval_metric is None:
                # specified custom metric, don't use pre-given metric
                eval_metric = "None"
            else:
                eval_name = eval_metric

            params['metric'] = eval_metric

        if val_ds is None:
            params = utils.update_dict(params, remove_keys=['early_stopping_round', 'early_stopping_rounds'])

        evals = [] if val_ds is None else [self._convert_ds(val_ds)]
        valid_names = [] if val_ds is None else ['val']
        evals_result = {}
        train_ds = self._convert_ds(train_ds)
        # warning filtering taken from https://auto.gluon.ai/dev/_modules/autogluon/tabular/models/lgb/lgb_model.html
        with warnings.catch_warnings():
            # Filter harmless warnings introduced in lightgbm 3.0,
            # future versions plan to remove: https://github.com/microsoft/LightGBM/issues/3379
            warnings.filterwarnings('ignore', message='Overriding the parameters from Reference Dataset.')
            warnings.filterwarnings('ignore', message='categorical_column in param dict is overridden.')
            bst = lgbm.train(utils.update_dict(params, remove_keys=['n_estimators']), train_ds, valid_sets=evals,
                             valid_names=valid_names, feval=feval,
                             callbacks=[record_evaluation(evals_result)],
                             num_boost_round=params['n_estimators'])

        if val_ds is not None:
            # print('evals_result val:', evals_result['val'], flush=True)
            val_errors = evals_result['val'][eval_name]
        else:
            val_errors = None
        return bst, val_errors

    def _predict(self, bst: lgbm.Booster, ds: DictDataset, n_classes: int, other_params: Dict[str, Any]) -> torch.Tensor:
        # print(f'LGBM _predict() with {other_params=}')
        num_iteration = None if other_params is None else other_params['n_estimators']
        y_pred = torch.as_tensor(bst.predict(self._convert_ds(ds).data, num_iteration=num_iteration),
                                 dtype=torch.float32)
        if n_classes == 0:
            y_pred = y_pred.unsqueeze(-1)
        elif n_classes <= 2:
            y_pred = torch.stack([1. - y_pred, y_pred], dim=-1)

        if n_classes >= 1:
            y_pred = torch.log(y_pred + 1e-30)
        # print(f'min: {torch.min(y_pred).item():g}, max: {torch.max(y_pred).item():g}')
        return y_pred

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        assert n_cv == 1
        assert n_refit == 0
        assert n_splits == 1
        updated_config = utils.join_dicts(dict(n_estimators=1000, num_leaves=31, max_n_threads=8), self.config)
        rc = ResourcePredictor(config=updated_config, time_params=ResourceParams.lgbm_class_time,
                               cpu_ram_params=ResourceParams.lgbm_class_ram)
        return rc.get_required_resources(ds)


class LGBMHyperoptAlgInterface(OptAlgInterface):
    def __init__(self, space=None, n_hyperopt_steps: int = 50, opt_method: str = 'hyperopt', **config):
        from hyperopt import hp
        default_config = {}
        max_config = dict()
        if space == 'catboost_quality_benchmarks':
            # space from catboost quality benchmarks,
            # https://github.com/catboost/benchmarks/blob/master/quality_benchmarks/lightgbm_experiment.py
            # the parameter names in the space are for the alg interface, not directly for the GBDT interface!
            space = {
                'learning_rate': hp.loguniform('learning_rate', -7, 0),
                'num_leaves': hp.qloguniform('num_leaves', 0, 7, 1),
                'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
                'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
                'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', 0, 6, 1),
                'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', -16, 5),
                'lambda_l1': hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)]),
                'lambda_l2': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)]),
            }
            default_config = dict(n_estimators=5000)
            max_config['num_leaves'] = 1000  # about exp(7)
        elif space == 'tabpfn' or space == 'hollmann':
            # from Hollmann, Müller, Eggensperger, Hutter,
            # TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second
            # the parameter names in the space are for the alg interface, not directly for the GBDT interface!
            space = {
                'n_estimators': hp.quniform('n_estimators', 50, 2000),
                # in the paper it says that this is not log but that's hard to believe,
                # especially when e^{-3} is the lower bound
                'learning_rate': hp.loguniform('learning_rate', -3, 0),
                'num_leaves': hp.qloguniform('num_leaves', np.log(5), np.log(50), 1),
                'max_depth': hp.qloguniform('max_depth', np.log(3), np.log(20), 1),
                'subsample': hp.uniform('subsample', 0.2, 0.8),
                'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', -5, 4),  # this is min_child_weight
                'lambda_l1': hp.choice('lambda_l1', [0, 1e-1, 1, 2, 5, 7, 10, 50, 100]),  # this is reg_alpha
                'lambda_l2': hp.choice('lambda_l2', [0, 1e-1, 1, 2, 5, 7, 10, 50, 100]),  # this is reg_lambda
            }
            max_config['num_leaves'] = 50
        elif space == 'mt-reg':
            # hand-guessed space for regression
            if opt_method == 'smac':
                from ConfigSpace import ConfigurationSpace, Float, Integer
                space = ConfigurationSpace()
                space.add_hyperparameters([
                    Integer('num_leaves', (16, 256), log=True, default=100),
                    Float('feature_fraction', (0.4, 1), default=0.7),
                    Float('bagging_fraction', (0.6, 1), default=1.0),
                    Integer('min_data_in_leaf', (1, 64), log=True, default=3),
                ])
            else:  # assume hyperopt
                space = {
                    'num_leaves': hp.qloguniform('num_leaves', np.log(16), np.log(256), 1),
                    'feature_fraction': hp.uniform('feature_fraction', 0.4, 1),
                    'bagging_fraction': hp.uniform('bagging_fraction', 0.6, 1),
                    'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', np.log(1), np.log(64), 1),
                }
            default_config = dict(n_estimators=1000, learning_rate=5e-2, min_sum_hessian_in_leaf=1e-5)
            max_config['num_leaves'] = 256
        elif space == 'mt-reg-2':
            # hand-guessed space for regression
            space = {
                'num_leaves': hp.qloguniform('num_leaves', np.log(16), np.log(256), 1),
                'learning_rate': hp.loguniform('learning_rate', np.log(2.5e-2), np.log(1e-1)),
                'feature_fraction': hp.uniform('feature_fraction', 0.4, 1),
                'bagging_fraction': hp.uniform('bagging_fraction', 0.6, 1),
                'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', np.log(1), np.log(64), 1),
            }
            default_config = dict(n_estimators=1000, min_sum_hessian_in_leaf=1e-5)
            max_config['num_leaves'] = 256
        config = utils.update_dict(default_config, config)
        opt_class = SMACOptimizer if opt_method == 'smac' else HyperoptOptimizer
        super().__init__(hyper_optimizer=opt_class(space=space, fixed_params=dict(),
                                                   n_hyperopt_steps=n_hyperopt_steps,
                                                   **config),
                         max_resource_config=utils.join_dicts(config, max_config),
                         **config)

    def create_alg_interface(self, n_sub_splits: int, **config) -> AlgInterface:
        return SingleSplitWrapperAlgInterface([LGBMSubSplitInterface(**config) for i in range(n_sub_splits)])


class RandomParamsLGBMAlgInterface(RandomParamsAlgInterface):
    def _sample_params(self, is_classification: bool, seed: int, n_train: int):
        rng = np.random.default_rng(seed)
        # adapted from catboost quality benchmarks
        space = {
            'learning_rate': np.exp(rng.uniform(-7, 0)),
            'num_leaves': round(np.exp(rng.uniform(0, 7))),
            'feature_fraction': rng.uniform(0.5, 1),
            'bagging_fraction': rng.uniform(0.5, 1),
            'min_data_in_leaf': round(np.exp(rng.uniform(0, 6))),
            'min_sum_hessian_in_leaf': np.exp(rng.uniform(-16, 5)),
            'lambda_l1': rng.choice([0.0, np.exp(rng.uniform(-16, 2))]),
            'lambda_l2': rng.choice([0.0, np.exp(rng.uniform(-16, 2))]),
            'n_estimators': 1000,
        }
        return space

    def _create_interface_from_config(self, n_tv_splits: int, **config):
        return SingleSplitWrapperAlgInterface([LGBMSubSplitInterface(**config) for i in range(n_tv_splits)])

