import copy
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union

import numpy as np
import torch

from pytabkit.models.alg_interfaces.resource_computation import ResourcePredictor
from pytabkit.models.alg_interfaces.resource_params import ResourceParams
from pytabkit.models import utils
from pytabkit.models.alg_interfaces.base import RequiredResources
from pytabkit.models.alg_interfaces.sub_split_interfaces import TreeBasedSubSplitInterface, \
    SingleSplitWrapperAlgInterface, \
    SklearnSubSplitInterface
from pytabkit.models.data.data import DictDataset
from pytabkit.models.hyper_opt.hyper_optimizers import HyperoptOptimizer

from pytabkit.models.alg_interfaces.alg_interfaces import AlgInterface, \
    OptAlgInterface, RandomParamsAlgInterface
from pytabkit.models.training.metrics import Metrics


class CatBoostSklearnSubSplitInterface(SklearnSubSplitInterface):
    def _get_cat_indexes_arg_name(self) -> str:
        return 'cat_features'

    def _create_sklearn_model(self, seed: int, n_threads: int, gpu_devices: List[str]) -> Any:
        params_config = [('n_estimators', None, 1000),
                         ('depth', ['depth', 'max_depth'], 6),
                         ('random_strength', None, 1.0),
                         ('l2_leaf_reg', None, 3.0),
                         ('depth', ['depth', 'max_depth'], 6),
                         ('learning_rate', ['lr', 'learning_rate', 'eta']),
                         ('one_hot_max_size', None),
                         ('bagging_temperature', None),
                         ('leaf_estimation_iterations', None),
                         ('bootstrap_type', None),
                         ('subsample', None),
                         ('sampling_frequency', None),
                         ('boosting_type', None),
                         ('colsample_bylevel', ['colsample_bylevel', 'rsm'], None),
                         ('min_data_in_leaf', ['min_data_in_leaf', 'min_child_samples'], None),
                         ('grow_policy', None),
                         ('num_leaves', None),
                         ('border_count', ['border_count', 'max_bin']),
                         ('thread_count', ['thread_count', 'n_threads'], n_threads),
                         ('verbose', None, False),
                         ('allow_writing_files', None, False),
                         ]

        params = utils.extract_params(self.config, params_config)
        if self.n_classes > 0:
            from catboost import CatBoostClassifier

            return CatBoostClassifier(random_state=seed, **params)
        else:
            from catboost import CatBoostRegressor

            return CatBoostRegressor(random_state=seed, **params)

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        assert n_cv == 1
        assert n_refit == 0
        assert n_splits == 1
        updated_config = utils.join_dicts(dict(n_estimators=1000, max_depth=6), self.config)
        rc = ResourcePredictor(config=updated_config, time_params=ResourceParams.cb_class_time,
                               cpu_ram_params=ResourceParams.cb_class_ram)
        return rc.get_required_resources(ds)


class CatBoostCustomMetric:
    # see https://stackoverflow.com/questions/65462220/how-to-create-custom-eval-metric-for-catboost
    # and https://catboost.ai/en/docs/concepts/python-usages-examples

    def __init__(self, metric_name: str, is_classification: bool, is_higher_better: bool = False,
                 select_pred_col: Optional[int] = None):
        self.metric_name = metric_name
        self.is_classification = is_classification
        self.is_higher_better = is_higher_better
        self.select_pred_col = select_pred_col

    def is_max_optimal(self):
        return self.is_higher_better

    def evaluate(self, approxes, target, weight):
        assert len(target) == len(approxes[0])
        assert weight is None

        y = torch.as_tensor(target, dtype=torch.long if self.is_classification else torch.float32)
        if len(y.shape) == 1:
            y = y[:, None]

        y_pred = torch.as_tensor(np.array(approxes), dtype=torch.float32).t()
        # CatBoost already provides logits in approxes

        if self.select_pred_col is not None:
            y_pred = y_pred[:, self.select_pred_col, None]

        if self.is_classification and y_pred.shape[1] == 1:
            # binary classification, CatBoost provides logits of the class 1
            p = torch.sigmoid(y_pred)
            y_pred_probs = torch.cat([1. - p, p], dim=1)
            y_pred = torch.log(y_pred_probs + 1e-30)

        # print(f'{y.shape=}, {y_pred.shape=}')
        # print(f'{y_pred=}')

        loss = Metrics.apply(y_pred, y, self.metric_name).item()

        weight_sum = y.shape[0]

        return weight_sum * loss, weight_sum

    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)


class CatBoostSubSplitInterface(TreeBasedSubSplitInterface):
    def _get_params(self):
        # target parameter names, possible source parameter names, default value
        params_config = [('n_estimators', None, 1000),
                         ('depth', ['depth', 'max_depth'], 6),
                         ('random_strength', None, 1.0),
                         ('l2_leaf_reg', None, 3.0),
                         ('learning_rate', ['lr', 'learning_rate', 'eta']),
                         ('one_hot_max_size', None),
                         ('bagging_temperature', None),
                         ('leaf_estimation_iterations', None),
                         ('bootstrap_type', None),
                         ('subsample', None),
                         ('boosting_type', None, 'Plain'),  # fix default to Plain to equalize CPU and GPU
                         ('colsample_bylevel', ['colsample_bylevel', 'rsm'], None),
                         ('min_data_in_leaf', ['min_data_in_leaf', 'min_child_samples'], None),
                         ('grow_policy', None),
                         ('max_leaves', ['max_leaves', 'num_leaves'], None),
                         ('border_count', ['border_count', 'max_bin'], 254),  # fix default to 254 for GPU as well
                         ('used_ram_limit', None),
                         ('od_type', 'Iter'),
                         ('od_pval', None),
                         ('od_wait', ['od_wait', 'early_stopping_rounds'], None),
                         ('sampling_frequency', None),
                         ('max_ctr_complexity', None),
                         ('model_size_reg', None),
                         ]

        params = utils.extract_params(self.config, params_config)
        params['verbose'] = self.config.get('verbosity', 0) > 0
        bootstrap_type = params.get('bootstrap_type', 'Bayesian')
        if bootstrap_type == 'Bayesian':
            if 'subsample' in params:
                del params['subsample']
        elif bootstrap_type == 'Bernoulli':
            if 'bagging_temperature' in params:
                del params['bagging_temperature']
        grow_policy = params.get('grow_policy', 'SymmetricTree')
        if grow_policy != 'Lossguide':
            if 'max_leaves' in params:
                del params['max_leaves']
        if grow_policy == 'SymmetricTree':
            if 'min_data_in_leaf' in params:
                del params['min_data_in_leaf']
        return params

    def get_refit_interface(self, n_refit: int, fit_params: Optional[List[Dict]] = None) -> 'AlgInterface':
        assert n_refit == 1
        return CatBoostSubSplitInterface(fit_params=fit_params or self.fit_params, **self.config)

    def _get_eval_metric(self, val_metric_name: Optional[str], n_classes: int) -> Union[str, CatBoostCustomMetric]:
        if n_classes == 0:
            if val_metric_name is None or val_metric_name == 'rmse':
                return 'RMSE'
            else:
                return CatBoostCustomMetric(metric_name=val_metric_name,
                                            is_classification=n_classes > 0,
                                            is_higher_better=False)
            # else:
            #     raise ValueError(f'Validation metric "{val_metric_name}" is currently not implemented for CatBoost')
        else:
            # classification
            if val_metric_name is None or val_metric_name == 'classification_error':
                return 'ZeroOneLoss'
            elif val_metric_name == 'cross_entropy':
                return 'Logloss' if n_classes == 2 else 'MultiClass'
            elif val_metric_name == 'brier' and n_classes == 2:
                # catboost doesn't support brier score for multiclass yet
                return 'BrierScore'
            else:
                return CatBoostCustomMetric(metric_name=val_metric_name,
                                            is_classification=n_classes > 0,
                                            is_higher_better=False)
            # else:
            #     raise ValueError(f'Validation metric "{val_metric_name}" is currently not implemented for CatBoost')

    # adapted from https://github.com/catboost/benchmarks/blob/master/quality_benchmarks/catboost_experiment.py
    def _preprocess_params(self, params: Dict[str, Any], n_classes: int) -> Dict[str, Any]:
        params = copy.deepcopy(params)

        device: Optional[str] = params.pop('device', None)
        if device is not None and device.startswith('cuda'):
            params['task_type'] = 'GPU'
            params['devices'] = device.split(':')[1] if device.startswith('cuda:') else '0'

        if n_classes == 0:
            train_metric_name = self.config.get('train_metric_name', 'mse')
            # val_metric_name = self.config.get('val_metric_name', 'rmse')
            if train_metric_name == 'mse':
                params['loss_function'] = 'RMSE'
            elif train_metric_name.startswith('pinball('):
                quantile_str = train_metric_name[len('pinball('):-1]
                params['loss_function'] = f'Quantile:alpha={quantile_str}'
            else:
                raise ValueError(f'Train metric "{train_metric_name}" is currently not supported!')
        elif n_classes == 2:
            params.update({'loss_function': 'Logloss'})
        else:
            params.update({'loss_function': 'MultiClass', 'classes_count': n_classes})
        params['eval_metric'] = self._get_eval_metric(self.config.get('val_metric_name', None), n_classes)
        params['allow_writing_files'] = False
        params['use_best_model'] = False  # otherwise trees would get removed based only on a single split
        for key in ['random_strength', 'one_hot_max_size', 'leaf_estimation_iterations']:
            if key in params:
                params[key] = int(params[key])
        return params

    def _convert_ds(self, ds: DictDataset) -> Any:
        import catboost

        x_df = ds.without_labels().to_df()
        label = None if 'y' not in ds.tensors else ds.tensors['y'].cpu().numpy()
        cat_features = x_df.select_dtypes(include='category').columns.tolist()
        return catboost.Pool(x_df, label, cat_features=cat_features)

    def _fit(self, train_ds: DictDataset, val_ds: Optional[DictDataset], params: Dict[str, Any], seed: int,
             n_threads: int, val_metric_name: Optional[str] = None,
             tmp_folder: Optional[Path] = None) -> Tuple[Any, Optional[List[float]]]:
        import catboost

        # print(f'Fitting CatBoost')
        n_classes = train_ds.tensor_infos['y'].get_cat_sizes()[0].item()
        params = self._preprocess_params(params, n_classes)
        params.update({'random_seed': seed, 'thread_count': n_threads})

        if val_ds is None:
            params = utils.update_dict(params, remove_keys=['od_type', 'od_pval', 'od_wait'])

        if tmp_folder is not None:
            params.update({'allow_writing_files': True, 'save_snapshot': True,
                           'snapshot_file': str(tmp_folder / 'catboost_model.cbm'),
                           'snapshot_interval': 120.0})
            # with these parameters, catboost will reload from the model automatically if it is there
        bst = catboost.CatBoost(params)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Can\'t optimze method "evaluate" because self argument is used')
            bst.fit(self._convert_ds(train_ds), eval_set=None if val_ds is None else self._convert_ds(val_ds))

        if val_ds is not None:
            evals_result = bst.get_evals_result()
            # print(f'{evals_result["validation"]=}')
            eval_metric = self._get_eval_metric(self.config.get('val_metric_name', None), n_classes)
            eval_metric_name = eval_metric if isinstance(eval_metric, str) else eval_metric.__class__.__name__
            val_errors = evals_result['validation'][eval_metric_name]
        else:
            val_errors = None
        return bst, val_errors

    def _predict(self, bst, ds: DictDataset, n_classes: int,
                 other_params: Dict[str, Any]) -> torch.Tensor:
        # bst should be of type catboost.CatBoost
        # print(f'CatBoost _predict(): {other_params=}')
        ntree_end = 0 if other_params is None else other_params['n_estimators']
        prediction_type = 'RawFormulaVal' if n_classes == 0 else 'LogProbability'
        y_pred = torch.as_tensor(
            bst.predict(self._convert_ds(ds), ntree_end=ntree_end, prediction_type=prediction_type),
            dtype=torch.float32)
        if n_classes == 0:
            y_pred = y_pred.unsqueeze(-1)

        # print(f'{y_pred.shape=}')
        # print(f'{y_pred.mean(dim=0)=}')
        #
        # if torch.any(y_pred == -np.inf):
        #     y_pred_prob = torch.softmax(y_pred, dim=-1)
        #     # y_pred_prob = y_pred_prob.clamp(1e-10, 1)
        #     y_pred = torch.log(y_pred_prob + 1e-30)

        # y_pred = torch.clamp(y_pred, -100.0, 100.0)  # todo

        # print(f'min: {torch.min(y_pred).item():g}, max: {torch.max(y_pred).item():g}')
        return y_pred

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        assert n_cv == 1
        assert n_refit == 0
        assert n_splits == 1
        updated_config = utils.join_dicts(dict(n_estimators=1000, max_n_threads=8, max_depth=6), self.config)
        rc = ResourcePredictor(config=updated_config, time_params=ResourceParams.cb_class_time,
                               cpu_ram_params=ResourceParams.cb_class_ram)
        return rc.get_required_resources(ds)


class CatBoostHyperoptAlgInterface(OptAlgInterface):
    def __init__(self, space=None, n_hyperopt_steps: int = 50, **config):
        from hyperopt import hp
        default_config = {}
        max_config = {}
        # if space is None:
        # modified space from catboost quality benchmarks
        # https://github.com/catboost/benchmarks/blob/master/quality_benchmarks/catboost_experiment.py
        # space = {
        #     'depth': hp.choice('depth', [6]),
        #     # only 'ctr_target_border_count' exists for this catboost version
        #     # 'ctr_border_count': hp.choice('ctr_border_count', [16]),
        #     'border_count': hp.choice('border_count', [128]),
        #     # deprecated, CounterMax not allowed
        #     # 'ctr_description': hp.choice('ctr_description', [['Borders', 'CounterMax']]),
        #     'learning_rate': hp.loguniform('learning_rate', -5, 0),
        #     'random_strength': hp.choice('random_strength', [1, 20]),
        #     'one_hot_max_size': hp.choice('one_hot_max_size', [0, 25]),
        #     'l2_leaf_reg': hp.loguniform('l2_leaf_reg', 0, np.log(10)),
        #     'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
        #     'used_ram_limit': hp.choice('used_ram_limit', [100000000000]),
        # }
        # need to add defaults as well
        if space is None:
            space = config.get('hpo_space_name', None)
        if space == 'NODE' or space == 'popov':
            # space from NODE paper:
            # Popov, Morozov, and Babenko, Neural oblivious decision ensembles for deep learning on tabular data
            # the parameter names in the space are for the alg interface, not directly for the GBDT interface!
            space = {
                'learning_rate': hp.loguniform('learning_rate', -5, 0),
                'random_strength': hp.quniform('random_strength', 1, 20, 1),
                'one_hot_max_size': hp.quniform('one_hot_max_size', 0, 25, 1),
                'l2_leaf_reg': hp.loguniform('l2_leaf_reg', 0, np.log(10)),
                'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
                'leaf_estimation_iterations': hp.quniform('leaf_estimation_iterations', 1, 10, 1),
            }
            default_config = dict(n_estimators=2048)
            max_config['max_depth'] = 6
        elif space == 'shwartz-ziv':
            # from Shwartz-Ziv and Armon, Tabular data: Deep learning is not all you need
            # same as NODE except higher upper bound for leaf estimation iterations
            # the parameter names in the space are for the alg interface, not directly for the GBDT interface!
            space = {
                'learning_rate': hp.loguniform('learning_rate', -5, 0),
                'random_strength': hp.quniform('random_strength', 1, 20, 1),
                'one_hot_max_size': hp.quniform('one_hot_max_size', 0, 25, 1),
                'l2_leaf_reg': hp.loguniform('l2_leaf_reg', 0, np.log(10)),
                'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
                'leaf_estimation_iterations': hp.quniform('leaf_estimation_iterations', 1, 20, 1),
            }
            default_config = dict(n_estimators=2048)  # not specified from the paper, so we take the value from NODE
            max_config['max_depth'] = 6
        elif space == 'tabpfn' or space == 'hollmann':
            # from Hollmann, Müller, Eggensperger, Hutter,
            # TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second
            # similar to shwartz-ziv except that one_hot_max_size is not specified and n_estimators is optimized
            # the parameter names in the space are for the alg interface, not directly for the GBDT interface!
            space = {
                'n_estimators': hp.quniform('n_estimators', 100, 4000, 1),
                'learning_rate': hp.loguniform('learning_rate', -5, 0),
                'random_strength': hp.quniform('random_strength', 1, 20, 1),
                'l2_leaf_reg': hp.loguniform('l2_leaf_reg', 0, np.log(10)),
                'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
                'leaf_estimation_iterations': hp.quniform('leaf_estimation_iterations', 1, 20, 1),
            }
        elif space == 'gorishniy':
            # from Gorishniy, Rubachev, Khrulkov, Babenko, Revisiting Deep Learning Models for Tabular Data
            space = {
                'max_depth': hp.quniform('max_depth', 3, 10),
                'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), 0),
                'l2_leaf_reg': hp.loguniform('l2_leaf_reg', 0, np.log(10)),
                'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
                'leaf_estimation_iterations': hp.quniform('leaf_estimation_iterations', 1, 10, 1),
            }
            default_config = dict(n_estimators=2000)
            max_config['max_depth'] = 10
        config = utils.update_dict(default_config, config)
        super().__init__(hyper_optimizer=HyperoptOptimizer(space=space, fixed_params=dict(),
                                                           n_hyperopt_steps=n_hyperopt_steps,
                                                           **config),
                         max_resource_config=utils.join_dicts(config, max_config),
                         **config)

    def create_alg_interface(self, n_sub_splits: int, **config) -> AlgInterface:
        return SingleSplitWrapperAlgInterface([CatBoostSubSplitInterface(**config) for i in range(n_sub_splits)])


class RandomParamsCatBoostAlgInterface(RandomParamsAlgInterface):
    def _sample_params(self, is_classification: bool, seed: int, n_train: int):
        rng = np.random.default_rng(seed)
        # adapted from Shwartz-Ziv et al.
        hpo_space_name = self.config.get('hpo_space_name', 'shwartz-ziv')
        if hpo_space_name == 'shwartz-ziv':
            space = {
                'learning_rate': np.exp(rng.uniform(-5, 0)),
                'random_strength': rng.integers(1, 20, endpoint=True),
                'one_hot_max_size': rng.integers(0, 25, endpoint=True),
                'l2_leaf_reg': np.exp(rng.uniform(0, np.log(10))),
                'bagging_temperature': rng.uniform(0, 1),
                'leaf_estimation_iterations': rng.integers(1, 20, endpoint=True),
                'n_estimators': 1000,
                'max_depth': 6
            }
        elif hpo_space_name == 'large':
            # todo: there should be no harm in tuning nan_mode in ['Min', 'Max']
            space = {
                'boosting_type': 'Plain',  # avoid Ordered as the default on GPU
                'n_estimators': 1000,
                'learning_rate': np.exp(rng.uniform(np.log(5e-3), np.log(5e-1))),

                # bootstrap
                'bootstrap_type': rng.choice(['Bayesian', 'Bernoulli']),  # todo: could do more
                'bagging_temperature': rng.uniform(0, 4),  # can only be used with Bayesian
                'subsample': rng.uniform(0.5, 1.0),  # can only be used with Bernoulli (or Poisson)!

                # PerTreeLevel not supported for Lossguide
                # 'sampling_frequency': rng.choice(['PerTree', 'PerTreeLevel']),  # CPU only!

                'grow_policy': rng.choice(['SymmetricTree', 'Depthwise', 'Lossguide']),
                'max_depth': rng.integers(1, 10, endpoint=True),  # todo: support more for Lossguide
                'min_data_in_leaf': round(np.exp(rng.uniform(np.log(1.0), np.log(128.0)))),  # only for Depthwise and Lossguide
                'max_leaves': round(np.exp(rng.uniform(np.log(4.0), np.log(128.0)))),  # only for Lossguide

                'colsample_bylevel': rng.uniform(0.5, 1.0),
                'random_strength': rng.uniform(0.0, 20.0),  # todo: make log-uniform?

                'l2_leaf_reg': np.exp(rng.uniform(np.log(1e-4), np.log(20))),
                'leaf_estimation_iterations': round(np.exp(rng.uniform(np.log(1.0), np.log(20.0)))),

                # categorical features
                'one_hot_max_size': rng.integers(2, 128, endpoint=True),
                'model_size_reg': np.exp(rng.uniform(np.log(1e-1), np.log(2e0))),
                'max_ctr_complexity': rng.integers(1, 5, endpoint=True),
            }
        elif hpo_space_name == 'large-v2':
            # slightly shrunk version of large
            # todo: there should be no harm in tuning nan_mode in ['Min', 'Max']
            space = {
                'boosting_type': 'Plain',  # avoid Ordered as the default on GPU
                'n_estimators': 1000,
                'learning_rate': np.exp(rng.uniform(np.log(3e-2), np.log(8e-2))),  # shrunk

                # bootstrap
                'bootstrap_type': 'Bernoulli',  # shrunk
                # 'bagging_temperature': rng.uniform(0, 4),  # can only be used with Bayesian
                'subsample': rng.uniform(0.5, 1.0),  # can only be used with Bernoulli (or Poisson)!

                # PerTreeLevel not supported for Lossguide
                # 'sampling_frequency': rng.choice(['PerTree', 'PerTreeLevel']),  # CPU only!

                'grow_policy': rng.choice(['SymmetricTree', 'Depthwise', 'Lossguide']),  # todo: could shrink
                'max_depth': rng.integers(2, 10, endpoint=True),  # todo: support more for Lossguide
                'min_data_in_leaf': round(np.exp(rng.uniform(np.log(1.0), np.log(128.0)))),  # only for Depthwise and Lossguide
                'max_leaves': round(np.exp(rng.uniform(np.log(4.0), np.log(128.0)))),  # only for Lossguide

                'colsample_bylevel': rng.uniform(0.5, 1.0),
                'random_strength': rng.uniform(0.0, 20.0),  # todo: make log-uniform?

                'l2_leaf_reg': np.exp(rng.uniform(np.log(1e-4), np.log(20))),
                'leaf_estimation_iterations': round(np.exp(rng.uniform(np.log(1.0), np.log(20.0)))),

                # categorical features
                'one_hot_max_size': rng.integers(2, 128, endpoint=True),
                'model_size_reg': np.exp(rng.uniform(np.log(1e-1), np.log(2e0))),
                'max_ctr_complexity': rng.integers(2, 5, endpoint=True),  # shrunk
            }
        elif hpo_space_name == 'large-v3':
            # slightly shrunk version of large-v2 (shrunk random_strength and max_depth, colsample_bylevel, model_size_reg)
            # avoided removing lossguide for now
            # todo: there should be no harm in tuning nan_mode in ['Min', 'Max']
            space = {
                'boosting_type': 'Plain',  # avoid Ordered as the default on GPU
                'n_estimators': 1000,
                'learning_rate': np.exp(rng.uniform(np.log(3e-2), np.log(8e-2))),  # shrunk

                # bootstrap
                'bootstrap_type': 'Bernoulli',  # shrunk
                # 'bagging_temperature': rng.uniform(0, 4),  # can only be used with Bayesian
                'subsample': rng.uniform(0.5, 1.0),  # can only be used with Bernoulli (or Poisson)!

                # PerTreeLevel not supported for Lossguide
                # 'sampling_frequency': rng.choice(['PerTree', 'PerTreeLevel']),  # CPU only!

                'grow_policy': rng.choice(['SymmetricTree', 'Depthwise', 'Lossguide']),  # todo: could shrink
                'max_depth': rng.integers(4, 10, endpoint=True),  # todo: support more for Lossguide
                'min_data_in_leaf': round(np.exp(rng.uniform(np.log(1.0), np.log(128.0)))),  # only for Depthwise and Lossguide
                'max_leaves': round(np.exp(rng.uniform(np.log(4.0), np.log(128.0)))),  # only for Lossguide

                'colsample_bylevel': rng.uniform(0.6, 1.0),
                'random_strength': rng.uniform(0.0, 2.0),  # shrunk

                'l2_leaf_reg': np.exp(rng.uniform(np.log(1e-4), np.log(20))),
                'leaf_estimation_iterations': round(np.exp(rng.uniform(np.log(1.0), np.log(20.0)))),

                # categorical features
                'one_hot_max_size': rng.integers(2, 128, endpoint=True),  # todo: make logarithmic?
                'model_size_reg': np.exp(rng.uniform(np.log(1e-1), np.log(1.5))),
                'max_ctr_complexity': rng.integers(2, 5, endpoint=True),  # shrunk
            }
        elif hpo_space_name == 'large-v4':
            # slightly shrunk version of large-v3:
            # removed Lossguide -> also removed max_leaves
            # shrunk colsample_bylevel, min_data_in_leaf, one_hot_max_size
            # todo: there should be no harm in tuning nan_mode in ['Min', 'Max']
            space = {
                'boosting_type': 'Plain',  # avoid Ordered as the default on GPU
                'n_estimators': 1000,
                'learning_rate': np.exp(rng.uniform(np.log(3e-2), np.log(8e-2))),  # shrunk

                # bootstrap
                'bootstrap_type': 'Bernoulli',  # shrunk
                # 'bagging_temperature': rng.uniform(0, 4),  # can only be used with Bayesian
                'subsample': rng.uniform(0.7, 1.0),  # can only be used with Bernoulli (or Poisson)!

                # PerTreeLevel not supported for Lossguide
                # 'sampling_frequency': rng.choice(['PerTree', 'PerTreeLevel']),  # CPU only!

                'grow_policy': rng.choice(['SymmetricTree', 'Depthwise']),
                'max_depth': rng.integers(4, 10, endpoint=True),
                'min_data_in_leaf': round(np.exp(rng.uniform(np.log(1.0), np.log(100.0)))),  # only for Depthwise and Lossguide

                'colsample_bylevel': rng.uniform(0.85, 1.0),
                'random_strength': rng.uniform(0.0, 2.0),  # shrunk

                'l2_leaf_reg': np.exp(rng.uniform(np.log(1e-4), np.log(20))),
                'leaf_estimation_iterations': round(np.exp(rng.uniform(np.log(1.0), np.log(20.0)))),

                # categorical features
                'one_hot_max_size': rng.integers(8, 128, endpoint=True),
                'model_size_reg': np.exp(rng.uniform(np.log(1e-1), np.log(1.5))),
                'max_ctr_complexity': rng.integers(2, 5, endpoint=True),  # shrunk
            }
        elif hpo_space_name == 'large-v5':
            # large-v4 but with max_depth <= 8 as in tabrepo1
            space = {
                'boosting_type': 'Plain',  # avoid Ordered as the default on GPU
                'n_estimators': 1000,
                'learning_rate': np.exp(rng.uniform(np.log(3e-2), np.log(8e-2))),  # shrunk

                # bootstrap
                'bootstrap_type': 'Bernoulli',  # shrunk
                # 'bagging_temperature': rng.uniform(0, 4),  # can only be used with Bayesian
                'subsample': rng.uniform(0.7, 1.0),  # can only be used with Bernoulli (or Poisson)!

                # PerTreeLevel not supported for Lossguide
                # 'sampling_frequency': rng.choice(['PerTree', 'PerTreeLevel']),  # CPU only!

                'grow_policy': rng.choice(['SymmetricTree', 'Depthwise']),
                'max_depth': rng.integers(4, 8, endpoint=True),
                'min_data_in_leaf': round(np.exp(rng.uniform(np.log(1.0), np.log(100.0)))),  # only for Depthwise and Lossguide

                'colsample_bylevel': rng.uniform(0.85, 1.0),
                'random_strength': rng.uniform(0.0, 2.0),  # shrunk

                'l2_leaf_reg': np.exp(rng.uniform(np.log(1e-4), np.log(20))),
                'leaf_estimation_iterations': round(np.exp(rng.uniform(np.log(1.0), np.log(20.0)))),

                # categorical features
                'one_hot_max_size': rng.integers(8, 128, endpoint=True),
                'model_size_reg': np.exp(rng.uniform(np.log(1e-1), np.log(1.5))),
                'max_ctr_complexity': rng.integers(2, 5, endpoint=True),  # shrunk
            }
        elif hpo_space_name == 'large-v6':
            # large-v5 but with tabrepo lr search space
            space = {
                'boosting_type': 'Plain',  # avoid Ordered as the default on GPU
                'n_estimators': 1000,
                'learning_rate': np.exp(rng.uniform(np.log(5e-3), np.log(1e-1))),  # shrunk

                # bootstrap
                'bootstrap_type': 'Bernoulli',  # shrunk
                # 'bagging_temperature': rng.uniform(0, 4),  # can only be used with Bayesian
                'subsample': rng.uniform(0.7, 1.0),  # can only be used with Bernoulli (or Poisson)!

                # PerTreeLevel not supported for Lossguide
                # 'sampling_frequency': rng.choice(['PerTree', 'PerTreeLevel']),  # CPU only!

                'grow_policy': rng.choice(['SymmetricTree', 'Depthwise']),
                'max_depth': rng.integers(4, 8, endpoint=True),
                'min_data_in_leaf': round(np.exp(rng.uniform(np.log(1.0), np.log(100.0)))),  # only for Depthwise and Lossguide

                'colsample_bylevel': rng.uniform(0.85, 1.0),
                'random_strength': rng.uniform(0.0, 2.0),  # shrunk

                'l2_leaf_reg': np.exp(rng.uniform(np.log(1e-4), np.log(20))),
                'leaf_estimation_iterations': round(np.exp(rng.uniform(np.log(1.0), np.log(20.0)))),

                # categorical features
                'one_hot_max_size': rng.integers(8, 128, endpoint=True),
                'model_size_reg': np.exp(rng.uniform(np.log(1e-1), np.log(1.5))),
                'max_ctr_complexity': rng.integers(2, 5, endpoint=True),  # shrunk
            }
        elif hpo_space_name == 'large-v7':
            # large-v5 but with early_stopping_rounds=50
            space = {
                'boosting_type': 'Plain',  # avoid Ordered as the default on GPU
                'n_estimators': 1000,
                'early_stopping_rounds': 50,
                'max_bin': 254,  # added this to be sure
                'learning_rate': np.exp(rng.uniform(np.log(3e-2), np.log(8e-2))),  # shrunk

                # bootstrap
                'bootstrap_type': 'Bernoulli',  # shrunk
                # 'bagging_temperature': rng.uniform(0, 4),  # can only be used with Bayesian
                'subsample': rng.uniform(0.7, 1.0),  # can only be used with Bernoulli (or Poisson)!

                # PerTreeLevel not supported for Lossguide
                # 'sampling_frequency': rng.choice(['PerTree', 'PerTreeLevel']),  # CPU only!

                'grow_policy': rng.choice(['SymmetricTree', 'Depthwise']),
                'max_depth': rng.integers(4, 8, endpoint=True),
                'min_data_in_leaf': round(np.exp(rng.uniform(np.log(1.0), np.log(100.0)))),  # only for Depthwise and Lossguide

                'colsample_bylevel': rng.uniform(0.85, 1.0),
                'random_strength': rng.uniform(0.0, 2.0),  # shrunk

                'l2_leaf_reg': np.exp(rng.uniform(np.log(1e-4), np.log(20))),
                'leaf_estimation_iterations': round(np.exp(rng.uniform(np.log(1.0), np.log(20.0)))),

                # categorical features
                'one_hot_max_size': rng.integers(8, 128, endpoint=True),
                'model_size_reg': np.exp(rng.uniform(np.log(1e-1), np.log(1.5))),
                'max_ctr_complexity': rng.integers(2, 5, endpoint=True),  # shrunk
            }
        elif hpo_space_name == 'large-v8-10k':
            # large-v7 but with 10k estimators and the tabrepo1 lr search space
            space = {
                'boosting_type': 'Plain',  # avoid Ordered as the default on GPU
                'n_estimators': 10_000,
                'early_stopping_rounds': 50,
                'max_bin': 254,  # added this to be sure
                'learning_rate': np.exp(rng.uniform(np.log(5e-3), np.log(1e-1))),

                # bootstrap
                'bootstrap_type': 'Bernoulli',  # shrunk
                # 'bagging_temperature': rng.uniform(0, 4),  # can only be used with Bayesian
                'subsample': rng.uniform(0.7, 1.0),  # can only be used with Bernoulli (or Poisson)!

                # PerTreeLevel not supported for Lossguide
                # 'sampling_frequency': rng.choice(['PerTree', 'PerTreeLevel']),  # CPU only!

                'grow_policy': rng.choice(['SymmetricTree', 'Depthwise']),
                'max_depth': rng.integers(4, 8, endpoint=True),
                'min_data_in_leaf': round(np.exp(rng.uniform(np.log(1.0), np.log(100.0)))),  # only for Depthwise and Lossguide

                'colsample_bylevel': rng.uniform(0.85, 1.0),
                'random_strength': rng.uniform(0.0, 2.0),  # shrunk

                'l2_leaf_reg': np.exp(rng.uniform(np.log(1e-4), np.log(20))),
                'leaf_estimation_iterations': round(np.exp(rng.uniform(np.log(1.0), np.log(20.0)))),

                # categorical features
                'one_hot_max_size': rng.integers(8, 128, endpoint=True),
                'model_size_reg': np.exp(rng.uniform(np.log(1e-1), np.log(1.5))),
                'max_ctr_complexity': rng.integers(2, 5, endpoint=True),  # shrunk
            }
        elif hpo_space_name == 'large-v9-10k':
            # large-v8-10k but without tuning random_strength
            space = {
                'boosting_type': 'Plain',  # avoid Ordered as the default on GPU
                'n_estimators': 10_000,
                'early_stopping_rounds': 50,
                'max_bin': 254,  # added this to be sure
                'learning_rate': np.exp(rng.uniform(np.log(5e-3), np.log(1e-1))),

                # bootstrap
                'bootstrap_type': 'Bernoulli',  # shrunk
                # 'bagging_temperature': rng.uniform(0, 4),  # can only be used with Bayesian
                'subsample': rng.uniform(0.7, 1.0),  # can only be used with Bernoulli (or Poisson)!

                # PerTreeLevel not supported for Lossguide
                # 'sampling_frequency': rng.choice(['PerTree', 'PerTreeLevel']),  # CPU only!

                'grow_policy': rng.choice(['SymmetricTree', 'Depthwise']),
                'max_depth': rng.integers(4, 8, endpoint=True),
                'min_data_in_leaf': round(np.exp(rng.uniform(np.log(1.0), np.log(100.0)))),  # only for Depthwise and Lossguide

                'colsample_bylevel': rng.uniform(0.85, 1.0),

                'l2_leaf_reg': np.exp(rng.uniform(np.log(1e-4), np.log(20))),
                'leaf_estimation_iterations': round(np.exp(rng.uniform(np.log(1.0), np.log(20.0)))),

                # categorical features
                'one_hot_max_size': rng.integers(8, 128, endpoint=True),
                'model_size_reg': np.exp(rng.uniform(np.log(1e-1), np.log(1.5))),
                'max_ctr_complexity': rng.integers(2, 5, endpoint=True),  # shrunk
            }
        elif hpo_space_name == 'tabrepo1':
            space = {
                'boosting_type': 'Plain',  # avoid Ordered as the default on GPU
                'learning_rate': np.exp(rng.uniform(np.log(5e-3), np.log(1e-1))),
                'max_depth': rng.integers(4, 8, endpoint=True),
                'l2_leaf_reg': rng.uniform(1.0, 5.0),
                'max_ctr_complexity': rng.integers(1, 5, endpoint=True),
                'one_hot_max_size': rng.choice([2, 3, 5, 10]),
                'grow_policy': rng.choice(['SymmetricTree', 'Depthwise']),
            }
        elif hpo_space_name == 'tabrepo1-es':
            space = {
                'boosting_type': 'Plain',  # avoid Ordered as the default on GPU
                'early_stopping_rounds': 50,
                'learning_rate': np.exp(rng.uniform(np.log(5e-3), np.log(1e-1))),
                'max_depth': rng.integers(4, 8, endpoint=True),
                'l2_leaf_reg': rng.uniform(1.0, 5.0),
                'max_ctr_complexity': rng.integers(1, 5, endpoint=True),
                'one_hot_max_size': rng.choice([2, 3, 5, 10]),
                'grow_policy': rng.choice(['SymmetricTree', 'Depthwise']),
            }
        elif hpo_space_name == 'tabrepo1-es-10k':
            space = {
                'boosting_type': 'Plain',  # avoid Ordered as the default on GPU
                'early_stopping_rounds': 50,
                'n_estimators': 10_000,
                'learning_rate': np.exp(rng.uniform(np.log(5e-3), np.log(1e-1))),
                'max_depth': rng.integers(4, 8, endpoint=True),
                'l2_leaf_reg': rng.uniform(1.0, 5.0),
                'max_ctr_complexity': rng.integers(1, 5, endpoint=True),
                'one_hot_max_size': rng.choice([2, 3, 5, 10]),
                'grow_policy': rng.choice(['SymmetricTree', 'Depthwise']),
            }
        elif hpo_space_name == 'tabarena':
            space = {
                'n_estimators': 10_000,
                'early_stopping_rounds': 300,  # probably not exactly equivalent to TabArena
                'learning_rate': np.exp(rng.uniform(np.log(5e-3), np.log(1e-1))),

                'bootstrap_type': 'Bernoulli',
                'subsample': rng.uniform(0.7, 1.0),  # can only be used with Bernoulli (or Poisson)!

                'grow_policy': rng.choice(['SymmetricTree', 'Depthwise']),
                'max_depth': rng.integers(4, 8, endpoint=True),

                'colsample_bylevel': rng.uniform(0.85, 1.0),
                'l2_leaf_reg': np.exp(rng.uniform(np.log(1e-4), np.log(5.0))),

                'leaf_estimation_iterations': np.floor(np.exp(rng.uniform(np.log(1.0), np.log(21.0)))),

                # categorical features
                'one_hot_max_size': np.floor(np.exp(rng.uniform(np.log(8.0), np.log(101.0)))),
                'model_size_reg': np.exp(rng.uniform(np.log(0.1), np.log(1.5))),
                'max_ctr_complexity': rng.integers(2, 5, endpoint=True),  # shrunk

                'boosting_type': 'Plain',  # avoid Ordered as the default on GPU
                'max_bin': 254,  # added this to be sure
            }
        else:
            raise ValueError()
        return space

    def _create_interface_from_config(self, n_tv_splits: int, **config):
        return SingleSplitWrapperAlgInterface([CatBoostSubSplitInterface(**config) for i in range(n_tv_splits)])
