from typing import Optional, Tuple, Any, Dict

import numpy as np

from pytabkit.bench.alg_wrappers.interface_wrappers import LGBMInterfaceWrapper, XGBInterfaceWrapper, \
    CatBoostInterfaceWrapper
from pytabkit.bench.data.common import SplitType
from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskDescription, TaskCollection
from pytabkit.bench.eval.evaluation import FunctionAlgFilter, MultiResultsTable, DefaultEvalModeSelector, \
    MeanTableAnalyzer
from pytabkit.bench.run.task_execution import RunConfig, TabBenchJobManager
from pytabkit.bench.scheduling.schedulers import SimpleJobScheduler
from pytabkit.models import utils
from pytabkit.models.hyper_opt.coord_opt import Hyperparameter, CoordOptimizer
from pytabkit.models.hyper_opt.hyper_optimizers import HyperoptOptimizer, SMACOptimizer, f_unpack_dict
from pytabkit.bench.scheduling.execution import RayJobManager
from pytabkit.models.nn_models.categorical import EncodingFactory, SingleOrdinalEncodingFactory
from pytabkit.models.training.logging import StdoutLogger


def load_score(alg_name: Optional[str] = None, coll_name: str = 'meta-train-class', n_cv: int = 1,
               val_metric_name: Optional[str] = None, test_metric_name: Optional[str] = None,
               split_type: str = SplitType.RANDOM, use_task_weighting: bool = True,
               data_path: Optional[str] = None) -> Tuple[float, Any]:
    paths = Paths(data_path) if data_path is not None else Paths.from_env_variables()
    if '/' in coll_name:
        # use a single task
        parts = coll_name.split('/')
        if len(parts) != 2:
            raise ValueError(f'Too many / in coll_name {coll_name}')
        task_collection = TaskCollection(coll_name, [TaskDescription(*parts)])
    else:
        task_collection = TaskCollection.from_name(coll_name, paths)
    # print('load table')
    # table = MultiResultsTable.load_summaries(task_collection, n_cv=n_cv, paths=paths)
    alg_filter = FunctionAlgFilter(lambda an, tags, aw: an == alg_name)
    table = MultiResultsTable.load(task_collection, n_cv=n_cv, paths=paths, split_type=split_type, alg_filter=alg_filter)
    # print('process table')
    test_table = table.get_test_results_table(DefaultEvalModeSelector(), alg_group_dict={},
                                              val_metric_name=val_metric_name,
                                              test_metric_name=test_metric_name)
    analyzer = MeanTableAnalyzer(f=lambda x: np.log(x + 1e-2) - np.log(1e-2), use_weighting=use_task_weighting)
    means = analyzer.get_means(test_table)
    print(f'Mean scores for {alg_name}: {means}')
    return means[0], None


class AlgConfigRunner:
    def __init__(self, paths: Paths, coll_name: str, create_wrapper, base_name: str, tag: Optional[str] = None,
                 short_key_map: Dict[str, str] = None, **default_params):
        self.paths = paths
        self.coll_name = coll_name
        self.create_wrapper = create_wrapper
        self.base_name = base_name
        self.tag = tag or base_name
        self.default_params = default_params
        self.short_key_map = short_key_map or {}

    def __call__(self, config):
        config = f_unpack_dict(config)
        print(f'HPO config: {config}')
        # compute alg_name, potentially round config arguments
        alg_name_parts = [self.base_name]
        rounded_config = {}
        for key, value in config.items():
            if key in self.short_key_map:
                short_key = self.short_key_map[key]
            else:
                short_key = key

            if isinstance(value, float):
                alg_name_parts.append(f'{short_key}-{value:g}')
                rounded_config[key] = float(f'{value:g}')
            else:
                alg_name_parts.append(f'{short_key}-{value}')
                rounded_config[key] = value
        alg_name = '_'.join(alg_name_parts)

        try:
            # if already computed, return the computed result
            return load_score(alg_name, self.coll_name)
        except IndexError:
            pass

        # call wrapper with alg_name, tag, default_params and config
        wrapper = self.create_wrapper(**utils.join_dicts(self.default_params, config))

        # run on task_infos
        task_infos = TaskCollection.from_name(self.coll_name, self.paths).load_infos(self.paths)
        job_mgr = TabBenchJobManager(self.paths)
        scheduler = SimpleJobScheduler(RayJobManager())
        config_10_1_0 = RunConfig(n_tt_splits=10, n_cv=1, n_refit=0)
        job_mgr.add_jobs(task_infos, config_10_1_0, alg_name, wrapper, tags=[self.tag])
        job_mgr.run_jobs(scheduler)
        # load result
        return load_score(alg_name, self.coll_name)


def test_hyperopt_seed():
    from hyperopt import hp
    space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(5e-3), np.log(3e-1)),
        'num_leaves': hp.qloguniform('num_leaves', np.log(7), np.log(256), 1),
        'feature_fraction': hp.uniform('feature_fraction', 0.3, 1),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.3, 1),
        'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', 0, 6, 1),
        'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', -16, 5),
        'lambda_l1': hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)]),
        'lambda_l2': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)]),
    }
    fixed_params = {
        'n_estimators': 1000,
        'bagging_freq': 1
    }
    opt = HyperoptOptimizer(space, fixed_params, n_hyperopt_steps=100, hyperopt_algo='tpe')
    def print_params(params):
        print(params)
        return 0.0, None
    opt.optimize(print_params, seed=1234, opt_desc='LGBM-tuning-1', logger=StdoutLogger(verbosity_level=1))


def run_lgbm_train_class():
    short_key_map = dict(n_estimators='nest', bagging_freq='bfreq', learning_rate='lr', num_leaves='nl',
                         feature_fraction='ff', bagging_fraction='bfrac', min_data_in_leaf='mdil',
                         min_sum_hessian_in_leaf='mshil', lambda_l1='ll1', lambda_l2='ll2')

    acr = AlgConfigRunner(paths=Paths.from_env_variables(),
                          coll_name='train-class',
                          create_wrapper=LGBMInterfaceWrapper,
                          base_name='LGBM-tuning-1',
                          short_key_map=short_key_map)
    from hyperopt import hp
    space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(5e-3), np.log(3e-1)),
        'num_leaves': hp.qloguniform('num_leaves', np.log(7), np.log(256), 1),
        'feature_fraction': hp.uniform('feature_fraction', 0.3, 1),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.3, 1),
        'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', 0, 6, 1),
        'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', -16, 5),
        'lambda_l1': hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)]),
        'lambda_l2': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)]),
    }
    fixed_params = {
        'n_estimators': 1000,
        'bagging_freq': 1
    }
    opt = HyperoptOptimizer(space, fixed_params, n_hyperopt_steps=100, hyperopt_algo='tpe')
    opt.optimize(acr, seed=1234, opt_desc='LGBM-tuning-1', logger=StdoutLogger(verbosity_level=1))


def run_lgbm_train_class_smac():
    short_key_map = dict(n_estimators='nest', bagging_freq='bfreq', learning_rate='lr', num_leaves='nl',
                         feature_fraction='ff', bagging_fraction='bfrac', min_data_in_leaf='mdil',
                         min_sum_hessian_in_leaf='mshil', lambda_l1='ll1', lambda_l2='ll2')

    acr = AlgConfigRunner(paths=Paths.from_env_variables(),
                          coll_name='meta-train-class',
                          create_wrapper=LGBMInterfaceWrapper,
                          base_name='LGBM-tuning-smac-1',
                          short_key_map=short_key_map)
    from ConfigSpace import Float, Integer, ConfigurationSpace
    space = ConfigurationSpace()
    space.add_hyperparameters([
        Float('learning_rate', (5e-3, 3e-1), log=True),
        Integer('num_leaves', (7, 256), log=True),
        Float('feature_fraction', (0.3, 1)),
        Float('bagging_fraction', (0.3, 1)),
        Integer('min_data_in_leaf', (1, 64), log=True),
        Float('min_sum_hessian_in_leaf', (np.exp(-16), np.exp(5)), log=True),
        Float('lambda_l1', (np.exp(-16), np.exp(2)), log=True),
        Float('lambda_l2', (np.exp(-16), np.exp(2)), log=True),
    ])
    fixed_params = {
        'n_estimators': 1000,
        'bagging_freq': 1
    }
    paths = Paths.from_env_variables()
    with paths.new_tmp_folder() as tmp_folder:
        opt = SMACOptimizer(space, fixed_params, n_hyperopt_steps=100, tmp_folder=tmp_folder)
        opt.optimize(acr, seed=1234, opt_desc='LGBM-tuning-smac-1', logger=StdoutLogger(verbosity_level=1))


def run_lgbm_train_class_smac_2(use_reg: bool = False):
    base_name = 'LGBM-tuning-smac-2-reg' if use_reg else 'LGBM-tuning-smac-2'
    short_key_map = dict(n_estimators='nest', bagging_freq='bfreq', learning_rate='lr', num_leaves='nl',
                         feature_fraction='ff', bagging_fraction='bfrac', min_data_in_leaf='mdil',
                         min_sum_hessian_in_leaf='mshil', lambda_l1='ll1', lambda_l2='ll2')

    acr = AlgConfigRunner(paths=Paths.from_env_variables(),
                          coll_name='meta-train-reg' if use_reg else 'meta-train-class',
                          create_wrapper=LGBMInterfaceWrapper,
                          base_name=base_name,
                          short_key_map=short_key_map)
    from ConfigSpace import Float, Integer, ConfigurationSpace
    space = ConfigurationSpace()
    space.add_hyperparameters([
        Float('learning_rate', (2e-2, 1e-1), log=True, default=6e-2),
        Integer('num_leaves', (16, 64), log=True, default=31),
        Float('feature_fraction', (0.5, 1), default=0.75),
        Float('bagging_fraction', (0.5, 1), default=0.75),
        Integer('min_data_in_leaf', (1, 64), log=True, default=5),
        Float('min_sum_hessian_in_leaf', (1e-7, 1e-2), log=True, default=1e-5),
        Float('lambda_l1', (1e-7, 1e-3), log=True, default=1e-7),
        Float('lambda_l2', (1e-7, 1e-3), log=True, default=1e-7),
    ])
    fixed_params = {
        'n_estimators': 1000,
        'bagging_freq': 1
    }
    paths = Paths.from_env_variables()
    with paths.new_tmp_folder() as tmp_folder:
        opt = SMACOptimizer(space, fixed_params, n_hyperopt_steps=100, tmp_folder=tmp_folder)
        opt.optimize(acr, seed=1234, opt_desc=base_name, logger=StdoutLogger(verbosity_level=1))


def run_lgbm_train_class_smac_3(use_reg: bool = False):
    base_name = 'LGBM-tuning-smac-3-reg' if use_reg else 'LGBM-tuning-smac-3'
    short_key_map = dict(n_estimators='nest', bagging_freq='bfreq', learning_rate='lr', num_leaves='nl',
                         feature_fraction='ff', bagging_fraction='bfrac', min_data_in_leaf='mdil',
                         min_sum_hessian_in_leaf='mshil', lambda_l1='ll1', lambda_l2='ll2')

    acr = AlgConfigRunner(paths=Paths.from_env_variables(),
                          coll_name='meta-train-reg' if use_reg else 'meta-train-class',
                          create_wrapper=LGBMInterfaceWrapper,
                          base_name=base_name,
                          short_key_map=short_key_map)
    from ConfigSpace import Float, Integer, ConfigurationSpace
    space = ConfigurationSpace()
    space.add_hyperparameters([
        Float('learning_rate', (2e-2, 1e-1), log=True, default=6e-2),
        Integer('num_leaves', (16, 128), log=True, default=31),  # larger max num_leaves than for smac-2
        Float('feature_fraction', (0.5, 1), default=0.75),
        Float('bagging_fraction', (0.5, 1), default=0.75),
        Integer('min_data_in_leaf', (1, 64), log=True, default=5),
        Float('min_sum_hessian_in_leaf', (1e-7, 1e-2), log=True, default=1e-5),
        Float('lambda_l1', (1e-7, 1e-3), log=True, default=1e-7),
        Float('lambda_l2', (1e-7, 1e-3), log=True, default=1e-7),
    ])
    fixed_params = {
        'n_estimators': 1000,
        'bagging_freq': 1
    }
    paths = Paths.from_env_variables()
    with paths.new_tmp_folder() as tmp_folder:
        opt = SMACOptimizer(space, fixed_params, n_hyperopt_steps=200 if use_reg else 100, tmp_folder=tmp_folder,
                            n_initial_design=25)
        opt.optimize(acr, seed=1234, opt_desc=base_name, logger=StdoutLogger(verbosity_level=1))


def run_lgbm_train_class_coord():
    base_name = 'LGBM-tuning-coord-1'
    short_key_map = dict(n_estimators='nest', bagging_freq='bfreq', learning_rate='lr', num_leaves='nl',
                         feature_fraction='ff', bagging_fraction='bfrac', min_data_in_leaf='mdil',
                         min_sum_hessian_in_leaf='mshil', lambda_l1='ll1', lambda_l2='ll2')

    acr = AlgConfigRunner(paths=Paths.from_env_variables(),
                          coll_name='meta-train-class',
                          create_wrapper=LGBMInterfaceWrapper,
                          base_name=base_name,
                          short_key_map=short_key_map)
    space = {
        'learning_rate': Hyperparameter(start_value=np.log(0.1), min_step_size=0.1, importance=1.0, log_scale=True),
        'num_leaves': Hyperparameter(np.log(31), 0.1, 0.2, log_scale=True, only_int=True),
        'feature_fraction': Hyperparameter(1.0, 0.01, 0.4, min_value=0.3, max_value=1.0),
        'bagging_fraction': Hyperparameter(1.0, 0.01, 0.4, min_value=0.3, max_value=1.0),
        'min_data_in_leaf': Hyperparameter(np.log(20), 0.1, 0.2, log_scale=True, only_int=True, max_value=np.log(128)),
        'min_sum_hessian_in_leaf': Hyperparameter(np.log(1e-3), 0.1, 0.6, log_scale=True),
        'lambda_l1': Hyperparameter(np.log(1e-5), 0.1, 0.2, log_scale=True, min_value=-16.0, max_value=2.0),
        'lambda_l2': Hyperparameter(np.log(1e-5), 0.1, 0.2, log_scale=True, min_value=-16.0, max_value=2.0),
    }
    fixed_params = {
        'n_estimators': 1000,
        'bagging_freq': 1
    }
    paths = Paths.from_env_variables()
    with paths.new_tmp_folder() as tmp_folder:
        opt = CoordOptimizer(space, fixed_params, n_hyperopt_steps=100, tmp_folder=tmp_folder)
        opt.optimize(acr, seed=1234, opt_desc=base_name, logger=StdoutLogger(verbosity_level=1))


def run_xgb_train_class_smac(use_reg: bool = False):
    # XGB-tuning-smac-1 accidentally used LightGBM
    base_name = 'XGB-tuning-smac-2-reg' if use_reg else 'XGB-tuning-smac-2'
    short_key_map = dict(n_estimators='nest', bagging_freq='bfreq', learning_rate='lr', num_leaves='nl',
                         colsample_bylevel='cbl', colsample_bytree='cbt', colsample_bynode='cbn',
                         max_depth='md', min_child_weight='mcw', reg_alpha='alph', reg_lambda='lam', reg_gamma='gam',
                         subsample='ss',
                         feature_fraction='ff', bagging_fraction='bfrac', min_data_in_leaf='mdil',
                         min_sum_hessian_in_leaf='mshil', lambda_l1='ll1', lambda_l2='ll2')

    oe_perm_factory = EncodingFactory(SingleOrdinalEncodingFactory(permute_ordinal_encoding=True))

    acr = AlgConfigRunner(paths=Paths.from_env_variables(),
                          coll_name='meta-train-reg' if use_reg else 'meta-train-class',
                          create_wrapper=lambda **kwargs: XGBInterfaceWrapper(factory=oe_perm_factory, **kwargs),
                          base_name=base_name,
                          short_key_map=short_key_map)
    from ConfigSpace import Float, Integer, ConfigurationSpace
    space = ConfigurationSpace()
    space.add_hyperparameters([
        Float('learning_rate', (2e-2, 1e-1), log=True, default=6e-2),
        Integer('max_depth', (4, 8), default=6),
        Float('subsample', (0.5, 1), default=0.75),
        Float('colsample_bytree', (0.6, 1), default=1.0),
        Float('colsample_bylevel', (0.6, 1), default=1.0),
        Float('colsample_bynode', (0.6, 1), default=1.0),
        Float('min_child_weight', (1e-7, 1e-2), log=True, default=1e-5),
        Float('reg_alpha', (1e-7, 1e-2), log=True, default=1e-7),
        Float('reg_lambda', (1e-7, 1e-2), log=True, default=1e-7),
        Float('reg_gamma', (1e-7, 1e-2), log=True, default=1e-7),
    ])
    fixed_params = {
        'n_estimators': 1000,
    }
    paths = Paths.from_env_variables()
    with paths.new_tmp_folder() as tmp_folder:
        opt = SMACOptimizer(space, fixed_params, n_hyperopt_steps=200 if use_reg else 100, tmp_folder=tmp_folder,
                            n_initial_design=25)
        opt.optimize(acr, seed=1234, opt_desc=base_name, logger=StdoutLogger(verbosity_level=1))


def run_xgb_train_class_smac_3(use_reg: bool = False):
    # XGB-tuning-smac-1 accidentally used LightGBM
    base_name = 'XGB-tuning-smac-3-reg' if use_reg else 'XGB-tuning-smac-3'
    short_key_map = dict(n_estimators='nest', bagging_freq='bfreq', learning_rate='lr', num_leaves='nl',
                         colsample_bylevel='cbl', colsample_bytree='cbt', colsample_bynode='cbn',
                         max_depth='md', min_child_weight='mcw', reg_alpha='alph', reg_lambda='lam', reg_gamma='gam',
                         subsample='ss',
                         feature_fraction='ff', bagging_fraction='bfrac', min_data_in_leaf='mdil',
                         min_sum_hessian_in_leaf='mshil', lambda_l1='ll1', lambda_l2='ll2')

    oe_perm_factory = EncodingFactory(SingleOrdinalEncodingFactory(permute_ordinal_encoding=True))

    acr = AlgConfigRunner(paths=Paths.from_env_variables(),
                          coll_name='meta-train-reg' if use_reg else 'meta-train-class',
                          create_wrapper=lambda **kwargs: XGBInterfaceWrapper(factory=oe_perm_factory, **kwargs),
                          base_name=base_name,
                          short_key_map=short_key_map)
    from ConfigSpace import Float, Integer, ConfigurationSpace
    space = ConfigurationSpace()
    space.add_hyperparameters([
        Float('learning_rate', (2e-2, 1e-1), log=True, default=6e-2),
        Integer('max_depth', (4, 10), default=6),  # increased upper bound to 10
        Float('subsample', (0.5, 1), default=0.75),
        Float('colsample_bytree', (0.6, 1), default=1.0),
        Float('colsample_bylevel', (0.6, 1), default=1.0),
        Float('colsample_bynode', (0.6, 1), default=1.0),
        Float('min_child_weight', (1e-7, 1e-2), log=True, default=1e-5),
        Float('reg_alpha', (1e-7, 1e-2), log=True, default=1e-7),
        Float('reg_lambda', (1e-7, 1e-2), log=True, default=1e-7),
        Float('reg_gamma', (1e-7, 1e-2), log=True, default=1e-7),
    ])
    fixed_params = {
        'n_estimators': 1000,
    }
    paths = Paths.from_env_variables()
    with paths.new_tmp_folder() as tmp_folder:
        opt = SMACOptimizer(space, fixed_params, n_hyperopt_steps=200 if use_reg else 100, tmp_folder=tmp_folder,
                            n_initial_design=25)
        opt.optimize(acr, seed=1234, opt_desc=base_name, logger=StdoutLogger(verbosity_level=1))


def run_catboost_train_class_smac(use_reg: bool = False):
    base_name = 'CatBoost-tuning-smac-reg' if use_reg else 'CatBoost-tuning-smac'
    short_key_map = dict(n_estimators='nest', bagging_freq='bfreq', learning_rate='lr', num_leaves='nl',
                         colsample_bylevel='cbl', colsample_bytree='cbt', colsample_bynode='cbn',
                         max_depth='md', min_child_weight='mcw', reg_alpha='alph', reg_lambda='lam', reg_gamma='gam',
                         subsample='ss', l2_leaf_reg='l2lr', bagging_temperature='bt', random_strength='rs',
                         one_hot_max_size='ohms', leaf_estimation_iterations='lei',
                         feature_fraction='ff', bagging_fraction='bfrac', min_data_in_leaf='mdil',
                         min_sum_hessian_in_leaf='mshil', lambda_l1='ll1', lambda_l2='ll2')

    acr = AlgConfigRunner(paths=Paths.from_env_variables(),
                          coll_name='meta-train-reg' if use_reg else 'meta-train-class',
                          create_wrapper=CatBoostInterfaceWrapper,
                          base_name=base_name,
                          short_key_map=short_key_map)
    from ConfigSpace import Float, Integer, ConfigurationSpace
    space = ConfigurationSpace()
    space.add_hyperparameters([
        Float('learning_rate', (2e-2, 1e-1), log=True, default=6e-2),
        Integer('max_depth', (4, 10), default=8),  # increased upper bound to 10
        Float('l2_leaf_reg', (1e-7, 1e-2), log=True, default=1e-5),
        Float('bagging_temperature', (0.0, 1.0), default=1.0),
        Float('random_strength', (1e-2, 20.0), log=True, default=1.0),
        Integer('one_hot_max_size', (0, 25), default=10),
        Integer('leaf_estimation_iterations', (1, 20), default=1)
    ])
    # todo: also try min_child_samples?
    # todo: try boosting_type and bootstrap_type?  ("Bayesian", "Bernoulli", "MVS")
    # possibly subsample for other bootstrap_type?
    #  https://www.kaggle.com/code/saurabhshahane/catboost-hyperparameter-tuning-with-optuna/notebook
    fixed_params = {
        'n_estimators': 1000,
    }
    paths = Paths.from_env_variables()
    with paths.new_tmp_folder() as tmp_folder:
        opt = SMACOptimizer(space, fixed_params, n_hyperopt_steps=100, tmp_folder=tmp_folder,
                            n_initial_design=25)
        opt.optimize(acr, seed=1234, opt_desc=base_name, logger=StdoutLogger(verbosity_level=1))


def run_catboost_train_class_hyperopt(use_reg: bool = False):
    base_name = 'CatBoost-tuning-hyperopt-reg' if use_reg else 'CatBoost-tuning-hyperopt'
    short_key_map = dict(n_estimators='nest', bagging_freq='bfreq', learning_rate='lr', num_leaves='nl',
                         colsample_bylevel='cbl', colsample_bytree='cbt', colsample_bynode='cbn',
                         bootstrap_type='boot', boosting_type='boost',
                         max_depth='md', min_child_weight='mcw', reg_alpha='alph', reg_lambda='lam', reg_gamma='gam',
                         subsample='ss', l2_leaf_reg='l2lr', bagging_temperature='bt', random_strength='rs',
                         one_hot_max_size='ohms', leaf_estimation_iterations='lei',
                         feature_fraction='ff', bagging_fraction='bfrac', min_data_in_leaf='mdil',
                         min_sum_hessian_in_leaf='mshil', lambda_l1='ll1', lambda_l2='ll2')

    acr = AlgConfigRunner(paths=Paths.from_env_variables(),
                          coll_name='meta-train-reg' if use_reg else 'meta-train-class',
                          create_wrapper=CatBoostInterfaceWrapper,
                          base_name=base_name,
                          short_key_map=short_key_map)
    from hyperopt import hp
    space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(2e-2), np.log(2e-1)),
        'max_depth': hp.quniform('max_depth', 4, 10, 1),  # this was ignored due to an implementation error
        'l2_leaf_reg': hp.loguniform('l2_leaf_reg', np.log(1e-6), np.log(1e-2)),
        'random_strength': hp.loguniform('random_strength', np.log(1e-3), np.log(5.0)),
        'one_hot_max_size': hp.quniform('one_hot_max_size', 0, 25, 1),
        'leaf_estimation_iterations': hp.quniform('leaf_estimation_iterations', 1, 20, 1),
        'boosting_type': 'Plain', #hp.choice('boosting_type', ['Ordered', 'Plain']),
        'bootstrap_type': hp.choice('bootstrap_type', [
            {'bootstrap_type': 'Bayesian', 'bagging_temperature': hp.uniform('bagging_temperature', 0, 1)},
            {'bootstrap_type': 'Bernoulli', 'subsample': hp.uniform('subsample', 0.5, 1.0)}
        ]),
        'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', np.log(1.0), np.log(100.0), 1),
    }
    # todo: also try min_child_samples?
    # todo: try boosting_type and bootstrap_type?  ("Bayesian", "Bernoulli", "MVS")
    # possibly subsample for other bootstrap_type?
    #  https://www.kaggle.com/code/saurabhshahane/catboost-hyperparameter-tuning-with-optuna/notebook
    fixed_params = {
        'n_estimators': 1000,
    }
    paths = Paths.from_env_variables()
    with paths.new_tmp_folder() as tmp_folder:
        opt = HyperoptOptimizer(space, fixed_params, n_hyperopt_steps=100)
        opt.optimize(acr, seed=1234, opt_desc=base_name, logger=StdoutLogger(verbosity_level=1))


def run_catboost_train_class_hyperopt_2(use_reg: bool = False):
    base_name = 'CatBoost-tuning-hyperopt-2-reg' if use_reg else 'CatBoost-tuning-hyperopt-2'
    short_key_map = dict(n_estimators='nest', bagging_freq='bfreq', learning_rate='lr', num_leaves='nl',
                         colsample_bylevel='cbl', colsample_bytree='cbt', colsample_bynode='cbn',
                         bootstrap_type='boot', boosting_type='boost',
                         max_depth='md', min_child_weight='mcw', reg_alpha='alph', reg_lambda='lam', reg_gamma='gam',
                         subsample='ss', l2_leaf_reg='l2lr', bagging_temperature='bt', random_strength='rs',
                         one_hot_max_size='ohms', leaf_estimation_iterations='lei',
                         feature_fraction='ff', bagging_fraction='bfrac', min_data_in_leaf='mdil',
                         min_sum_hessian_in_leaf='mshil', lambda_l1='ll1', lambda_l2='ll2')

    acr = AlgConfigRunner(paths=Paths.from_env_variables(),
                          coll_name='meta-train-reg' if use_reg else 'meta-train-class',
                          create_wrapper=CatBoostInterfaceWrapper,
                          base_name=base_name,
                          short_key_map=short_key_map)
    from hyperopt import hp
    space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(2e-2), np.log(2e-1)),
        'max_depth': hp.quniform('max_depth', 4, 10, 1),  # this was ignored due to an implementation error
        'l2_leaf_reg': hp.loguniform('l2_leaf_reg', np.log(1e-6), np.log(1e-2)),
        'random_strength': hp.loguniform('random_strength', np.log(1e-3), np.log(5.0)),
        'one_hot_max_size': hp.quniform('one_hot_max_size', 0, 25, 1),
        'leaf_estimation_iterations': hp.quniform('leaf_estimation_iterations', 1, 20, 1),
        'boosting_type': 'Plain', #hp.choice('boosting_type', ['Ordered', 'Plain']),
        'bootstrap_type': hp.choice('bootstrap_type', [
            {'bootstrap_type': 'Bayesian', 'bagging_temperature': hp.uniform('bagging_temperature', 0, 1)},
            {'bootstrap_type': 'Bernoulli', 'subsample': hp.uniform('subsample', 0.5, 1.0)}
        ]),
        'min_data_in_leaf': hp.qloguniform('min_data_in_leaf', np.log(1.0), np.log(100.0), 1),
    }
    #  https://www.kaggle.com/code/saurabhshahane/catboost-hyperparameter-tuning-with-optuna/notebook
    fixed_params = {
        'n_estimators': 1000,
    }
    paths = Paths.from_env_variables()
    with paths.new_tmp_folder() as tmp_folder:
        opt = HyperoptOptimizer(space, fixed_params, n_hyperopt_steps=100)
        opt.optimize(acr, seed=1234, opt_desc=base_name, logger=StdoutLogger(verbosity_level=1))


def run_catboost_train_class_hyperopt_3(use_reg: bool = False):
    base_name = 'CatBoost-tuning-hyperopt-3-reg' if use_reg else 'CatBoost-tuning-hyperopt-3'
    short_key_map = dict(n_estimators='nest', bagging_freq='bfreq', learning_rate='lr', num_leaves='nl',
                         colsample_bylevel='cbl', colsample_bytree='cbt', colsample_bynode='cbn',
                         bootstrap_type='boot', boosting_type='boost',
                         max_depth='md', min_child_weight='mcw', reg_alpha='alph', reg_lambda='lam', reg_gamma='gam',
                         subsample='ss', l2_leaf_reg='l2lr', bagging_temperature='bt', random_strength='rs',
                         one_hot_max_size='ohms', leaf_estimation_iterations='lei',
                         feature_fraction='ff', bagging_fraction='bfrac', min_data_in_leaf='mdil',
                         min_sum_hessian_in_leaf='mshil', lambda_l1='ll1', lambda_l2='ll2')

    acr = AlgConfigRunner(paths=Paths.from_env_variables(),
                          coll_name='meta-train-reg' if use_reg else 'meta-train-class',
                          create_wrapper=CatBoostInterfaceWrapper,
                          base_name=base_name,
                          short_key_map=short_key_map)
    from hyperopt import hp
    space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(2e-2), np.log(2e-1)),
        'max_depth': hp.quniform('max_depth', 4, 10, 1),  # this was ignored due to an implementation error
        'l2_leaf_reg': hp.loguniform('l2_leaf_reg', np.log(1e-6), np.log(1e-2)),
        'random_strength': hp.loguniform('random_strength', np.log(1e-3), np.log(5.0)),
        'one_hot_max_size': hp.quniform('one_hot_max_size', 0, 25, 1),
        'leaf_estimation_iterations': hp.quniform('leaf_estimation_iterations', 1, 20, 1),
        'boosting_type': 'Plain', #hp.choice('boosting_type', ['Ordered', 'Plain']),
        'bootstrap_type': hp.choice('bootstrap_type', [
            {'bootstrap_type': 'Bayesian', 'bagging_temperature': hp.uniform('bagging_temperature', 0, 1)},
            {'bootstrap_type': 'Bernoulli', 'subsample': hp.uniform('subsample', 0.5, 1.0)}
        ]),  # removed min_data_in_leaf since it is not used with SymmetricTree
    }
    #  https://www.kaggle.com/code/saurabhshahane/catboost-hyperparameter-tuning-with-optuna/notebook
    fixed_params = {
        'n_estimators': 1000,
    }
    paths = Paths.from_env_variables()
    with paths.new_tmp_folder() as tmp_folder:
        opt = HyperoptOptimizer(space, fixed_params, n_hyperopt_steps=100)
        opt.optimize(acr, seed=1234, opt_desc=base_name, logger=StdoutLogger(verbosity_level=1))


if __name__ == '__main__':
    # load_score('NN-class-special-2', 'train-class')
    # run_lgbm_train_class()
    # run_lgbm_train_class_smac()
    # run_lgbm_train_class_smac_2()
    # run_lgbm_train_class_coord()
    # run_xgb_train_class_smac()
    # run_lgbm_train_class_smac_3(use_reg=True)
    # run_xgb_train_class_smac_3(use_reg=True)
    # run_catboost_train_class_smac(use_reg=True)
    # run_catboost_train_class_hyperopt_2(use_reg=True)
    # run_catboost_train_class_hyperopt_2(use_reg=False)
    run_catboost_train_class_hyperopt_3(use_reg=False)
    # test_hyperopt_seed()
    pass
