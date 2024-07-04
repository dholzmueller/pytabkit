import multiprocessing
import time
from typing import List, Dict, Any, Callable

import numpy as np
import sklearn
import torch

from sklearn.base import BaseEstimator

from pytabkit.bench.scheduling.execution import FunctionProcess
from pytabkit.models.alg_interfaces.resource_computation import UniformSampler, FeatureSpec, get_resource_features, \
    process_resource_features, \
    Sampler, ds_to_xy, fit_resource_factors, TimeWrapper
from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskDescription, TaskCollection
from pytabkit.models import utils
from pytabkit.models.data.data import DictDataset
from pytabkit.models.sklearn.sklearn_interfaces import CatBoost_TD_Classifier, XGB_TD_Classifier, LGBM_TD_Classifier


def get_param_grid(grids_1d: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    configs = [dict()]
    for key, values in grids_1d.items():
        configs = [utils.update_dict(c, {key: val}) for val in values for c in configs]
    return configs


def estimate_params(paths: Paths, exp_name: str, coll_name: str, estimator: BaseEstimator, is_lgbm: bool = False,
                    rerun: bool = False):
    if is_lgbm:
        # use num_leaves instead of max_depth
        learner_space = dict(
            n_estimators=UniformSampler(2, 2, log=True, is_int=True),
            n_threads=UniformSampler(4, 4, log=True, is_int=True),
            num_leaves=UniformSampler(10, 100, log=True, is_int=True),
        )
        time_feature_spec = FeatureSpec.concat('', 'ds_size_gb', 'ds_prep_size_gb', 'ds_onehot_size_gb',
                                               FeatureSpec.product('n_cv_refit', 'n_splits',
                                                                   ['', 'log_num_leaves', 'num_leaves'],
                                                                   'n_estimators', '1/n_threads',
                                                                   FeatureSpec.powerset_products('n_features',
                                                                                                 'n_samples',
                                                                                                 'n_tree_repeats')))
        ram_feature_spec = FeatureSpec.concat('', 'ds_size_gb', 'ds_prep_size_gb', 'ds_onehot_size_gb',
                                              FeatureSpec.product(['', 'log_num_leaves', 'num_leaves'],
                                                                  FeatureSpec.powerset_products('n_features',
                                                                                                'n_samples',
                                                                                                'n_tree_repeats')))
    else:
        learner_space = dict(
            n_estimators=UniformSampler(2, 2, log=True, is_int=True),
            n_threads=UniformSampler(4, 4, log=True, is_int=True),
            max_depth=UniformSampler(3, 10, is_int=True),
        )
        time_feature_spec = FeatureSpec.concat('', 'ds_size_gb', 'ds_prep_size_gb', 'ds_onehot_size_gb',
                                               FeatureSpec.product('n_cv_refit', 'n_splits',
                                                                   ['', 'max_depth', '2_power_maxdepth'],
                                                                   'n_estimators', '1/n_threads',
                                                                   FeatureSpec.powerset_products('n_features',
                                                                                                 'n_samples',
                                                                                                 'n_tree_repeats')))
        ram_feature_spec = FeatureSpec.concat('', 'ds_size_gb', 'ds_prep_size_gb', 'ds_onehot_size_gb',
                                              FeatureSpec.product(['', 'max_depth', '2_power_maxdepth'],
                                                                  FeatureSpec.powerset_products('n_features',
                                                                                                'n_samples',
                                                                                                'n_tree_repeats')))

    coefs = calibrate_resources(exp_name, paths=paths, learner_space=learner_space,
                                coll_name=coll_name,
                                time_feature_spec=time_feature_spec,
                                ram_feature_spec=ram_feature_spec, sklearn_learner=estimator, n_combinations=300,
                                rerun=rerun)
    print(f'time_params={coefs["time_s"]}')
    print(f'cpu_ram_params={coefs["ram_gb"]}')

    ram_params = coefs['ram_gb']
    time_params = coefs['time_s']

    print(f'Analyzing dionis:')
    task_info = TaskDescription('openml-class', 'dionis').load_info(paths)
    # task_info = TaskDescription('uci-bin-class', 'madelon').load_info(paths)
    ds = DictDataset(tensors=None, tensor_infos=task_info.tensor_infos, device='cpu', n_samples=task_info.n_samples)
    config = dict(n_estimators=1000, n_threads=4, max_depth=6, num_leaves=31)
    raw_features = get_resource_features(config, ds, n_cv=1, n_refit=0, n_splits=1)
    ram_features = process_resource_features(raw_features, ram_feature_spec)
    ram_gb = sum([ram_features[key] * ram_params[key] for key in ram_params])
    time_features = process_resource_features(raw_features, time_feature_spec)
    time_s = sum([time_features[key] * time_params[key] for key in time_params])
    print(f'{ram_gb=}, {time_s=}')


def estimate_params_new(paths: Paths, exp_name: str, coll_name: str, estimator: BaseEstimator,
                        hparam_grid: List[Dict[str, Any]], short_name: str, is_lgbm: bool = False, rerun: bool = False):
    if is_lgbm:
        # use num_leaves instead of max_depth
        time_feature_spec = FeatureSpec.concat('', 'ds_size_gb', 'ds_prep_size_gb', 'ds_onehot_size_gb',
                                               FeatureSpec.product('n_cv_refit', 'n_splits',
                                                                   ['', 'log_num_leaves', 'num_leaves'],
                                                                   'n_estimators', '1/n_threads',
                                                                   FeatureSpec.powerset_products('n_features',
                                                                                                 'n_samples',
                                                                                                 'n_tree_repeats')))
        ram_feature_spec = FeatureSpec.concat('', 'ds_size_gb', 'ds_prep_size_gb', 'ds_onehot_size_gb',
                                              FeatureSpec.product(['', 'log_num_leaves', 'num_leaves'],
                                                                  FeatureSpec.powerset_products('n_features',
                                                                                                'n_samples',
                                                                                                'n_tree_repeats')))
    else:
        time_feature_spec = FeatureSpec.concat('', 'ds_size_gb', 'ds_prep_size_gb', 'ds_onehot_size_gb',
                                               FeatureSpec.product('n_cv_refit', 'n_splits',
                                                                   ['', 'max_depth', '2_power_maxdepth'],
                                                                   'n_estimators', '1/n_threads',
                                                                   FeatureSpec.powerset_products('n_features',
                                                                                                 'n_samples',
                                                                                                 'n_tree_repeats')))
        ram_feature_spec = FeatureSpec.concat('', 'ds_size_gb', 'ds_prep_size_gb', 'ds_onehot_size_gb',
                                              FeatureSpec.product(['', 'max_depth', '2_power_maxdepth'],
                                                                  FeatureSpec.powerset_products('n_features',
                                                                                                'n_samples',
                                                                                                'n_tree_repeats')))

    coefs = calibrate_resources_new_2(exp_name, paths=paths, hparam_grid=hparam_grid,
                                      coll_name=coll_name,
                                      time_feature_spec=time_feature_spec,
                                      ram_feature_spec=ram_feature_spec, sklearn_learner=estimator,
                                      rerun=rerun)
    print(f'{short_name}_time={coefs["time_s"]}')
    print(f'{short_name}_ram={coefs["ram_gb"]}')

    ram_params = coefs['ram_gb']
    time_params = coefs['time_s']

    print(f'Analyzing dionis:')
    task_info = TaskDescription('openml-class', 'dionis').load_info(paths)
    # task_info = TaskDescription('uci-bin-class', 'madelon').load_info(paths)
    ds = DictDataset(tensors=None, tensor_infos=task_info.tensor_infos, device='cpu', n_samples=task_info.n_samples)
    config = dict(n_estimators=1000, n_threads=4, max_depth=6, num_leaves=31)
    raw_features = get_resource_features(config, ds, n_cv=1, n_refit=0, n_splits=1)
    ram_features = process_resource_features(raw_features, ram_feature_spec)
    ram_gb = sum([ram_features[key] * ram_params[key] for key in ram_params])
    time_features = process_resource_features(raw_features, time_feature_spec)
    time_s = sum([time_features[key] * time_params[key] for key in time_params])
    print(f'{ram_gb=}, {time_s=}')


if __name__ == '__main__':
    print(get_param_grid(dict(n_estimators=[2], max_depth=[4, 6, 7, 9])))
    paths = Paths.from_env_variables()
    # estimate_catboost_params(paths)
    # estimate_params(paths, 'CB-class-7', 'meta-test-class', CatBoostTDClassifier(verbosity=2))
    # # estimate_params(paths, 'CB-reg-7', 'meta-test-reg', CatBoostTDClassifier(verbosity=2))
    # estimate_params(paths, 'XGB-class-2', 'meta-test-class',
    #                 XGBTDClassifier(verbosity=2, subsample=1.0, colsample_bytree=1.0, colsample_bylevel=1.0))
    # estimate_params(paths, 'LGBM-class-3', 'meta-test-class',
    #                 LGBMTDClassifier(subsample=1.0), is_lgbm=True)
    estimate_params_new(paths, 'CB-class-11', 'meta-test-class',
                        CatBoost_TD_Classifier(subsample=1.0),
                        hparam_grid=get_param_grid(dict(n_estimators=[2], n_threads=[4], max_depth=[4, 6, 7, 9])),
                        short_name='cb_class')
    # estimate_params(paths, 'CB-reg-7', 'meta-test-reg', CatBoostTDClassifier(verbosity=2))
    estimate_params_new(paths, 'XGB-class-3', 'meta-test-class',
                        XGB_TD_Classifier(subsample=1.0, colsample_bytree=1.0, colsample_bylevel=1.0),
                        hparam_grid=get_param_grid(dict(n_estimators=[2], n_threads=[4], max_depth=[4, 6, 8, 11])),
                        short_name='xgb_class')
    estimate_params_new(paths, 'LGBM-class-4', 'meta-test-class',
                        LGBM_TD_Classifier(subsample=1.0, colsample_bytree=1.0),
                        hparam_grid=get_param_grid(dict(n_estimators=[2], n_threads=[4],
                                                        num_leaves=[31, 100, 300, 1000])),
                        short_name='lgbm_class',
                        is_lgbm=True)
    pass


def calibrate_resources(exp_name: str, paths: Paths,
                        learner_space: Dict[str, Sampler],
                        coll_name: str,
                        time_feature_spec: List[str],
                        ram_feature_spec: List[str],
                        sklearn_learner: BaseEstimator, n_combinations: int,
                        rerun: bool) \
        -> Dict[str, Dict[str, float]]:
    if multiprocessing.get_start_method() != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    all_results = []
    task_infos = TaskCollection.from_name(coll_name, paths).load_infos(paths)
    for i in range(n_combinations):
        np.random.seed(i)
        torch.manual_seed(i)
        file_path = paths.resources_exp_it(exp_name, i) / 'results.yaml'
        learner_params = {key: value.sample() for key, value in learner_space.items()}
        task_idx = np.random.randint(len(task_infos))
        task_info = task_infos[task_idx]
        print(f'Iteration {i + 1}/{n_combinations}: Evaluating {type(sklearn_learner)} with \n'
              f'{str(task_info.task_desc)=}\n'
              f'{learner_params=}', flush=True)

        if utils.existsFile(file_path) and not rerun:
            print(f'Loading saved result')
            all_results.append(utils.deserialize(file_path, use_yaml=True))
        else:
            print(f'Running estimator...')
            # compute it

            learner: BaseEstimator = sklearn.base.clone(sklearn_learner)
            learner.set_params(**learner_params)
            ds = task_info.load_task(paths).ds
            X, y = ds_to_xy(ds)
            f = lambda learner_=learner, X_=X, y_=y[:, 0]: learner_.fit(X_, y_)
            new_results: Dict[str, Dict[str, Any]] = dict()
            new_results['measured'] = measure_resources(f)
            new_results['features'] = get_resource_features(config=learner_params, ds=ds,
                                                            n_cv=1, n_refit=0, n_splits=1)
            # new_results['features'] = {'time_s': time_feature_map.get_features(ds),
            #                            'ram_gb': ram_feature_map.get_features(ds)}

            all_results.append(new_results)
            utils.serialize(file_path, new_results, use_yaml=True)

        print(all_results[-1]['measured'])

    coefs = dict()

    coefs['time_s'] = fit_resource_factors([(process_resource_features(results['features'], time_feature_spec),
                                             results['measured']['time_s'])
                                            for results in all_results], pessimistic=False)
    coefs['ram_gb'] = fit_resource_factors([(process_resource_features(results['features'], ram_feature_spec),
                                             results['measured']['ram_gb'])
                                            for results in all_results], pessimistic=True)
    return coefs


def calibrate_resources_new_2(exp_name: str, paths: Paths,
                              hparam_grid: List[Dict[str, Any]],
                              coll_name: str,
                              time_feature_spec: List[str],
                              ram_feature_spec: List[str],
                              sklearn_learner: BaseEstimator,
                              rerun: bool) \
        -> Dict[str, Dict[str, float]]:
    if multiprocessing.get_start_method() != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)
    all_results = []
    task_infos = TaskCollection.from_name(coll_name, paths).load_infos(paths)
    for idx_1, task_info in enumerate(task_infos):
        for idx_2, learner_params in enumerate(hparam_grid):
            i = idx_1 * len(hparam_grid) + idx_2
            np.random.seed(i)
            torch.manual_seed(i)
            file_path = paths.resources_exp_it(exp_name, i) / 'results.yaml'
            print(f'Iteration {i + 1}/{len(task_infos)*len(hparam_grid)}: Evaluating {type(sklearn_learner)} with \n'
                  f'{str(task_info.task_desc)=}\n'
                  f'{learner_params=}', flush=True)

            if utils.existsFile(file_path) and not rerun:
                print(f'Loading saved result')
                all_results.append(utils.deserialize(file_path, use_yaml=True))
            else:
                print(f'Running estimator...')
                # compute it

                learner: BaseEstimator = sklearn.base.clone(sklearn_learner)
                learner.set_params(**learner_params)
                ds = task_info.load_task(paths).ds
                X, y = ds_to_xy(ds)
                f = lambda learner_=learner, X_=X, y_=y[:, 0]: learner_.fit(X_, y_)
                new_results: Dict[str, Dict[str, Any]] = dict()
                new_results['measured'] = measure_resources(f)
                new_results['features'] = get_resource_features(config=learner_params, ds=ds,
                                                                n_cv=1, n_refit=0, n_splits=1)
                # new_results['features'] = {'time_s': time_feature_map.get_features(ds),
                #                            'ram_gb': ram_feature_map.get_features(ds)}

                all_results.append(new_results)
                utils.serialize(file_path, new_results, use_yaml=True)

            print(all_results[-1]['measured'])

    coefs = dict()

    coefs['time_s'] = fit_resource_factors([(process_resource_features(results['features'], time_feature_spec),
                                             results['measured']['time_s'])
                                            for results in all_results], pessimistic=True)
    coefs['ram_gb'] = fit_resource_factors([(process_resource_features(results['features'], ram_feature_spec),
                                             results['measured']['ram_gb'])
                                            for results in all_results], pessimistic=True, coef_factor=1.6)
    return coefs


def measure_resources(f: Callable[[], None]) -> Dict[str, float]:
    # open function in one process (that measures the time), poll the RAM usages from another process
    process = FunctionProcess(TimeWrapper(f))
    process.start()
    time_interval = 0.01
    max_ram_usage_gb = 0.0
    while not process.is_done():
        max_ram_usage_gb = max(max_ram_usage_gb, process.get_ram_usage_gb())
        time.sleep(time_interval)
    process_time = process.pop_result()
    return {'time_s': process_time, 'ram_gb': max_ram_usage_gb}
