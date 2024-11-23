from typing import List, Optional, Dict, Any

import numpy as np

from pytabkit.bench.alg_wrappers.interface_wrappers import RandomParamsNNInterfaceWrapper, NNInterfaceWrapper, \
    AutoGluonModelInterfaceWrapper, CatBoostInterfaceWrapper, LGBMInterfaceWrapper, XGBInterfaceWrapper, \
    XGBHyperoptInterfaceWrapper, CatBoostHyperoptInterfaceWrapper, LGBMHyperoptInterfaceWrapper
from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskCollection
from pytabkit.bench.run.task_execution import RunConfig, TabBenchJobManager
from pytabkit.bench.scheduling.execution import RayJobManager
from pytabkit.bench.scheduling.schedulers import SimpleJobScheduler
from pytabkit.models import utils
from pytabkit.models.alg_interfaces.nn_interfaces import RealMLPParamSampler
from pytabkit.models.sklearn.default_params import DefaultParams


def run_extra_realmlp_tuning_configs(paths: Paths, n_steps: int = 50, tag: str = 'paper_reamlp-hpo-clr',
                                     rerun: bool = False):
    # 1h8m for 5 steps of clr on meta-train. 2h40m for 5 steps of ms on meta-train.
    job_mgr = TabBenchJobManager(paths)
    scheduler = SimpleJobScheduler(RayJobManager())
    config_10_1_0 = RunConfig(n_tt_splits=10, n_cv=1, n_refit=0, save_y_pred=True)

    train_class_task_infos = TaskCollection.from_name('meta-train-class', paths).load_infos(paths)
    train_reg_task_infos = TaskCollection.from_name('meta-train-reg', paths).load_infos(paths)
    test_class_task_infos = TaskCollection.from_name('meta-test-class', paths).load_infos(paths)
    test_reg_task_infos = TaskCollection.from_name('meta-test-reg', paths).load_infos(paths)

    class_task_infos = train_class_task_infos + test_class_task_infos
    reg_task_infos = train_reg_task_infos + test_reg_task_infos
    train_task_infos = train_class_task_infos + train_reg_task_infos
    test_task_infos = test_class_task_infos + test_reg_task_infos
    all_task_infos = class_task_infos + reg_task_infos

    for step_idx in range(n_steps):
        job_mgr.add_jobs(train_task_infos, config_10_1_0,
                         f'RealMLP-HPO-clr_step-{step_idx}',
                         RandomParamsNNInterfaceWrapper(model_idx=step_idx, hpo_space_name='clr'),
                         tags=['realmlp-hpo-clr'], rerun=rerun)
        job_mgr.add_jobs(train_task_infos, config_10_1_0,
                         f'RealMLP-HPO-moresigma_step-{step_idx}',
                         RandomParamsNNInterfaceWrapper(model_idx=step_idx, hpo_space_name='moresigma'),
                         tags=['realmlp-hpo-ms'], rerun=rerun)
        job_mgr.add_jobs(train_task_infos, config_10_1_0,
                         f'RealMLP-HPO-moresigmadim_step-{step_idx}',
                         RandomParamsNNInterfaceWrapper(model_idx=step_idx, hpo_space_name='moresigmadim'),
                         tags=['realmlp-hpo-msd'], rerun=rerun)
        job_mgr.add_jobs(train_task_infos, config_10_1_0,
                         f'RealMLP-HPO-moresigmadimreg_step-{step_idx}',
                         RandomParamsNNInterfaceWrapper(model_idx=step_idx, hpo_space_name='moresigmadimreg'),
                         tags=['realmlp-hpo-msdr'], rerun=rerun)
        job_mgr.add_jobs(train_task_infos, config_10_1_0,
                         f'RealMLP-HPO-moresigmadimsize_step-{step_idx}',
                         RandomParamsNNInterfaceWrapper(model_idx=step_idx, hpo_space_name='moresigmadimsize'),
                         tags=['realmlp-hpo-msds'], rerun=rerun)
        job_mgr.add_jobs(train_task_infos, config_10_1_0,
                         f'RealMLP-HPO-moresigmadimlr_step-{step_idx}',
                         RandomParamsNNInterfaceWrapper(model_idx=step_idx, hpo_space_name='moresigmadimlr'),
                         tags=['realmlp-hpo-msdl'], rerun=rerun)

    job_mgr.run_jobs(scheduler)



def run_mlp_random_configs(paths: Paths, n_steps: int = 50, tag: str = 'mlp_random', rerun: bool = False):
    job_mgr = TabBenchJobManager(paths)
    scheduler = SimpleJobScheduler(RayJobManager())
    config_10_1_0 = RunConfig(n_tt_splits=10, n_cv=1, n_refit=0, save_y_pred=True)

    train_class_task_infos = TaskCollection.from_name('meta-train-class', paths).load_infos(paths)
    train_reg_task_infos = TaskCollection.from_name('meta-train-reg', paths).load_infos(paths)
    test_class_task_infos = TaskCollection.from_name('meta-test-class', paths).load_infos(paths)
    test_reg_task_infos = TaskCollection.from_name('meta-test-reg', paths).load_infos(paths)

    class_task_infos = train_class_task_infos + test_class_task_infos
    reg_task_infos = train_reg_task_infos + test_reg_task_infos
    train_task_infos = train_class_task_infos + train_reg_task_infos
    test_task_infos = test_class_task_infos + test_reg_task_infos
    all_task_infos = class_task_infos + reg_task_infos

    sampler = RealMLPParamSampler(is_classification=False)

    for step_idx in range(n_steps):
        params = sampler.sample_params(seed=step_idx)
        relevant_params = {key: value for key, value in params.items() if key in
                           ['num_emb_type', 'add_front_scale', 'lr', 'p_drop', 'wd',
                            'plr_sigma', 'act', 'hidden_sizes', 'ls_eps']}
        config_str = ''
        for key, value in relevant_params.items():
            if key == 'hidden_sizes':
                value = f'{value[0]}x{len(value)}'
            config_str = config_str + '_' + key.replace('_', '-') + '-' + str(value)
        job_mgr.add_jobs(train_reg_task_infos, config_10_1_0,
                         f'RealMLP-reg' + config_str,
                         NNInterfaceWrapper(**params),
                         tags=[tag], rerun=rerun)

    sampler = RealMLPParamSampler(is_classification=True)

    for step_idx in range(n_steps):
        params = sampler.sample_params(seed=step_idx)
        relevant_params = {key: value for key, value in params.items() if key in
                           ['num_emb_type', 'add_front_scale', 'lr', 'p_drop', 'wd',
                            'plr_sigma', 'act', 'hidden_sizes', 'ls_eps']}
        config_str = ''
        for key, value in relevant_params.items():
            if key == 'hidden_sizes':
                value = f'{value[0]}x{len(value)}'
            config_str = config_str + '_' + key.replace('_', '-') + '-' + str(value)
        job_mgr.add_jobs(train_class_task_infos, config_10_1_0,
                         f'RealMLP-class' + config_str,
                         NNInterfaceWrapper(**params),
                         tags=[tag], rerun=rerun)

    job_mgr.run_jobs(scheduler)


def run_mlp_random_seed_configs(paths: Paths, n_steps: int = 50, tag: str = 'mlp_random_seeds', rerun: bool = False):
    job_mgr = TabBenchJobManager(paths)
    scheduler = SimpleJobScheduler(RayJobManager())
    config_10_1_0 = RunConfig(n_tt_splits=10, n_cv=1, n_refit=0, save_y_pred=True)

    train_class_task_infos = TaskCollection.from_name('meta-train-class', paths).load_infos(paths)
    train_reg_task_infos = TaskCollection.from_name('meta-train-reg', paths).load_infos(paths)
    test_class_task_infos = TaskCollection.from_name('meta-test-class', paths).load_infos(paths)
    test_reg_task_infos = TaskCollection.from_name('meta-test-reg', paths).load_infos(paths)

    class_task_infos = train_class_task_infos + test_class_task_infos
    reg_task_infos = train_reg_task_infos + test_reg_task_infos
    train_task_infos = train_class_task_infos + train_reg_task_infos
    test_task_infos = test_class_task_infos + test_reg_task_infos
    all_task_infos = class_task_infos + reg_task_infos

    for step_idx in range(n_steps):
        job_mgr.add_jobs(train_reg_task_infos, config_10_1_0,
                         f'RealMLP-reg_seed-offset-{step_idx}',
                         NNInterfaceWrapper(**DefaultParams.RealMLP_TD_REG, random_seed_offset=step_idx),
                         tags=[tag], rerun=rerun)

    job_mgr.run_jobs(scheduler)



def run_additional_configs(paths: Paths, tag: str = 'paper_additional', rerun: bool = False):
    # not in the paper
    # this took around 17h24m
    job_mgr = TabBenchJobManager(paths)
    scheduler = SimpleJobScheduler(RayJobManager())
    config_10_1_0 = RunConfig(n_tt_splits=10, n_cv=1, n_refit=0, save_y_pred=True)

    train_class_task_infos = TaskCollection.from_name('meta-train-class', paths).load_infos(paths)
    train_reg_task_infos = TaskCollection.from_name('meta-train-reg', paths).load_infos(paths)
    test_class_task_infos = TaskCollection.from_name('meta-test-class', paths).load_infos(paths)
    test_reg_task_infos = TaskCollection.from_name('meta-test-reg', paths).load_infos(paths)
    grinsztajn_class_task_infos = TaskCollection.from_name('grinsztajn-class', paths).load_infos(paths)
    grinsztajn_reg_task_infos = TaskCollection.from_name('grinsztajn-reg', paths).load_infos(paths)

    class_task_infos = train_class_task_infos + test_class_task_infos + grinsztajn_class_task_infos
    reg_task_infos = train_reg_task_infos + test_reg_task_infos + grinsztajn_reg_task_infos
    train_task_infos = train_class_task_infos + train_reg_task_infos
    test_task_infos = test_class_task_infos + test_reg_task_infos
    all_task_infos = class_task_infos + reg_task_infos

    # run class-on-reg and reg-on-class

    job_mgr.add_jobs(train_reg_task_infos, config_10_1_0,
                     'RealMLP-TD-class-on-reg',
                     NNInterfaceWrapper(**utils.update_dict(DefaultParams.RealMLP_TD_CLASS,
                                                            dict(use_ls=False, ls_eps=0.0, normalize_output=True,
                                                                 clamp_output=True))),
                     tags=[tag], rerun=rerun)

    job_mgr.add_jobs(train_class_task_infos, config_10_1_0,
                     'RealMLP-TD-reg-on-class',
                     NNInterfaceWrapper(**utils.update_dict(DefaultParams.RealMLP_TD_REG,
                                                            dict(use_ls=True, ls_eps=0.1, normalize_output=False,
                                                                 clamp_output=False))),
                     tags=[tag], rerun=rerun)

    job_mgr.add_jobs(train_reg_task_infos, config_10_1_0,
                     'RealMLP-TD-S-class-on-reg',
                     NNInterfaceWrapper(**utils.update_dict(DefaultParams.RealMLP_TD_S_CLASS,
                                                            dict(use_ls=False, ls_eps=0.0, normalize_output=True))),
                     tags=[tag], rerun=rerun)

    job_mgr.add_jobs(train_class_task_infos, config_10_1_0,
                     'RealMLP-TD-S-reg-on-class',
                     NNInterfaceWrapper(**utils.update_dict(DefaultParams.RealMLP_TD_S_REG,
                                                            dict(use_ls=True, ls_eps=0.1, normalize_output=False))),
                     tags=[tag], rerun=rerun)

    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'RealMLP-TD-class_only-one-hot',
                     NNInterfaceWrapper(**utils.update_dict(DefaultParams.RealMLP_TD_CLASS,
                                                            dict(max_one_hot_cat_size=-1))),
                     tags=[tag], rerun=rerun)

    job_mgr.add_jobs(reg_task_infos, config_10_1_0,
                     'RealMLP-TD-reg_only-one-hot',
                     NNInterfaceWrapper(**utils.update_dict(DefaultParams.RealMLP_TD_REG,
                                                            dict(max_one_hot_cat_size=-1))),
                     tags=[tag], rerun=rerun)

    job_mgr.run_jobs(scheduler)


def run_seed_opt_configs(paths: Paths, random_seed_offset: int, tag: str = 'paper', rerun: bool = False):
    # not used in the paper
    # this took around 17h24m
    job_mgr = TabBenchJobManager(paths)
    scheduler = SimpleJobScheduler(RayJobManager())
    config_10_1_0 = RunConfig(n_tt_splits=10, n_cv=1, n_refit=0, save_y_pred=True)

    train_class_task_infos = TaskCollection.from_name('meta-train-class', paths).load_infos(paths)
    train_reg_task_infos = TaskCollection.from_name('meta-train-reg', paths).load_infos(paths)
    test_class_task_infos = TaskCollection.from_name('meta-test-class', paths).load_infos(paths)
    test_reg_task_infos = TaskCollection.from_name('meta-test-reg', paths).load_infos(paths)

    class_task_infos = train_class_task_infos + test_class_task_infos
    reg_task_infos = train_reg_task_infos + test_reg_task_infos
    train_task_infos = train_class_task_infos + train_reg_task_infos
    test_task_infos = test_class_task_infos + test_reg_task_infos
    all_task_infos = class_task_infos + reg_task_infos

    job_mgr.add_jobs(train_class_task_infos, config_10_1_0,
                     f'RealMLP-TD-class_alt-seed-{random_seed_offset}',
                     NNInterfaceWrapper(**DefaultParams.RealMLP_TD_CLASS, random_seed_offset=random_seed_offset),
                     tags=[tag], rerun=rerun)

    job_mgr.add_jobs(train_reg_task_infos, config_10_1_0,
                     f'RealMLP-TD-reg_alt-seed-{random_seed_offset}',
                     NNInterfaceWrapper(**DefaultParams.RealMLP_TD_REG, random_seed_offset=random_seed_offset),
                     tags=[tag], rerun=rerun)

    job_mgr.run_jobs(scheduler)


def run_ag_nn_configs(paths: Paths, tag: str = 'paper', rerun: bool = False,
                      only_meta_train: bool = False, only_meta_test: bool = False,
                      with_ftt: bool = True,
                      start_split: int = 0, end_split: int = 10):
    # ca 50 min for meta-train-reg
    job_mgr = TabBenchJobManager(paths)
    scheduler = SimpleJobScheduler(RayJobManager())
    config_10_1_0 = RunConfig(n_tt_splits=end_split, min_split_idx=start_split, n_cv=1, n_refit=0, save_y_pred=True)

    train_class_task_infos = TaskCollection.from_name('meta-train-class', paths).load_infos(paths)
    train_reg_task_infos = TaskCollection.from_name('meta-train-reg', paths).load_infos(paths)
    test_class_task_infos = TaskCollection.from_name('meta-test-class', paths).load_infos(paths)
    test_reg_task_infos = TaskCollection.from_name('meta-test-reg', paths).load_infos(paths)
    grinsztajn_class_task_infos = TaskCollection.from_name('grinsztajn-class', paths).load_infos(paths)
    grinsztajn_reg_task_infos = TaskCollection.from_name('grinsztajn-reg', paths).load_infos(paths)

    class_task_infos = train_class_task_infos + test_class_task_infos + grinsztajn_class_task_infos
    reg_task_infos = train_reg_task_infos + test_reg_task_infos + grinsztajn_reg_task_infos
    train_task_infos = train_class_task_infos + train_reg_task_infos
    test_task_infos = test_class_task_infos + test_reg_task_infos
    all_task_infos = class_task_infos + reg_task_infos

    if only_meta_train:
        class_task_infos = train_class_task_infos
        reg_task_infos = train_reg_task_infos
    elif only_meta_test:
        class_task_infos = test_class_task_infos
        reg_task_infos = test_reg_task_infos

    # fastai on meta-train took 40 GPU-minutes
    # MLP-AGT took 1h31m on one RTX 3090
    # FT-T with some RAM estimates: ca 44m + 17h + 40m
    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'MLP-FAI-D-class',
                     AutoGluonModelInterfaceWrapper(use_gpu=True, hp_family='default', model_types='FASTAI',
                                                    max_n_models_per_type=1),
                     tags=[tag], rerun=rerun)
    job_mgr.add_jobs(reg_task_infos, config_10_1_0,
                     'MLP-FAI-D-reg',
                     AutoGluonModelInterfaceWrapper(use_gpu=True, hp_family='default', model_types='FASTAI',
                                                    max_n_models_per_type=1),
                     tags=[tag], rerun=rerun)
    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'MLP-AGT-D-class',
                     AutoGluonModelInterfaceWrapper(use_gpu=True, hp_family='default', model_types='NN_TORCH',
                                                    max_n_models_per_type=1),
                     tags=[tag], rerun=rerun)
    job_mgr.add_jobs(reg_task_infos, config_10_1_0,
                     'MLP-AGT-D-reg',
                     AutoGluonModelInterfaceWrapper(use_gpu=True, hp_family='default', model_types='NN_TORCH',
                                                    max_n_models_per_type=1),
                     tags=[tag], rerun=rerun)
    if with_ftt:
        job_mgr.add_jobs(class_task_infos, config_10_1_0,
                         'FT-Transformer-D-class',
                         AutoGluonModelInterfaceWrapper(use_gpu=True, hp_family='default_FTT',
                                                        model_types='FT_TRANSFORMER',
                                                        max_n_models_per_type=1),
                         tags=[tag], rerun=rerun)
        job_mgr.add_jobs(reg_task_infos, config_10_1_0,
                         'FT-Transformer-D-reg',
                         AutoGluonModelInterfaceWrapper(use_gpu=True, hp_family='default_FTT',
                                                        model_types='FT_TRANSFORMER',
                                                        max_n_models_per_type=1),
                         tags=[tag], rerun=rerun)

    job_mgr.run_jobs(scheduler)


def run_trees_custom(paths: Paths, n_estimators: int, tag: str = 'paper', with_defaults: bool = True):
    # only for speed-testing
    # this generates about 10GB of data
    # took 7h17m for n_estimators=2
    # took about 6h30m for n_estimators=1  (but slightly more tasks were run for that because of the rerun=True)
    # the large main overhead is probably mainly for evaluating the metrics
    job_mgr = TabBenchJobManager(paths)
    scheduler = SimpleJobScheduler(RayJobManager())
    config_10_1_0 = RunConfig(n_tt_splits=10, n_cv=1, n_refit=0, save_y_pred=True)

    train_class_task_infos = TaskCollection.from_name('meta-train-class', paths).load_infos(paths)
    train_reg_task_infos = TaskCollection.from_name('meta-train-reg', paths).load_infos(paths)
    test_class_task_infos = TaskCollection.from_name('meta-test-class', paths).load_infos(paths)
    test_reg_task_infos = TaskCollection.from_name('meta-test-reg', paths).load_infos(paths)

    class_task_infos = train_class_task_infos + test_class_task_infos
    reg_task_infos = train_reg_task_infos + test_reg_task_infos
    all_task_infos = class_task_infos + reg_task_infos

    job_mgr.add_jobs(all_task_infos, config_10_1_0, f'XGB_hyperopt-50_grinsztajn_nest-{n_estimators}',
                     XGBHyperoptInterfaceWrapper(n_estimators=n_estimators, n_hyperopt_steps=50,
                                                 tree_method='hist', space='grinsztajn'),
                     tags=[tag], rerun=True)
    job_mgr.add_jobs(all_task_infos, config_10_1_0, f'CatBoost_hyperopt-50_shwartz-ziv_nest-{n_estimators}',
                     CatBoostHyperoptInterfaceWrapper(n_estimators=n_estimators, n_hyperopt_steps=50,
                                                      space='shwartz-ziv'),
                     tags=[tag], rerun=True)
    job_mgr.add_jobs(all_task_infos, config_10_1_0, f'LGBM_hyperopt-50_cqb_nest-{n_estimators}',
                     LGBMHyperoptInterfaceWrapper(n_estimators=n_estimators, n_hyperopt_steps=50,
                                                  space='catboost_quality_benchmarks'), rerun=True)

    if with_defaults:
        # optimized default parameters
        # classification
        job_mgr.add_jobs(class_task_infos, config_10_1_0, f'LGBM-TD-class_nest-{n_estimators}',
                         LGBMInterfaceWrapper(**utils.update_dict(DefaultParams.LGBM_TD_CLASS,
                                                                  dict(n_estimators=n_estimators))),
                         tags=[tag], rerun=True)
        job_mgr.add_jobs(class_task_infos, config_10_1_0,
                         f'XGB-TD-class_nest-{n_estimators}',
                         XGBInterfaceWrapper(**utils.update_dict(DefaultParams.XGB_TD_CLASS,
                                                                 dict(n_estimators=n_estimators))),
                         tags=[tag], rerun=True)
        job_mgr.add_jobs(class_task_infos, config_10_1_0,
                         f'CatBoost-TD-class_nest-{n_estimators}',
                         CatBoostInterfaceWrapper(**utils.update_dict(DefaultParams.CB_TD_CLASS,
                                                                      dict(n_estimators=n_estimators))),
                         tags=[tag], rerun=True)

        # regression
        job_mgr.add_jobs(reg_task_infos, config_10_1_0,
                         f'LGBM-TD-reg_nest-{n_estimators}',
                         LGBMInterfaceWrapper(**utils.update_dict(DefaultParams.LGBM_TD_REG,
                                                                  dict(n_estimators=n_estimators))),
                         tags=[tag], rerun=True)
        job_mgr.add_jobs(reg_task_infos, config_10_1_0,
                         f'XGB-TD-reg_nest-{n_estimators}',
                         XGBInterfaceWrapper(**utils.update_dict(DefaultParams.XGB_TD_REG,
                                                                 dict(n_estimators=n_estimators))),
                         tags=[tag], rerun=True)
        job_mgr.add_jobs(reg_task_infos, config_10_1_0,
                         f'CatBoost-TD-reg_nest-{n_estimators}',
                         CatBoostInterfaceWrapper(**utils.update_dict(DefaultParams.CB_TD_REG,
                                                                      dict(n_estimators=n_estimators))),
                         tags=[tag], rerun=True)

    job_mgr.run_jobs(scheduler)


def run_cumulative_ablations(paths: Paths, tag: str = 'paper_cumulative_ablations', rerun: bool = False):
    job_mgr = TabBenchJobManager(paths)
    scheduler = SimpleJobScheduler(RayJobManager())
    config_10_1_0 = RunConfig(n_tt_splits=10, n_cv=1, n_refit=0, save_y_pred=False)  # todo: it's false

    train_class_task_infos = TaskCollection.from_name('meta-train-class', paths).load_infos(paths)
    train_reg_task_infos = TaskCollection.from_name('meta-train-reg', paths).load_infos(paths)

    # lr_grid_ntp = [0.01, 0.015, 0.025, 0.04, 0.07, 0.1, 0.2, 0.3, 0.4]
    # lr_grid_std = [4e-4, 7e-4, 1e-3, 1.5e-3, 2.5e-3, 4e-3, 7e-3, 1e-2, 2e-2]
    lr_grid_std = [2e-3, 4e-3]
    lr_grid_ntp = [0.04, 0.2]

    config_class = dict()
    config_reg = dict()
    ablation_counter = 1

    def add_config(name: str, lr_grid: List[float],
                   add: Optional[Dict[str, Any]] = None, add_class: Optional[Dict[str, Any]] = None,
                   add_reg: Optional[Dict[str, Any]] = None, run_this: bool = True):
        nonlocal ablation_counter
        nonlocal config_class
        nonlocal config_reg

        if add is not None:
            config_class = utils.join_dicts(config_class, add)
            config_reg = utils.join_dicts(config_reg, add)
        if add_class is not None:
            config_class = utils.join_dicts(config_class, add_class)
        if add_reg is not None:
            config_reg = utils.join_dicts(config_reg, add_reg)

        if run_this:
            for lr in lr_grid:
                for task_infos, task_type_name, config in [(train_class_task_infos, 'class', config_class),
                                                           (train_reg_task_infos, 'reg', config_reg)]:
                    job_mgr.add_jobs(task_infos, config_10_1_0,
                                     f'MLP-cumul-abl-{ablation_counter}-{task_type_name}_{name}_lr-{lr:g}',
                                     NNInterfaceWrapper(**utils.update_dict(config, dict(lr=lr))),
                                     tags=[tag], rerun=rerun)

        ablation_counter += 1

    mlp_rtdl_repr_config_class = dict(
        hidden_sizes=[128, 256, 128],
        p_drop=0.1,
        block_str='w-b-a-d',
        lr=1e-3,  # will be overridden by the lrs from the grid anyway
        opt='adam',
        tfms=['quantile_tabr', 'embedding'],
        embedding_size=8,
        batch_size=128,
        n_epochs=1000,
        use_early_stopping=True,
        early_stopping_multiplicative_patience=1,
        early_stopping_additive_patience=20,
        act='relu',
        weight_param='standard',
        weight_init_mode='uniform',
        weight_init_gain=1. / np.sqrt(3.),
        bias_init_mode='pytorch-default',
        max_n_vectorized=1,
        use_last_best_epoch=False,
        emb_init_mode='kaiming-uniform-t',
    )

    # for reproducing: weight decay
    # initialize missing embeddings to zero
    # have a different early stopping tolerance threshold

    # MLP-RTDL also uses the two-output + cross-entropy thing
    # hard to reproduce: handling unknown classes with different embedding category initialized to zero

    # todo: include all lr factors etc.

    add_config('rtdl-d-reprod', [1e-3], add=mlp_rtdl_repr_config_class,
               add_reg=dict(normalize_output=True), run_this=True)
    add_config('tune-lr', lr_grid_std)
    add_config('max-epochs-256', lr_grid_std, dict(n_epochs=256))
    add_config('batch-size-256', lr_grid_std, dict(batch_size=256))
    add_config('hidden-256x3', lr_grid_std, dict(hidden_sizes=[256] * 3))
    add_config('normal-emb-init', lr_grid_std, dict(emb_init_mode='normal'))
    add_config('one-hot-small-cat', lr_grid_std, dict(tfms=['quantile_tabr', 'one_hot', 'embedding'],
                                                      max_one_hot_cat_size=9))
    # quantile_tabr was not well-suited for vectorization, now we can vectorize
    add_config('robust-scale-smooth-clip', lr_grid_std,
               dict(tfms=['one_hot', 'median_center', 'robust_scale', 'smooth_clip', 'embedding'],
                    max_n_vectorized=50))
    add_config('no-early-stop', lr_grid_std, dict(use_early_stopping=False))
    add_config('last-best-epoch', lr_grid_std, dict(use_last_best_epoch=True))
    add_config('lr-multi-cycle', lr_grid_std, dict(lr_sched='coslog4'))
    add_config('beta2-0.95', lr_grid_std, dict(sq_mom=0.95))
    add_config('label-smoothing', lr_grid_std, add_class=dict(use_ls=True, ls_eps=0.1))
    add_config('output-clipping', lr_grid_std, add_reg=dict(clamp_output=True))
    add_config('ntp', lr_grid_ntp, dict(weight_param='ntk', bias_lr_factor=0.1))
    add_config('weight-init-std', lr_grid_ntp, dict(weight_init_mode='std', weight_init_gain=1.0))
    add_config('bias-init-he+5', lr_grid_ntp, dict(bias_init_mode='he+5'))
    add_config('different-act', lr_grid_ntp, add_class=dict(act='selu'), add_reg=dict(act='mish'))
    add_config('param-act', lr_grid_ntp, dict(use_parametric_act=True, act_lr_factor=0.1))
    add_config('front-scale', lr_grid_ntp, dict(add_front_scale=True, scale_lr_factor=6.0))
    add_config('num-emb-pl', lr_grid_ntp,
               dict(num_emb_type='pl', plr_sigma=0.1, plr_hidden_1=16, plr_hidden_2=4, plr_lr_factor=0.1))
    add_config('num-emb-pbld', lr_grid_ntp, dict(num_emb_type='pbld'))
    add_config('pdrop-0.15', lr_grid_ntp, dict(p_drop=0.15))
    add_config('pdrop-flat-cos', lr_grid_ntp, dict(p_drop_sched='flat_cos'))
    add_config('wd-0.02', lr_grid_ntp, dict(wd=0.02, bias_wd_factor=0.0))
    add_config('wd-flat-cos', lr_grid_ntp, dict(wd_sched='flat_cos'), run_this=True)

    job_mgr.run_jobs(scheduler)
    pass


if __name__ == '__main__':
    pass
    # ----- not in the paper, only experimental -----
    # for i in range(50):
    #     if (i + 1) % 5 == 0:
    #         run_extra_realmlp_tuning_configs(paths, n_steps=i + 1)

    # run_additional_configs(paths)
    # run_ag_nn_configs(paths, tag='paper', only_meta_train=True, with_ftt=True)
    # run_mlp_random_configs(paths, n_steps=50)  # not in the paper
    # run_mlp_random_seed_configs(paths, n_steps=20)  # not in the paper
    ## run_seed_opt_configs(paths, random_seed_offset=1, tag='paper_seeds')
    # run_cumulative_ablations(paths)
