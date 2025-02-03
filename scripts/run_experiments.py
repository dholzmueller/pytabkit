from typing import Optional, Dict, Any, List

import numpy as np

from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskCollection
from pytabkit.bench.alg_wrappers.interface_wrappers import \
    LGBMInterfaceWrapper, \
    XGBInterfaceWrapper, LGBMHyperoptInterfaceWrapper, XGBHyperoptInterfaceWrapper, CatBoostHyperoptInterfaceWrapper, \
    CatBoostInterfaceWrapper, RFInterfaceWrapper, XGBSklearnInterfaceWrapper, LGBMSklearnInterfaceWrapper, \
    CatBoostSklearnInterfaceWrapper, SklearnMLPInterfaceWrapper, NNInterfaceWrapper, CaruanaEnsembleWrapper, \
    LoadResultsWrapper, RandomParamsNNInterfaceWrapper, AlgorithmSelectionWrapper, ResNetRTDLInterfaceWrapper, \
    MLPRTDLInterfaceWrapper, RandomParamsRTDLMLPInterfaceWrapper, RandomParamsResnetInterfaceWrapper, \
    TabRInterfaceWrapper, RandomParamsXGBInterfaceWrapper, RandomParamsLGBMInterfaceWrapper, \
    RandomParamsCatBoostInterfaceWrapper, AutoGluonModelInterfaceWrapper, RandomParamsTabRInterfaceWrapper, \
    RandomParamsRFInterfaceWrapper, FTTransformerInterfaceWrapper, RandomParamsFTTransformerInterfaceWrapper
from pytabkit.bench.eval.analysis import get_ensemble_groups
from pytabkit.bench.run.task_execution import RunConfig, TabBenchJobManager, run_alg_selection
from pytabkit.bench.scheduling.schedulers import SimpleJobScheduler
from pytabkit.models import utils
from pytabkit.models.alg_interfaces.nn_interfaces import RealMLPParamSampler
from pytabkit.bench.scheduling.execution import RayJobManager
from pytabkit.models.sklearn.default_params import DefaultParams


def run_gbdt_rs_configs(paths: Optional[Paths] = None, min_step_idx: int = 0, n_steps: int = 50, rerun: bool = False,
                        with_lgbm: bool = True,
                        with_xgb: bool = True, with_cb: bool = True, min_split_idx: int = 0, n_splits: int = 10,
                        only_meta_train: bool = False):
    if paths is None:
        paths = Paths.from_env_variables()
    job_mgr = TabBenchJobManager(paths)
    scheduler = SimpleJobScheduler(RayJobManager(available_cpu_ram_multiplier=0.5))
    run_config = RunConfig(min_split_idx=min_split_idx, n_tt_splits=min_split_idx + n_splits, n_cv=1, n_refit=0,
                           save_y_pred=True)

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
    if only_meta_train:
        all_task_infos = train_task_infos
    else:
        all_task_infos = class_task_infos + reg_task_infos

    for step_idx in range(min_step_idx, min_step_idx + n_steps):
        if with_xgb:
            job_mgr.add_jobs(all_task_infos, run_config,
                             f'XGB-HPO_step-{step_idx}',
                             RandomParamsXGBInterfaceWrapper(model_idx=step_idx),
                             tags=['paper_xgb_rs'], rerun=rerun)
        if with_lgbm:
            job_mgr.add_jobs(all_task_infos, run_config,
                             f'LGBM-HPO_step-{step_idx}',
                             RandomParamsLGBMInterfaceWrapper(model_idx=step_idx),
                             tags=['paper_lgbm_rs'], rerun=rerun)
        if with_cb:
            job_mgr.add_jobs(all_task_infos, run_config,
                             f'CatBoost-HPO_step-{step_idx}',
                             RandomParamsCatBoostInterfaceWrapper(model_idx=step_idx),
                             tags=['paper_cb_rs'], rerun=rerun)

    job_mgr.run_jobs(scheduler)


def run_rf_rs_configs(paths: Optional[Paths] = None, min_step_idx: int = 0, n_steps: int = 50, rerun: bool = False,
                      min_split_idx: int = 0, n_splits: int = 10):
    # took 18h30m on the Grinsztajn benchmark
    if paths is None:
        paths = Paths.from_env_variables()
    job_mgr = TabBenchJobManager(paths)
    scheduler = SimpleJobScheduler(RayJobManager(available_cpu_ram_multiplier=0.5))
    run_config = RunConfig(min_split_idx=min_split_idx, n_tt_splits=min_split_idx + n_splits, n_cv=1, n_refit=0,
                           save_y_pred=True)

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

    for step_idx in range(min_step_idx, min_step_idx + n_steps):
        job_mgr.add_jobs(grinsztajn_reg_task_infos + grinsztajn_class_task_infos, run_config,
                         f'RF-HPO_step-{step_idx}',
                         RandomParamsRFInterfaceWrapper(model_idx=step_idx),
                         tags=['paper_rf-hpo'], rerun=rerun)

    job_mgr.run_jobs(scheduler)


def run_realmlp_tuning_configs(paths: Paths, n_steps: int = 50, tag: str = 'paper', rerun: bool = False):
    # 2h37m for 10 steps on meta-train-class
    # for 5 steps on all: 1h20m + 13h4m
    # 1h50m for 10 steps on grinsztajn-benchmark
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

    for step_idx in range(n_steps):
        job_mgr.add_jobs(all_task_infos, config_10_1_0,
                         f'RealMLP-HPO_step-{step_idx}',
                         RandomParamsNNInterfaceWrapper(model_idx=step_idx),
                         tags=[tag], rerun=rerun)

    job_mgr.run_jobs(scheduler)


def run_rtdl_tuning_configs(paths: Paths, n_steps: int = 50, rerun: bool = False,
                            with_mlp: bool = True, with_resnet: bool = True, with_mlp_plr: bool = True,
                            with_ftt: bool = True,
                            only_meta_train: bool = False,
                            only_meta_test: bool = False, start_split=0, end_split=10):
    # MLP-PLR takes about 1h5m per step
    # takes around 4d6h for MLP-HPO and MLP-PLR-HPO together
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
    all_train_task_infos = train_class_task_infos + train_reg_task_infos
    grinsztajn_task_infos = grinsztajn_class_task_infos + grinsztajn_reg_task_infos

    if only_meta_train:
        class_task_infos = train_class_task_infos
        reg_task_infos = train_reg_task_infos
        all_task_infos = train_class_task_infos + train_reg_task_infos
    elif only_meta_test:
        class_task_infos = test_class_task_infos
        reg_task_infos = test_reg_task_infos
        all_task_infos = test_class_task_infos + test_reg_task_infos

    for step_idx in range(n_steps):
        if with_mlp:
            job_mgr.add_jobs(all_task_infos, config_10_1_0,
                             f'MLP-RTDL-HPO_step-{step_idx}',
                             RandomParamsRTDLMLPInterfaceWrapper(model_idx=step_idx),
                             tags=['paper_mlp-rtdl-hpo'], rerun=rerun)
        if with_resnet:
            job_mgr.add_jobs(all_task_infos, config_10_1_0,
                             f'ResNet-RTDL-HPO_step-{step_idx}',
                             RandomParamsResnetInterfaceWrapper(model_idx=step_idx),
                             tags=['paper_resnet-hpo'], rerun=rerun)
        if with_mlp_plr:
            job_mgr.add_jobs(all_task_infos, config_10_1_0,
                             f'MLP-PLR-HPO_step-{step_idx}',
                             RandomParamsRTDLMLPInterfaceWrapper(model_idx=step_idx, num_emb_type='plr'),
                             tags=['paper_mlp-plr-hpo'], rerun=rerun)

        if with_ftt:
            job_mgr.add_jobs(grinsztajn_task_infos, config_10_1_0,
                             f'FTT-HPO_step-{step_idx}',
                             RandomParamsFTTransformerInterfaceWrapper(model_idx=step_idx),
                             tags=['paper_ftt-hpo'], rerun=rerun)

    job_mgr.run_jobs(scheduler)


def run_tabr_tuning_configs(paths: Paths, n_steps: int = 50, rerun: bool = False,
                            start_split=0, end_split=10):
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
    grinsztajn_task_infos = grinsztajn_class_task_infos + grinsztajn_reg_task_infos
    all_task_infos = class_task_infos + reg_task_infos
    all_train_task_infos = train_class_task_infos + train_reg_task_infos

    for step_idx in range(n_steps):
        job_mgr.add_jobs(grinsztajn_task_infos, config_10_1_0,
                         f'TabR-HPO_step-{step_idx}',
                         RandomParamsTabRInterfaceWrapper(model_idx=step_idx),
                         tags=['paper_tabr-hpo'], rerun=rerun)
        job_mgr.add_jobs(grinsztajn_task_infos, config_10_1_0,
                         f'RealTabR-HPO_step-{step_idx}',
                         RandomParamsTabRInterfaceWrapper(model_idx=step_idx, hpo_space_name='realtabr'),
                         tags=['paper_realtabr-hpo'], rerun=rerun)

    job_mgr.run_jobs(scheduler)


def run_refit_configs(paths: Paths, tag: str = 'paper', rerun: bool = False):
    # refit experiments took 3 to 3.5 days
    job_mgr = TabBenchJobManager(paths)
    scheduler = SimpleJobScheduler(RayJobManager(available_cpu_ram_multiplier=0.5))
    config_10_5_5 = RunConfig(n_tt_splits=10, n_cv=5, n_refit=5, save_y_pred=True)

    train_class_task_infos = TaskCollection.from_name('meta-train-class', paths).load_infos(paths)
    train_reg_task_infos = TaskCollection.from_name('meta-train-reg', paths).load_infos(paths)
    test_class_task_infos = TaskCollection.from_name('meta-test-class', paths).load_infos(paths)
    test_reg_task_infos = TaskCollection.from_name('meta-test-reg', paths).load_infos(paths)

    class_task_infos = train_class_task_infos + test_class_task_infos
    reg_task_infos = train_reg_task_infos + test_reg_task_infos
    train_task_infos = train_class_task_infos + train_reg_task_infos
    test_task_infos = test_class_task_infos + test_reg_task_infos
    all_task_infos = class_task_infos + reg_task_infos

    for mean_cv, mean_refit in [(False, False), (True, True)]:
        extra_str = f'mean-cv-{mean_cv}_mean-refit-{mean_refit}'
        job_mgr.add_jobs(class_task_infos, config_10_5_5,
                         f'RealMLP-TD-class_{extra_str}',
                         NNInterfaceWrapper(**DefaultParams.RealMLP_TD_CLASS,
                                            use_best_mean_epoch_for_cv=mean_cv,
                                            use_best_mean_epoch_for_refit=mean_refit,
                                            ),
                         tags=[tag], rerun=rerun)

        job_mgr.add_jobs(reg_task_infos, config_10_5_5,
                         f'RealMLP-TD-reg_{extra_str}',
                         NNInterfaceWrapper(**DefaultParams.RealMLP_TD_REG,
                                            use_best_mean_epoch_for_cv=mean_cv,
                                            use_best_mean_epoch_for_refit=mean_refit,
                                            ),
                         tags=[tag], rerun=rerun)

        job_mgr.add_jobs(class_task_infos, config_10_5_5, f'LGBM-TD-class_{extra_str}',
                         LGBMInterfaceWrapper(**DefaultParams.LGBM_TD_CLASS,
                                              use_best_mean_iteration_for_cv=mean_cv,
                                              use_best_mean_iteration_for_refit=mean_refit,
                                              ),
                         tags=[tag], rerun=rerun)

        job_mgr.add_jobs(reg_task_infos, config_10_5_5, f'LGBM-TD-reg_{extra_str}',
                         LGBMInterfaceWrapper(**DefaultParams.LGBM_TD_REG,
                                              use_best_mean_iteration_for_cv=mean_cv,
                                              use_best_mean_iteration_for_refit=mean_refit,
                                              ),
                         tags=[tag], rerun=rerun)

    job_mgr.run_jobs(scheduler)


def run_ablations(paths: Paths, param_configs: Dict[str, Any], with_class: bool = True, with_reg: bool = True,
                  tune_lr: bool = True,
                  tag: str = 'paper_mlp_ablations', rerun: bool = False):
    job_mgr = TabBenchJobManager(paths)
    scheduler = SimpleJobScheduler(RayJobManager())
    config_10_1_0 = RunConfig(n_tt_splits=10, n_cv=1, n_refit=0, save_y_pred=False)  # todo: it's false

    train_class_task_infos = TaskCollection.from_name('meta-train-class', paths).load_infos(paths)
    train_reg_task_infos = TaskCollection.from_name('meta-train-reg', paths).load_infos(paths)
    test_class_task_infos = TaskCollection.from_name('meta-test-class', paths).load_infos(paths)
    test_reg_task_infos = TaskCollection.from_name('meta-test-reg', paths).load_infos(paths)

    class_task_infos = train_class_task_infos + test_class_task_infos
    reg_task_infos = train_reg_task_infos + test_reg_task_infos
    train_task_infos = train_class_task_infos + train_reg_task_infos
    test_task_infos = test_class_task_infos + test_reg_task_infos
    all_task_infos = class_task_infos + reg_task_infos

    combinations = []
    if with_class:
        combinations.append((train_class_task_infos, DefaultParams.RealMLP_TD_CLASS, 'class'))
    if with_reg:
        combinations.append((train_reg_task_infos, DefaultParams.RealMLP_TD_REG, 'reg'))

    # lr_factors = [1.5**k for k in range(-3, 4)] if tune_lr else [1]
    # lr_factors = [0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0] if tune_lr else [1.0]
    # lr_factors = [0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 4.0, 6.0] if tune_lr else [1.0]
    lr_factors = [0.1, 0.15, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 4.0] if tune_lr else [1.0]

    for task_infos, default_params, task_type_name in combinations:
        for param_config_name, extra_params in param_configs.items():
            for lr_factor_idx, lr_factor in enumerate(lr_factors):
                params = utils.update_dict(default_params, extra_params)
                params['lr'] *= lr_factor  # todo: what if the lr is a dict?
                alg_name = f'RealMLP-TD-{task_type_name}-ablation_{param_config_name}_lrfactor-{lr_factor}'
                job_mgr.add_jobs(task_infos, config_10_1_0,
                                 alg_name,
                                 NNInterfaceWrapper(**params),
                                 tags=[tag], rerun=rerun)

    job_mgr.run_jobs(scheduler)


def run_td_configs(paths: Paths, tag: str = 'paper', rerun: bool = False):
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

    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'RealMLP-TD-class',
                     NNInterfaceWrapper(**DefaultParams.RealMLP_TD_CLASS),
                     tags=[tag], rerun=rerun)

    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'RealMLP-TD-S-class',
                     NNInterfaceWrapper(**DefaultParams.RealMLP_TD_S_CLASS),
                     tags=[tag], rerun=rerun)

    job_mgr.add_jobs(reg_task_infos, config_10_1_0,
                     'RealMLP-TD-reg',
                     NNInterfaceWrapper(**DefaultParams.RealMLP_TD_REG),
                     tags=[tag], rerun=rerun)

    job_mgr.add_jobs(reg_task_infos, config_10_1_0,
                     'RealMLP-TD-S-reg',
                     NNInterfaceWrapper(**DefaultParams.RealMLP_TD_S_REG),
                     tags=[tag], rerun=rerun)

    job_mgr.add_jobs(class_task_infos, config_10_1_0, 'LGBM-TD-class',
                     LGBMInterfaceWrapper(**DefaultParams.LGBM_TD_CLASS),
                     tags=[tag], rerun=rerun)
    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'XGB-TD-class',
                     XGBInterfaceWrapper(**DefaultParams.XGB_TD_CLASS),
                     tags=[tag], rerun=rerun)
    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'CatBoost-TD-class',
                     CatBoostInterfaceWrapper(**DefaultParams.CB_TD_CLASS),
                     tags=[tag], rerun=rerun)

    # regression
    job_mgr.add_jobs(reg_task_infos, config_10_1_0,
                     'LGBM-TD-reg',
                     LGBMInterfaceWrapper(**DefaultParams.LGBM_TD_REG),
                     tags=[tag], rerun=rerun)
    job_mgr.add_jobs(reg_task_infos, config_10_1_0,
                     'XGB-TD-reg',
                     XGBInterfaceWrapper(**DefaultParams.XGB_TD_REG),
                     tags=[tag], rerun=rerun)
    job_mgr.add_jobs(reg_task_infos, config_10_1_0,
                     'CatBoost-TD-reg',
                     CatBoostInterfaceWrapper(**DefaultParams.CB_TD_REG),
                     tags=[tag], rerun=rerun)

    job_mgr.run_jobs(scheduler)


def run_default_ce_configs(paths: Paths, tag: str = 'paper_val_ce', rerun: bool = False):
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

    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'RealMLP-TD-class_val-ce',
                     NNInterfaceWrapper(
                         **utils.join_dicts(DefaultParams.RealMLP_TD_CLASS, dict(val_metric_name='cross_entropy'))),
                     tags=[tag], rerun=rerun)
    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'RealMLP-TD-class_val-ce_no-ls',
                     NNInterfaceWrapper(
                         **utils.join_dicts(DefaultParams.RealMLP_TD_CLASS,
                                            dict(val_metric_name='cross_entropy', use_ls=False, ls_eps=0.0))),
                     tags=[tag], rerun=rerun)
    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'RealMLP-TD-S-class_val-ce',
                     NNInterfaceWrapper(**utils.join_dicts(DefaultParams.RealMLP_TD_S_CLASS,
                                                           dict(val_metric_name='cross_entropy'))),
                     tags=[tag], rerun=rerun)
    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'RealMLP-TD-S-class_val-ce_no-ls',
                     NNInterfaceWrapper(
                         **utils.join_dicts(DefaultParams.RealMLP_TD_S_CLASS,
                                            dict(val_metric_name='cross_entropy', use_ls=False, ls_eps=0.0))),
                     tags=[tag], rerun=rerun)

    job_mgr.add_jobs(class_task_infos, config_10_1_0, 'LGBM-TD-class_val-ce',
                     LGBMInterfaceWrapper(
                         **utils.join_dicts(DefaultParams.LGBM_TD_CLASS, dict(val_metric_name='cross_entropy'))),
                     tags=[tag], rerun=rerun)
    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'XGB-TD-class_val-ce',
                     XGBInterfaceWrapper(
                         **utils.join_dicts(DefaultParams.XGB_TD_CLASS, dict(val_metric_name='cross_entropy'))),
                     tags=[tag], rerun=rerun)
    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'CatBoost-TD-class_val-ce',
                     CatBoostInterfaceWrapper(
                         **utils.join_dicts(DefaultParams.CB_TD_CLASS, dict(val_metric_name='cross_entropy'))),
                     tags=[tag], rerun=rerun)

    job_mgr.add_jobs(class_task_infos, config_10_1_0, 'LGBM-D-class_val-ce',
                     LGBMInterfaceWrapper(
                         **utils.join_dicts(DefaultParams.LGBM_D, dict(val_metric_name='cross_entropy'))),
                     tags=[tag], rerun=rerun)
    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'XGB-D-class_val-ce',
                     XGBInterfaceWrapper(
                         **utils.join_dicts(DefaultParams.XGB_D, dict(val_metric_name='cross_entropy'))),
                     tags=[tag], rerun=rerun)
    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'CatBoost-D-class_val-ce',
                     XGBInterfaceWrapper(
                         **utils.join_dicts(DefaultParams.CB_D, dict(val_metric_name='cross_entropy'))),
                     tags=[tag], rerun=rerun)

    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'XGB-PBB-D_val-ce',  # Probst, Boulestix, and Bischl, "Tunability: Importance of ..."
                     XGBInterfaceWrapper(n_estimators=4168, lr=0.018, min_child_weight=2.06,
                                         max_depth=13, reg_lambda=0.982, reg_alpha=1.113, subsample=0.839,
                                         colsample_bytree=0.752, colsample_bylevel=0.585,
                                         tree_method='hist', max_n_threads=64,
                                         val_metric_name='cross_entropy',
                                         tfms=['one_hot'], max_one_hot_cat_size=20),
                     tags=[tag], rerun=rerun)

    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'MLP-RTDL-D-class_val-ce',
                     MLPRTDLInterfaceWrapper(
                         **utils.join_dicts(DefaultParams.MLP_RTDL_D_CLASS_TabZilla,
                                            dict(val_metric_name='cross_entropy'))),
                     tags=[tag], rerun=rerun)
    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'MLP-PLR-D-class_val-ce',
                     MLPRTDLInterfaceWrapper(
                         **utils.join_dicts(DefaultParams.MLP_PLR_D_CLASS,
                                            dict(val_metric_name='cross_entropy'))),
                     tags=[tag], rerun=rerun)
    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'ResNet-RTDL-D-class_val-ce',
                     ResNetRTDLInterfaceWrapper(
                         **utils.join_dicts(DefaultParams.RESNET_RTDL_D_CLASS_TabZilla,
                                            dict(val_metric_name='cross_entropy'))),
                     tags=[tag], rerun=rerun)

    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'TabR-S-D-class_val-ce',
                     TabRInterfaceWrapper(
                         **utils.join_dicts(DefaultParams.TABR_S_D_CLASS,
                                            dict(val_metric_name='cross_entropy'))),
                     tags=[tag], rerun=rerun)
    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'RealTabR-D-class_val-ce',
                     TabRInterfaceWrapper(
                         **utils.join_dicts(DefaultParams.RealTABR_D_CLASS,
                                            dict(val_metric_name='cross_entropy'))),
                     tags=[tag], rerun=rerun)
    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'RealTabR-D-class_val-ce_no-ls',
                     TabRInterfaceWrapper(
                         **utils.join_dicts(DefaultParams.RealTABR_D_CLASS,
                                            dict(ls_eps=0.0, val_metric_name='cross_entropy'))),
                     tags=[tag], rerun=rerun)

    job_mgr.add_jobs(grinsztajn_class_task_infos + train_class_task_infos, config_10_1_0,
                     'FTT-D-class_val-ce',
                     FTTransformerInterfaceWrapper(
                         **utils.join_dicts(DefaultParams.FTT_D_CLASS,
                                            dict(val_metric_name='cross_entropy'))),
                     tags=[tag], rerun=rerun)

    job_mgr.run_jobs(scheduler)


def run_nns_no_ls(paths: Paths, tag: str = 'paper', rerun: bool = False):
    # this took around 48m
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

    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'RealMLP-TD-S-class_no-ls',
                     NNInterfaceWrapper(
                         **utils.join_dicts(DefaultParams.RealMLP_TD_S_CLASS,
                                            dict(use_ls=False, ls_eps=0.0))),
                     tags=[tag], rerun=rerun)
    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'RealMLP-TD-class_no-ls',
                     NNInterfaceWrapper(
                         **utils.join_dicts(DefaultParams.RealMLP_TD_CLASS,
                                            dict(use_ls=False, ls_eps=0.0))),
                     tags=[tag], rerun=rerun)

    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'RealTabR-D-class_no-ls',
                     TabRInterfaceWrapper(
                         **utils.join_dicts(DefaultParams.RealTABR_D_CLASS,
                                            dict(ls_eps=0.0))),
                     tags=[tag], rerun=rerun)

    job_mgr.run_jobs(scheduler)


def run_tabr_configs(paths: Paths, rerun: bool = False):
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

    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'RealTabR-D-class',
                     TabRInterfaceWrapper(**DefaultParams.RealTABR_D_CLASS),
                     tags=['paper'], rerun=rerun)

    job_mgr.add_jobs(reg_task_infos, config_10_1_0,
                     'RealTabR-D-reg',
                     TabRInterfaceWrapper(**DefaultParams.RealTABR_D_REG),
                     tags=['paper'], rerun=rerun)

    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'TabR-S-D-class_val-ce',
                     TabRInterfaceWrapper(
                         **utils.join_dicts(DefaultParams.TABR_S_D_CLASS,
                                            dict(val_metric_name='cross_entropy'))),
                     tags=['paper_val_ce'], rerun=rerun)
    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'TabR-S-D-class_rssc',
                     TabRInterfaceWrapper(
                         **utils.join_dicts(DefaultParams.TABR_S_D_CLASS,
                                            dict(tfms=['median_center', 'robust_scale', 'smooth_clip']))),
                     tags=['paper'], rerun=rerun)
    job_mgr.add_jobs(reg_task_infos, config_10_1_0,
                     'TabR-S-D-reg_rssc',
                     TabRInterfaceWrapper(
                         **utils.join_dicts(DefaultParams.TABR_S_D_REG,
                                            dict(tfms=['median_center', 'robust_scale', 'smooth_clip']))),
                     tags=['paper'], rerun=rerun)
    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'TabR-S-D-class',
                     TabRInterfaceWrapper(
                         **utils.join_dicts(DefaultParams.TABR_S_D_CLASS,
                                            dict())),
                     tags=['paper'], rerun=rerun)
    job_mgr.add_jobs(reg_task_infos, config_10_1_0,
                     'TabR-S-D-reg',
                     TabRInterfaceWrapper(
                         **utils.join_dicts(DefaultParams.TABR_S_D_REG,
                                            dict())),
                     tags=['paper'], rerun=rerun)

    job_mgr.run_jobs(scheduler)


def run_early_stopping_configs(paths: Paths, tag: str = 'paper_early_stopping', rerun: bool = False):
    # around 4h
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

    for esr in [10, 20, 50, 100, 300, 1000]:
        job_mgr.add_jobs(train_class_task_infos, config_10_1_0, f'LGBM-TD-class_esr-{esr}',
                         LGBMInterfaceWrapper(
                             **utils.join_dicts(DefaultParams.LGBM_TD_CLASS, dict(early_stopping_rounds=esr))),
                         tags=[tag], rerun=rerun)
        job_mgr.add_jobs(train_class_task_infos, config_10_1_0,
                         f'XGB-TD-class_esr-{esr}',
                         XGBInterfaceWrapper(
                             **utils.join_dicts(DefaultParams.XGB_TD_CLASS, dict(early_stopping_rounds=esr))),
                         tags=[tag], rerun=rerun)
        job_mgr.add_jobs(train_class_task_infos, config_10_1_0,
                         f'CatBoost-TD-class_esr-{esr}',
                         CatBoostInterfaceWrapper(
                             **utils.join_dicts(DefaultParams.CB_TD_CLASS, dict(early_stopping_rounds=esr))),
                         tags=[tag], rerun=rerun)

        # regression
        job_mgr.add_jobs(train_reg_task_infos, config_10_1_0,
                         f'LGBM-TD-reg_esr-{esr}',
                         LGBMInterfaceWrapper(
                             **utils.join_dicts(DefaultParams.LGBM_TD_REG, dict(early_stopping_rounds=esr))),
                         tags=[tag], rerun=rerun)
        job_mgr.add_jobs(train_reg_task_infos, config_10_1_0,
                         f'XGB-TD-reg_esr-{esr}',
                         XGBInterfaceWrapper(
                             **utils.join_dicts(DefaultParams.XGB_TD_REG, dict(early_stopping_rounds=esr))),
                         tags=[tag], rerun=rerun)
        job_mgr.add_jobs(train_reg_task_infos, config_10_1_0,
                         f'CatBoost-TD-reg_esr-{esr}',
                         CatBoostInterfaceWrapper(
                             **utils.join_dicts(DefaultParams.CB_TD_REG, dict(early_stopping_rounds=esr))),
                         tags=[tag], rerun=rerun)

    job_mgr.run_jobs(scheduler)


def run_brier_stopping_configs(paths: Paths, tag: str = 'paper_early_stopping', rerun: bool = False):
    # around 4h
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

    for esr in [10, 20, 50, 100, 300, 1000]:
        # for esr in [300]:
        job_mgr.add_jobs(train_class_task_infos, config_10_1_0, f'LGBM-TD-class_val-brier_esr-{esr}',
                         LGBMInterfaceWrapper(
                             **utils.join_dicts(DefaultParams.LGBM_TD_CLASS,
                                                dict(early_stopping_rounds=esr, val_metric_name='brier'))),
                         tags=[tag], rerun=rerun)
        job_mgr.add_jobs(train_class_task_infos, config_10_1_0,
                         f'XGB-TD-class_val-brier_esr-{esr}',
                         XGBInterfaceWrapper(
                             **utils.join_dicts(DefaultParams.XGB_TD_CLASS,
                                                dict(early_stopping_rounds=esr, val_metric_name='brier'))),
                         tags=[tag], rerun=rerun)
        job_mgr.add_jobs(train_class_task_infos, config_10_1_0,
                         f'CatBoost-TD-class_val-brier_esr-{esr}',
                         CatBoostInterfaceWrapper(
                             **utils.join_dicts(DefaultParams.CB_TD_CLASS,
                                                dict(early_stopping_rounds=esr, val_metric_name='brier'))),
                         tags=[tag], rerun=rerun)

    job_mgr.run_jobs(scheduler)


def run_cross_entropy_stopping_configs(paths: Paths, tag: str = 'paper_early_stopping', rerun: bool = False):
    # around 4h
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

    for esr in [10, 20, 50, 100, 300, 1000]:
        # for esr in [300]:
        job_mgr.add_jobs(train_class_task_infos, config_10_1_0, f'LGBM-TD-class_val-ce_esr-{esr}',
                         LGBMInterfaceWrapper(
                             **utils.join_dicts(DefaultParams.LGBM_TD_CLASS,
                                                dict(early_stopping_rounds=esr, val_metric_name='cross_entropy'))),
                         tags=[tag], rerun=rerun)
        job_mgr.add_jobs(train_class_task_infos, config_10_1_0,
                         f'XGB-TD-class_val-ce_esr-{esr}',
                         XGBInterfaceWrapper(
                             **utils.join_dicts(DefaultParams.XGB_TD_CLASS,
                                                dict(early_stopping_rounds=esr, val_metric_name='cross_entropy'))),
                         tags=[tag], rerun=rerun)
        job_mgr.add_jobs(train_class_task_infos, config_10_1_0,
                         f'CatBoost-TD-class_val-ce_esr-{esr}',
                         CatBoostInterfaceWrapper(
                             **utils.join_dicts(DefaultParams.CB_TD_CLASS,
                                                dict(early_stopping_rounds=esr, val_metric_name='cross_entropy'))),
                         tags=[tag], rerun=rerun)

    job_mgr.run_jobs(scheduler)


def run_ensemble_configs(paths: Paths, tag: str = 'paper', rerun: bool = False):
    # around 20 minutes or so
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

    for task_infos, task_type_name in [(class_task_infos, 'class'), (reg_task_infos, 'reg')]:
        for alg_group_name, alg_names in get_ensemble_groups(task_type_name).items():
            job_mgr.add_jobs(task_infos, config_10_1_0, f'Ensemble{alg_group_name}',
                             CaruanaEnsembleWrapper([LoadResultsWrapper(alg_name) for alg_name in alg_names]),
                             tags=[tag], rerun=rerun)

    job_mgr.run_jobs(scheduler)


def run_realmlp_hpo_alg_selection(paths: Paths, n_hpo_steps: int, tag: str = 'paper', rerun: bool = False):
    job_mgr = TabBenchJobManager(paths)
    scheduler = SimpleJobScheduler(RayJobManager(max_n_threads=32))
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

    alg_names = [f'RealMLP-HPO_step-{i}' for i in range(n_hpo_steps)]

    for task_infos, val_metric_name in [(reg_task_infos, 'rmse'), (class_task_infos, 'class_error')]:
        run_alg_selection(paths, config_10_1_0, task_infos, f'RealMLP-HPO', alg_names, val_metric_name)

    run_alg_selection(paths, config_10_1_0, class_task_infos, f'RealMLP-HPO_best-1-auc-ovr', alg_names, '1-auc_ovr')

    msd_alg_names = [f'RealMLP-HPO-moresigmadim_step-{i}' for i in range(n_hpo_steps)]
    for task_infos, val_metric_name in [(train_reg_task_infos, 'rmse'), (train_class_task_infos, 'class_error')]:
        run_alg_selection(paths, config_10_1_0, task_infos, f'RealMLP-HPO-moresigmadim', msd_alg_names, val_metric_name,
                          tags=[tag])


def run_rtdl_hpo_alg_selection(paths: Paths, n_hpo_steps: int, tag: str = 'paper', rerun: bool = False):
    job_mgr = TabBenchJobManager(paths)
    scheduler = SimpleJobScheduler(RayJobManager(max_n_threads=32))
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

    alg_names = [f'MLP-RTDL-HPO_step-{i}' for i in range(n_hpo_steps)]
    plr_alg_names = [f'MLP-PLR-HPO_step-{i}' for i in range(n_hpo_steps)]
    resnet_alg_names = [f'ResNet-RTDL-HPO_step-{i}' for i in range(n_hpo_steps)]
    ftt_alg_names = [f'FTT-HPO_step-{i}' for i in range(n_hpo_steps)]

    for task_infos, val_metric_name in [(reg_task_infos, 'rmse'), (class_task_infos, 'class_error')]:
        run_alg_selection(paths, config_10_1_0, task_infos, f'MLP-RTDL-HPO', alg_names, val_metric_name)
        run_alg_selection(paths, config_10_1_0, task_infos, f'MLP-PLR-HPO', plr_alg_names, val_metric_name)
        run_alg_selection(paths, config_10_1_0, task_infos, f'ResNet-RTDL-HPO', resnet_alg_names, val_metric_name)

    for task_infos, val_metric_name in [(grinsztajn_reg_task_infos, 'rmse'), (grinsztajn_class_task_infos, 'class_error')]:
        run_alg_selection(paths, config_10_1_0, task_infos, f'FTT-HPO', ftt_alg_names, val_metric_name)

    run_alg_selection(paths, config_10_1_0, class_task_infos, f'MLP-RTDL-HPO_best-1-auc-ovr', alg_names, '1-auc_ovr')
    run_alg_selection(paths, config_10_1_0, class_task_infos, f'MLP-PLR-HPO_best-1-auc-ovr', plr_alg_names, '1-auc_ovr')
    run_alg_selection(paths, config_10_1_0, class_task_infos, f'ResNet-RTDL-HPO_best-1-auc-ovr', resnet_alg_names,
                      '1-auc_ovr')
    run_alg_selection(paths, config_10_1_0, grinsztajn_class_task_infos, f'FTT-HPO_best-1-auc-ovr', ftt_alg_names,
                      '1-auc_ovr')


def run_tabr_hpo_alg_selection(paths: Paths, n_hpo_steps: int, tag: str = 'paper', rerun: bool = False):
    job_mgr = TabBenchJobManager(paths)
    scheduler = SimpleJobScheduler(RayJobManager(max_n_threads=32))
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

    alg_names = [f'TabR-HPO_step-{i}' for i in range(n_hpo_steps)]
    realtabr_alg_names = [f'RealTabR-HPO_step-{i}' for i in range(n_hpo_steps)]

    for task_infos, val_metric_name in [(grinsztajn_reg_task_infos, 'rmse'),
                                        (grinsztajn_class_task_infos, 'class_error')]:
        run_alg_selection(paths, config_10_1_0, task_infos, f'TabR-HPO', alg_names, val_metric_name, tags=[tag],
                          rerun=rerun)
        run_alg_selection(paths, config_10_1_0, task_infos, f'RealTabR-HPO', realtabr_alg_names, val_metric_name,
                          tags=[tag],
                          rerun=rerun)

    run_alg_selection(paths, config_10_1_0, grinsztajn_class_task_infos, f'TabR-HPO_best-1-auc-ovr', alg_names,
                      '1-auc_ovr', tags=[tag], rerun=rerun)
    run_alg_selection(paths, config_10_1_0, grinsztajn_class_task_infos, f'RealTabR-HPO_best-1-auc-ovr',
                      realtabr_alg_names,
                      '1-auc_ovr', tags=[tag], rerun=rerun)


def run_gbdt_hpo_alg_selection(paths: Paths, n_hpo_steps: int, tag: str = 'paper', rerun: bool = False):
    job_mgr = TabBenchJobManager(paths)
    scheduler = SimpleJobScheduler(RayJobManager(max_n_threads=16))
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

    for gbdt_name in ['XGB', 'LGBM', 'CatBoost']:
        alg_names = [f'{gbdt_name}-HPO_step-{i}' for i in range(n_hpo_steps)]
        job_mgr.add_jobs(all_task_infos, config_10_1_0, f'{gbdt_name}-HPO',
                         AlgorithmSelectionWrapper([LoadResultsWrapper(alg_name) for alg_name in alg_names]),
                         tags=[tag], rerun=rerun)
        job_mgr.add_jobs(class_task_infos, config_10_1_0, f'{gbdt_name}-HPO_best-1-auc-ovr',
                         AlgorithmSelectionWrapper([LoadResultsWrapper(alg_name) for alg_name in alg_names],
                                                   alg_sel_metric_name='1-auc_ovr'),
                         tags=[tag], rerun=rerun)

    job_mgr.run_jobs(scheduler)


def run_rf_hpo_alg_selection(paths: Paths, n_hpo_steps: int, tag: str = 'paper', rerun: bool = False):
    job_mgr = TabBenchJobManager(paths)
    scheduler = SimpleJobScheduler(RayJobManager(max_n_threads=16))
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

    for task_infos, val_metric_name in [(grinsztajn_reg_task_infos, 'rmse'),
                                        (grinsztajn_class_task_infos, 'class_error')]:
        alg_names = [f'RF-HPO_step-{i}' for i in range(n_hpo_steps)]
        run_alg_selection(paths, config_10_1_0, task_infos, f'RF-HPO', alg_names, val_metric_name, tags=[tag],
                          rerun=rerun)
    run_alg_selection(paths, config_10_1_0, task_infos, f'RF-HPO_best-1-auc-ovr', alg_names, '1-auc_ovr', tags=[tag],
                      rerun=rerun)

    job_mgr.run_jobs(scheduler)


def run_rtdl_default_configs(paths: Paths, tag: str = 'paper', rerun: bool = False, with_mlp: bool = True,
                             with_resnet: bool = True, only_meta_train: bool = False, only_meta_test: bool = False,
                             tabzilla_defaults: bool = True, with_plr: bool = True, with_ftt: bool = True):
    # ca 50 min for meta-train-reg
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

    if only_meta_train:
        class_task_infos = train_class_task_infos
        reg_task_infos = train_reg_task_infos
    elif only_meta_test:
        class_task_infos = test_class_task_infos
        reg_task_infos = test_reg_task_infos

    if with_resnet:
        job_mgr.add_jobs(class_task_infos, config_10_1_0,
                         'ResNet-RTDL-D-class_grinsztajn' if not tabzilla_defaults else
                         'ResNet-RTDL-D-class',
                         ResNetRTDLInterfaceWrapper(
                             **DefaultParams.RESNET_RTDL_D_CLASS_Grinsztajn if not tabzilla_defaults else
                             DefaultParams.RESNET_RTDL_D_CLASS_TabZilla),
                         tags=[tag], rerun=rerun)
        job_mgr.add_jobs(reg_task_infos, config_10_1_0,
                         'ResNet-RTDL-D-reg_grinsztajn' if not tabzilla_defaults else
                         'ResNet-RTDL-D-reg',
                         ResNetRTDLInterfaceWrapper(
                             **DefaultParams.RESNET_RTDL_D_REG_Grinsztajn if not tabzilla_defaults else
                             DefaultParams.RESNET_RTDL_D_REG_TabZilla),
                         tags=[tag], rerun=rerun)

    if with_mlp:
        job_mgr.add_jobs(class_task_infos, config_10_1_0,
                         'MLP-RTDL-D-class_grinsztajn' if not tabzilla_defaults else
                         'MLP-RTDL-D-class',
                         MLPRTDLInterfaceWrapper(
                             **DefaultParams.MLP_RTDL_D_CLASS_Grinsztajn if not tabzilla_defaults else
                             DefaultParams.MLP_RTDL_D_CLASS_TabZilla),
                         tags=[tag], rerun=rerun)
        job_mgr.add_jobs(reg_task_infos, config_10_1_0,
                         'MLP-RTDL-D-reg_grinsztajn' if not tabzilla_defaults else
                         'MLP-RTDL-D-reg',
                         MLPRTDLInterfaceWrapper(
                             **DefaultParams.MLP_RTDL_D_REG_Grinsztajn if not tabzilla_defaults else
                             DefaultParams.MLP_RTDL_D_REG_TabZilla),
                         tags=[tag], rerun=rerun)

    if with_plr:
        job_mgr.add_jobs(class_task_infos, config_10_1_0,
                         'MLP-PLR-D-class',
                         MLPRTDLInterfaceWrapper(
                             **DefaultParams.MLP_PLR_D_CLASS),
                         tags=[tag], rerun=rerun)
        job_mgr.add_jobs(reg_task_infos, config_10_1_0,
                         'MLP-PLR-D-reg',
                         MLPRTDLInterfaceWrapper(
                             **DefaultParams.MLP_PLR_D_REG),
                         tags=[tag], rerun=rerun)

    if with_ftt:
        job_mgr.add_jobs(grinsztajn_class_task_infos + train_class_task_infos, config_10_1_0,
                         'FTT-D-class',
                         FTTransformerInterfaceWrapper(
                             **DefaultParams.FTT_D_CLASS),
                         tags=[tag], rerun=rerun)
        job_mgr.add_jobs(grinsztajn_reg_task_infos + train_reg_task_infos, config_10_1_0,
                         'FTT-D-reg',
                         FTTransformerInterfaceWrapper(
                             **DefaultParams.FTT_D_REG),
                         tags=[tag], rerun=rerun)

    job_mgr.run_jobs(scheduler)


def run_rtdl_rssc_default_configs(paths: Paths, tag: str = 'paper', rerun: bool = False, with_mlp: bool = True,
                                  with_resnet: bool = True, with_plr: bool = True, with_tabr: bool = True, with_ftt: bool = True,
                                  only_meta_train: bool = False, only_meta_test: bool = False):
    # ca 50 min for meta-train-reg (without TabR/FTT)
    # ca 8h30m for FTT (on meta-train + grinsztajn)
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

    if only_meta_train:
        class_task_infos = train_class_task_infos
        reg_task_infos = train_reg_task_infos
    elif only_meta_test:
        class_task_infos = test_class_task_infos
        reg_task_infos = test_reg_task_infos

    if with_resnet:
        job_mgr.add_jobs(class_task_infos, config_10_1_0,
                         'ResNet-RTDL-D-class_rssc',
                         ResNetRTDLInterfaceWrapper(
                             **utils.join_dicts(DefaultParams.RESNET_RTDL_D_CLASS_TabZilla,
                                                dict(tfms=['median_center', 'robust_scale', 'smooth_clip']))),
                         tags=[tag], rerun=rerun)
        job_mgr.add_jobs(reg_task_infos, config_10_1_0,
                         'ResNet-RTDL-D-reg_rssc',
                         ResNetRTDLInterfaceWrapper(
                             **utils.join_dicts(DefaultParams.RESNET_RTDL_D_REG_TabZilla,
                                                dict(tfms=['median_center', 'robust_scale', 'smooth_clip']))),
                         tags=[tag], rerun=rerun)

    if with_mlp:
        job_mgr.add_jobs(class_task_infos, config_10_1_0,
                         'MLP-RTDL-D-class_rssc',
                         MLPRTDLInterfaceWrapper(
                             **utils.join_dicts(DefaultParams.MLP_RTDL_D_CLASS_TabZilla,
                                                dict(tfms=['median_center', 'robust_scale', 'smooth_clip']))),
                         tags=[tag], rerun=rerun)
        job_mgr.add_jobs(reg_task_infos, config_10_1_0,
                         'MLP-RTDL-D-reg_rssc',
                         MLPRTDLInterfaceWrapper(
                             **utils.join_dicts(DefaultParams.MLP_RTDL_D_REG_TabZilla,
                                                dict(tfms=['median_center', 'robust_scale', 'smooth_clip']))),
                         tags=[tag], rerun=rerun)

    if with_plr:
        job_mgr.add_jobs(class_task_infos, config_10_1_0,
                         'MLP-PLR-D-class_rssc',
                         MLPRTDLInterfaceWrapper(
                             **utils.join_dicts(DefaultParams.MLP_PLR_D_CLASS,
                                                dict(tfms=['median_center', 'robust_scale', 'smooth_clip']))),
                         tags=[tag], rerun=rerun)
        job_mgr.add_jobs(reg_task_infos, config_10_1_0,
                         'MLP-PLR-D-reg_rssc',
                         MLPRTDLInterfaceWrapper(
                             **utils.join_dicts(DefaultParams.MLP_PLR_D_REG,
                                                dict(tfms=['median_center', 'robust_scale', 'smooth_clip']))),
                         tags=[tag], rerun=rerun)

    if with_tabr:
        job_mgr.add_jobs(class_task_infos, config_10_1_0,
                         'TabR-S-D-class_rssc',
                         TabRInterfaceWrapper(
                             **utils.join_dicts(DefaultParams.TABR_S_D_CLASS,
                                                dict(tfms=['median_center', 'robust_scale', 'smooth_clip']))),
                         tags=[tag], rerun=rerun)
        job_mgr.add_jobs(reg_task_infos, config_10_1_0,
                         'TabR-S-D-reg_rssc',
                         TabRInterfaceWrapper(
                             **utils.join_dicts(DefaultParams.TABR_S_D_REG,
                                                dict(tfms=['median_center', 'robust_scale', 'smooth_clip']))),
                         tags=[tag], rerun=rerun)

    if with_ftt:
        job_mgr.add_jobs(grinsztajn_class_task_infos + train_class_task_infos, config_10_1_0,
                         'FTT-D-class_rssc',
                         FTTransformerInterfaceWrapper(
                             **utils.join_dicts(DefaultParams.FTT_D_CLASS,
                                                dict(tfms=['median_center', 'robust_scale', 'smooth_clip']))),
                         tags=[tag], rerun=rerun)
        job_mgr.add_jobs(grinsztajn_reg_task_infos + train_reg_task_infos, config_10_1_0,
                         'FTT-D-reg_rssc',
                         FTTransformerInterfaceWrapper(
                             **utils.join_dicts(DefaultParams.FTT_D_REG,
                                                dict(tfms=['median_center', 'robust_scale', 'smooth_clip']))),
                         tags=[tag], rerun=rerun)

    job_mgr.run_jobs(scheduler)


def run_tabr_default_configs(paths: Paths, tag: str = 'paper', rerun: bool = False,
                             only_meta_train: bool = False, only_meta_test: bool = False,
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

    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'TabR-S-D-class',
                     TabRInterfaceWrapper(
                         **DefaultParams.TABR_S_D_CLASS),
                     tags=[tag], rerun=rerun)
    job_mgr.add_jobs(reg_task_infos, config_10_1_0,
                     'TabR-S-D-reg',
                     TabRInterfaceWrapper(
                         **DefaultParams.TABR_S_D_REG),
                     tags=[tag], rerun=rerun)

    job_mgr.run_jobs(scheduler)


def run_default_configs(paths: Paths, tag: str = 'paper', rerun: bool = False):
    # took 12h55s
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

    job_mgr.add_jobs(all_task_infos, config_10_1_0, 'LGBM-D',
                     LGBMInterfaceWrapper(**DefaultParams.LGBM_D),
                     tags=[tag], rerun=rerun)
    job_mgr.add_jobs(all_task_infos, config_10_1_0,
                     'XGB-D',
                     XGBInterfaceWrapper(**DefaultParams.XGB_D),
                     tags=[tag], rerun=rerun)
    job_mgr.add_jobs(all_task_infos, config_10_1_0,
                     'CatBoost-D',
                     CatBoostInterfaceWrapper(**DefaultParams.CB_D),
                     tags=[tag], rerun=rerun)

    # it was too bad to include in the plots
    # job_mgr.add_jobs(all_task_infos, config_10_1_0,
    #                  'MLP-SKL-D',
    #                  SklearnMLPInterfaceWrapper(tfms=['mean_center', 'l2_normalize', 'one_hot']),
    #                  tags=[tag], rerun=rerun)

    job_mgr.add_jobs(all_task_infos, config_10_1_0,
                     'RF-SKL-D',
                     RFInterfaceWrapper(tfms=['ordinal_encoding'], permute_ordinal_encoding=True),
                     tags=[tag, 'paper_val_ce'], rerun=rerun)

    job_mgr.add_jobs(class_task_infos, config_10_1_0,
                     'XGB-PBB-D',  # Probst, Boulestix, and Bischl, "Tunability: Importance of ..."
                     XGBInterfaceWrapper(n_estimators=4168, lr=0.018, min_child_weight=2.06,
                                         max_depth=13, reg_lambda=0.982, reg_alpha=1.113, subsample=0.839,
                                         colsample_bytree=0.752, colsample_bylevel=0.585,
                                         tree_method='hist', max_n_threads=64,
                                         tfms=['one_hot'], max_one_hot_cat_size=20),
                     tags=['paper'])

    job_mgr.run_jobs(scheduler)


def run_gbdts_hpo_tpe(paths: Paths, n_estimators: int = 1000, early_stopping_rounds: int = 300,
                      tag: str = 'paper'):
    # this generates about 10GB of data
    # took 7h17m for n_estimators=2
    # took about 6h30m for n_estimators=1  (but slightly more tasks were run for that because of the rerun=True)
    job_mgr = TabBenchJobManager(paths)
    scheduler = SimpleJobScheduler(RayJobManager())
    config_10_1_0 = RunConfig(n_tt_splits=10, n_cv=1, n_refit=0, save_y_pred=True)
    config_5_1_0 = RunConfig(n_tt_splits=5, n_cv=1, n_refit=0, save_y_pred=True)

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

    for task_infos, config in [(train_task_infos, config_10_1_0), (test_task_infos, config_10_1_0)]:
        job_mgr.add_jobs(task_infos, config, f'XGB-HPO-TPE',
                         XGBHyperoptInterfaceWrapper(n_estimators=n_estimators, n_hyperopt_steps=50,
                                                     early_stopping_rounds=early_stopping_rounds,
                                                     tree_method='hist', space='grinsztajn'),
                         tags=[tag])
        job_mgr.add_jobs(task_infos, config, f'CatBoost-HPO-TPE',
                         CatBoostHyperoptInterfaceWrapper(n_estimators=n_estimators, n_hyperopt_steps=50,
                                                          early_stopping_rounds=early_stopping_rounds,
                                                          space='shwartz-ziv'),
                         tags=[tag])
        job_mgr.add_jobs(task_infos, config, f'LGBM-HPO-TPE',
                         LGBMHyperoptInterfaceWrapper(n_estimators=n_estimators, n_hyperopt_steps=50,
                                                      early_stopping_rounds=early_stopping_rounds,
                                                      space='catboost_quality_benchmarks'),
                         tags=[tag])

    job_mgr.run_jobs(scheduler)


def run_preprocessing_experiments(paths: Paths, tag: str = 'paper_preprocessing'):
    # this took 7h9m for just two different scikit-learn based transformation configurations!
    job_mgr = TabBenchJobManager(paths)
    scheduler = SimpleJobScheduler(RayJobManager())
    config_10_1_0 = RunConfig(n_tt_splits=10, n_cv=1, n_refit=0, save_y_pred=True)

    train_class_task_infos = TaskCollection.from_name('meta-train-class', paths).load_infos(paths)
    train_reg_task_infos = TaskCollection.from_name('meta-train-reg', paths).load_infos(paths)

    for task_infos, defaults in [(train_class_task_infos, DefaultParams.RealMLP_TD_S_CLASS),
                                 (train_reg_task_infos, DefaultParams.RealMLP_TD_S_REG)]:
        job_mgr.add_jobs(task_infos, config_10_1_0, 'RealMLP-TD-S_tfms-mc-rs-sc-oh',
                         NNInterfaceWrapper(**utils.update_dict(defaults, dict(
                             tfms=['median_center', 'robust_scale', 'smooth_clip', 'one_hot']
                         ))),
                         [tag])
        job_mgr.add_jobs(task_infos, config_10_1_0, 'RealMLP-TD-S_tfms-mc-rs-oh',
                         NNInterfaceWrapper(**utils.update_dict(defaults, dict(
                             tfms=['median_center', 'robust_scale', 'one_hot']
                         ))),
                         [tag])
        job_mgr.add_jobs(task_infos, config_10_1_0, 'RealMLP-TD-S_tfms-std-oh',
                         NNInterfaceWrapper(**utils.update_dict(defaults, dict(
                             tfms=['mean_center', 'l2_normalize', 'one_hot'],
                             l2_normalize_eps=1e-30,
                         ))),
                         [tag])
        job_mgr.add_jobs(task_infos, config_10_1_0, 'RealMLP-TD-S_tfms-std-sc-oh',
                         NNInterfaceWrapper(**utils.update_dict(defaults, dict(
                             tfms=['mean_center', 'l2_normalize', 'smooth_clip', 'one_hot'],
                             l2_normalize_eps=1e-30,
                         ))),
                         [tag])
        job_mgr.add_jobs(task_infos, config_10_1_0, 'RealMLP-TD-S_tfms-kdi1-oh',
                         NNInterfaceWrapper(**utils.update_dict(defaults, dict(
                             tfms=['kdi', 'one_hot'], kdi_alpha=1.0,
                             max_n_vectorized=1,
                         ))),
                         [tag])
        job_mgr.add_jobs(task_infos, config_10_1_0, 'RealMLP-TD-S_tfms-quantile-oh',
                         NNInterfaceWrapper(**utils.update_dict(defaults, dict(
                             tfms=['quantile', 'one_hot'],
                             max_n_vectorized=1,
                         ))),
                         [tag])
        job_mgr.add_jobs(task_infos, config_10_1_0, 'RealMLP-TD-S_tfms-quantiletabr-oh',
                         NNInterfaceWrapper(**utils.update_dict(defaults, dict(
                             tfms=['quantile_tabr', 'one_hot'],
                             max_n_vectorized=1,
                         ))),
                         [tag])

    job_mgr.run_jobs(scheduler)


def run_all_ablations(paths: Paths, with_class: bool = True, with_reg: bool = True):
    run_ablations(paths, {
        'default': dict(),
    }, with_class=with_class, with_reg=with_reg)
    run_ablations(paths, {
        'lr-cos-decay': dict(lr_sched='cos'),
        'lr-constant': dict(lr_sched='constant'),
    }, with_class=with_class, with_reg=with_reg)
    run_ablations(paths, {
        'wd-0.0': dict(wd=0.0, wd_sched='constant', bias_wd_factor=0.0),
        'wd-0.02': dict(wd=0.02, wd_sched='constant', bias_wd_factor=0.0),
    }, with_class=with_class, with_reg=with_reg)
    # run_ablations(paths, {
    #     'wd-0.01-flatcos': dict(wd=0.01, wd_sched='flat_cos', bias_wd_factor=0.0),
    #     'wd-0.01': dict(wd=0.01, wd_sched='constant', bias_wd_factor=0.0),
    # }, with_class=False, with_reg=with_reg)
    # run_ablations(paths, {
    #     'wd-0.0': dict(wd=0.0, wd_sched='constant', bias_wd_factor=0.0),
    #     'wd-0.01': dict(wd=0.01, wd_sched='constant', bias_wd_factor=0.0),
    # }, with_class=with_class, with_reg=False)
    run_ablations(paths, {
        'pdrop-0.0': dict(p_drop=0.0, p_drop_sched='constant'),
        'pdrop-0.15': dict(p_drop=0.15, p_drop_sched='constant'),
    }, with_class=with_class, with_reg=with_reg)
    run_ablations(paths, {
        'no-front-scale': dict(first_layer_config=dict()),
    }, with_class=with_class, with_reg=with_reg)
    run_ablations(paths, {
        'normal-init': dict(bias_init_mode='zeros', weight_init_mode='normal'),
    }, with_class=with_class, with_reg=with_reg)
    run_ablations(paths, {
        'standard-param_no-wd': dict(weight_param='standard', bias_lr_factor=1 / 16, weight_lr_factor=1 / 16, wd=0.0),
    }, with_class=with_class, with_reg=with_reg)
    run_ablations(paths, {
        'non-parametric-act': dict(use_parametric_act=False),
    }, with_class=with_class, with_reg=with_reg)
    run_ablations(paths, {
        'act-relu': dict(act='relu'),
        'act-mish': dict(act='mish')
    }, with_class=with_class, with_reg=False)
    run_ablations(paths, {
        'act-relu': dict(act='relu'),
        'act-selu': dict(act='selu')
    }, with_class=False, with_reg=with_reg)
    run_ablations(paths, {
        'no-label-smoothing': dict(use_ls=False, ls_eps=0.0),
    }, with_class=with_class, with_reg=False)
    run_ablations(paths, {
        'num-embeddings-plr': dict(plr_act_name='relu', plr_use_densenet=False, plr_use_cos_bias=False),
        'num-embeddings-pl': dict(plr_act_name='linear', plr_use_densenet=False, plr_use_cos_bias=False),
        'num-embeddings-none': dict(use_plr_embeddings=False)
    }, with_class=with_class, with_reg=with_reg)
    run_ablations(paths, {
        'beta2-0.999': dict(sq_mom=0.999),
    }, with_class=with_class, with_reg=with_reg)
    run_ablations(paths, {
        'first-best-epoch': dict(use_last_best_epoch=False),
    }, with_class=with_class, with_reg=with_reg)
    run_ablations(paths, {
        'no-cat-embs': dict(max_one_hot_cat_size=-1),
    }, with_class=with_class, with_reg=with_reg)


def run_architecture_ablations(paths: Paths, tag: str = 'paper', rerun: bool = False,
                               only_meta_train: bool = False, only_meta_test: bool = False,
                               start_split: int = 0, end_split: int = 10):
    # ca 1h45m + 40m + 2h
    job_mgr = TabBenchJobManager(paths)
    scheduler = SimpleJobScheduler(RayJobManager())
    config_10_1_0 = RunConfig(n_tt_splits=end_split, min_split_idx=start_split, n_cv=1, n_refit=0, save_y_pred=True)

    train_class_task_infos = TaskCollection.from_name('meta-train-class', paths).load_infos(paths)
    train_reg_task_infos = TaskCollection.from_name('meta-train-reg', paths).load_infos(paths)
    test_class_task_infos = TaskCollection.from_name('meta-test-class', paths).load_infos(paths)
    test_reg_task_infos = TaskCollection.from_name('meta-test-reg', paths).load_infos(paths)

    class_task_infos = train_class_task_infos + test_class_task_infos
    reg_task_infos = train_reg_task_infos + test_reg_task_infos
    train_task_infos = train_class_task_infos + train_reg_task_infos
    test_task_infos = test_class_task_infos + test_reg_task_infos
    all_task_infos = class_task_infos + reg_task_infos

    if only_meta_train:
        class_task_infos = train_class_task_infos
        reg_task_infos = train_reg_task_infos
    elif only_meta_test:
        class_task_infos = test_class_task_infos
        reg_task_infos = test_reg_task_infos

    lr_grid_std = [1.5e-3, 7e-4, 1e-3, 4e-4, 2.5e-3, 4e-3, 7e-3, 1e-2, 1.5e-2]
    lr_grid_ntp = [0.04, 0.2, 0.1, 0.02, 0.07, 0.01, 0.3, 0.03, 0.4]

    mlp_rtdl_repr_config_class = dict(
        hidden_sizes=[128, 256, 128],
        p_drop=0.1,
        block_str='w-b-a-d',
        lr=2.5e-3,  # will be changed later
        opt='adam',
        tfms=['median_center', 'robust_scale', 'smooth_clip', 'embedding'],
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
        use_last_best_epoch=False,
        emb_init_mode='kaiming-uniform-t',
    )

    mlp_rtdl_repr_config_reg = utils.join_dicts(mlp_rtdl_repr_config_class, dict(
        normalize_output=True, lr=1.5e-3))

    mlp_rtdl_num_emb_repr_config_class = utils.join_dicts(mlp_rtdl_repr_config_class, dict(
        num_emb_type='plr', plr_sigma=0.1, plr_hidden_1=16, plr_hidden_2=4, plr_lr_factor=0.1,
        # todo: or pl embeddings?
        lr=2.5e-3,
    ))

    mlp_rtdl_num_emb_repr_config_reg = utils.join_dicts(mlp_rtdl_num_emb_repr_config_class,
                                                        dict(normalize_output=True, lr=7e-4))

    mlp_rtdl_pl_config_class = utils.join_dicts(mlp_rtdl_num_emb_repr_config_class, dict(
        num_emb_type='pl', lr=4e-3))
    mlp_rtdl_pl_config_reg = utils.join_dicts(mlp_rtdl_pl_config_class, dict(normalize_output=True, lr=4e-4))

    realmlp_arch_class = dict(
        hidden_sizes=[128, 256, 128],
        p_drop=0.1,
        block_str='w-b-a-d',
        opt='adam',
        tfms=['median_center', 'robust_scale', 'smooth_clip', 'embedding'],
        embedding_size=8,
        batch_size=128,
        n_epochs=1000,
        use_early_stopping=True,
        early_stopping_multiplicative_patience=1,
        early_stopping_additive_patience=20,

        weight_init_mode='uniform',
        weight_init_gain=1. / np.sqrt(3.),
        bias_init_mode='pytorch-default',
        use_last_best_epoch=False,
        emb_init_mode='kaiming-uniform-t',
        lr=2e-2,
        num_emb_type='pbld', plr_sigma=0.1, plr_hidden_1=16, plr_hidden_2=4, plr_lr_factor=0.1,
        weight_param='ntk', bias_lr_factor=0.1,
        act='selu',
        use_parametric_act=True, act_lr_factor=0.1,
        add_front_scale=True, scale_lr_factor=6.0,
    )

    realmlp_arch_reg = utils.join_dicts(realmlp_arch_class, dict(act='mish', normalize_output=True, lr=1e-2))

    def add_jobs(name: str, config_class: dict, config_reg: dict, lr_grid: List[float], with_meta_test: bool = True):
        for task_infos, all_task_infos, task_type_name, config in [
            (train_class_task_infos, class_task_infos, 'class', config_class),
            (train_reg_task_infos, reg_task_infos, 'reg', config_reg)]:
            for lr in lr_grid:
                job_mgr.add_jobs(task_infos, config_10_1_0,
                                 f'{name}_lr-{lr:g}',
                                 NNInterfaceWrapper(**utils.update_dict(config, dict(lr=lr))),
                                 tags=['paper_arch-lr-tuning'], rerun=rerun)

            if with_meta_test:
                job_mgr.add_jobs(all_task_infos, config_10_1_0,
                                 f'{name}',
                                 NNInterfaceWrapper(**config),
                                 tags=['paper'], rerun=rerun)

    add_jobs('MLP-RTDL-reprod', mlp_rtdl_repr_config_class, mlp_rtdl_repr_config_reg, lr_grid_std)
    add_jobs('MLP-RTDL-reprod-plr', mlp_rtdl_num_emb_repr_config_class, mlp_rtdl_num_emb_repr_config_reg, lr_grid_std,
             with_meta_test=False)
    add_jobs('MLP-RTDL-reprod-pl', mlp_rtdl_pl_config_class, mlp_rtdl_pl_config_reg, lr_grid_std)
    add_jobs('MLP-RTDL-reprod-RealMLP-arch', realmlp_arch_class, realmlp_arch_reg, lr_grid_ntp)

    job_mgr.run_jobs(scheduler)


def run_cumulative_ablations_new(paths: Paths, n_lrs: int = -1, tag: str = 'paper_cumulative_ablations_new',
                                 rerun: bool = False):
    job_mgr = TabBenchJobManager(paths)
    scheduler = SimpleJobScheduler(RayJobManager())
    config_10_1_0 = RunConfig(n_tt_splits=10, n_cv=1, n_refit=0, save_y_pred=False)  # todo: it's false

    train_class_task_infos = TaskCollection.from_name('meta-train-class', paths).load_infos(paths)
    train_reg_task_infos = TaskCollection.from_name('meta-train-reg', paths).load_infos(paths)

    # lr_grid_ntp = [0.01, 0.015, 0.025, 0.04, 0.07, 0.1, 0.2, 0.3, 0.4]
    # lr_grid_std = [4e-4, 7e-4, 1e-3, 1.5e-3, 2.5e-3, 4e-3, 7e-3, 1e-2, 2e-2]
    lr_grid_std = [1.5e-3, 7e-4, 1e-3, 4e-4, 2.5e-3, 4e-3, 7e-3, 1e-2, 1.5e-2]
    lr_grid_ntp = [0.04, 0.2, 0.1, 0.02, 0.07, 0.01, 0.3, 0.03, 0.4]

    if n_lrs > 0:
        lr_grid_std = lr_grid_std[:n_lrs]
        lr_grid_ntp = lr_grid_ntp[:n_lrs]

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
                                     f'MLP-cumul-abl-new-{ablation_counter}-{task_type_name}_{name}_lr-{lr:g}',
                                     NNInterfaceWrapper(**utils.update_dict(config, dict(lr=lr))),
                                     tags=[tag], rerun=rerun)

        ablation_counter += 1

    vanilla_config_class = dict(
        hidden_sizes=[256] * 3,
        p_drop=0.0,
        block_str='w-b-a-d',
        opt='adam',
        tfms=['quantile', 'embedding'],
        embedding_size=8,
        batch_size=256,
        n_epochs=256,
        use_early_stopping=True,
        early_stopping_multiplicative_patience=1,
        early_stopping_additive_patience=40,
        act='relu',
        weight_param='standard',
        weight_init_mode='uniform',
        weight_init_gain=1. / np.sqrt(3.),
        bias_init_mode='pytorch-default',
        max_n_vectorized=1,
        use_last_best_epoch=False,
    )

    add_config('vanilla', lr_grid_std, add=vanilla_config_class,
               add_reg=dict(normalize_output=True), run_this=True)
    # quantile_tabr was not well-suited for vectorization, now we can vectorize
    add_config('robust-scale-smooth-clip', lr_grid_std,
               dict(tfms=['median_center', 'robust_scale', 'smooth_clip', 'embedding'],
                    max_n_vectorized=50))
    add_config('one-hot-small-cat', lr_grid_std,
               dict(tfms=['one_hot', 'median_center', 'robust_scale', 'smooth_clip', 'embedding'],
                    max_one_hot_cat_size=9))
    add_config('no-early-stop', lr_grid_std, dict(use_early_stopping=False))
    add_config('last-best-epoch', lr_grid_std, dict(use_last_best_epoch=True))
    add_config('lr-multi-cycle', lr_grid_std, dict(lr_sched='coslog4'))
    add_config('beta2-0.95', lr_grid_std, dict(sq_mom=0.95))
    add_config('label-smoothing', lr_grid_std, add_class=dict(use_ls=True, ls_eps=0.1))
    add_config('output-clipping', lr_grid_std, add_reg=dict(clamp_output=True))
    add_config('ntp', lr_grid_ntp, dict(weight_param='ntk', bias_lr_factor=0.1))
    add_config('different-act', lr_grid_ntp, add_class=dict(act='selu'), add_reg=dict(act='mish'))
    add_config('param-act', lr_grid_ntp, dict(use_parametric_act=True, act_lr_factor=0.1))
    add_config('front-scale', lr_grid_ntp, dict(add_front_scale=True, scale_lr_factor=6.0))
    add_config('num-emb-pl', lr_grid_ntp,
               dict(num_emb_type='pl', plr_sigma=0.1, plr_hidden_1=16, plr_hidden_2=4, plr_lr_factor=0.1))
    add_config('num-emb-pbld', lr_grid_ntp, dict(num_emb_type='pbld'))
    add_config('alt-pdrop-0.15', lr_grid_ntp, dict(p_drop=0.15))
    add_config('alt-pdrop-flat-cos', lr_grid_ntp, dict(p_drop_sched='flat_cos'))
    add_config('alt-wd-0.02', lr_grid_ntp, dict(wd=0.02, bias_wd_factor=0.0))
    add_config('alt-wd-flat-cos', lr_grid_ntp, dict(wd_sched='flat_cos'))
    add_config('alt-bias-init-he+5', lr_grid_ntp, dict(bias_init_mode='he+5'))
    add_config('alt-weight-init-std', lr_grid_ntp, dict(weight_init_mode='std', weight_init_gain=1.0))

    # add_config('bias-init-he+5', lr_grid_ntp, dict(bias_init_mode='he+5'))
    # add_config('weight-init-std', lr_grid_ntp, dict(weight_init_mode='std', weight_init_gain=1.0))
    # add_config('pdrop-0.15', lr_grid_ntp, dict(p_drop=0.15))
    # add_config('pdrop-flat-cos', lr_grid_ntp, dict(p_drop_sched='flat_cos'))
    # add_config('wd-0.02', lr_grid_ntp, dict(wd=0.02, bias_wd_factor=0.0))
    # add_config('wd-flat-cos', lr_grid_ntp, dict(wd_sched='flat_cos'))

    job_mgr.run_jobs(scheduler)
    pass


if __name__ == '__main__':
    paths = Paths.from_env_variables()

    run_td_configs(paths, tag='paper', rerun=False)
    run_default_configs(paths, tag='paper', rerun=False)
    run_rtdl_default_configs(paths, tag='paper', tabzilla_defaults=True)
    run_tabr_configs(paths)

    run_gbdt_rs_configs()
    run_rf_rs_configs()
    for i in range(50):
        if (i + 1) % 10 == 0:
            run_rtdl_tuning_configs(paths, n_steps=i + 1, with_resnet=True, only_meta_train=False)
    for i in range(50):
        if (i + 1) % 10 == 0:
            run_realmlp_tuning_configs(paths, n_steps=i + 1, tag='paper_mlp-hpo', rerun=False)
    for n_steps in [1, 2, 5, 10, 20, 30, 40, 50]:
        run_tabr_tuning_configs(paths, n_steps=n_steps)

    run_rtdl_hpo_alg_selection(paths, n_hpo_steps=50, tag='paper')
    run_gbdt_hpo_alg_selection(paths, n_hpo_steps=50, tag='paper')
    run_rf_hpo_alg_selection(paths, n_hpo_steps=50, tag='paper')
    run_realmlp_hpo_alg_selection(paths, n_hpo_steps=50, tag='paper', rerun=False)
    run_tabr_hpo_alg_selection(paths, n_hpo_steps=50)
    run_ensemble_configs(paths, tag='paper')

    # ----- ablations (mostly for the appendix) -----

    for n_lrs in [10]:  # range(1, 10):
        run_cumulative_ablations_new(paths, n_lrs=n_lrs)

    run_rtdl_rssc_default_configs(paths, tag='paper')
    run_default_ce_configs(paths)
    run_nns_no_ls(paths)

    run_all_ablations(paths)
    run_architecture_ablations(paths)
    run_preprocessing_experiments(paths)
    run_refit_configs(paths, tag='paper', rerun=False)
    run_early_stopping_configs(paths)
    run_brier_stopping_configs(paths)
    run_cross_entropy_stopping_configs(paths)
    run_refit_configs(paths, tag='paper', rerun=False)

