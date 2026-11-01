from pytabkit.bench.alg_wrappers.interface_wrappers import XGBInterfaceWrapper, LGBMInterfaceWrapper, \
    CatBoostInterfaceWrapper, TabPFN2InterfaceWrapper, NNInterfaceWrapper, TabICLInterfaceWrapper, RFInterfaceWrapper, \
    TabMInterfaceWrapper, ExtraTreesInterfaceWrapper, DummyInterfaceWrapper, LinearModelInterfaceWrapper
from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskCollection
from pytabkit.bench.run.task_execution import RunConfig, TabBenchJobManager
from pytabkit.bench.scheduling.execution import RayJobManager
from pytabkit.bench.scheduling.schedulers import SimpleJobScheduler
from pytabkit.models.sklearn.default_params import DefaultParams


def run_fewshot_experiments():
    paths = Paths.from_env_variables()
    config = RunConfig(n_tt_splits=2, n_cv=4, n_refit=0, max_trainval_samples=16, save_y_pred=True)
    job_mgr = TabBenchJobManager(paths)
    scheduler = SimpleJobScheduler(RayJobManager())

    class_task_infos = TaskCollection.from_name('talent-class-tiny', paths).load_infos(paths)
    reg_task_infos = TaskCollection.from_name('talent-reg-tiny', paths).load_infos(paths)

    tags = ['fewshot']

    val_metric_name = 'cross_entropy'

    # todo: xgb-pbb-d?
    # todo: linear/logistic?
    job_mgr.add_jobs(class_task_infos, config,
                     'Dummy-class_fewshot',
                     DummyInterfaceWrapper(tfms=['ordinal_encoding']), tags=tags)
    job_mgr.add_jobs(class_task_infos, config,
                     'Linear-class_fewshot',
                     LinearModelInterfaceWrapper(tfms=['mean_center', 'l2_normalize', 'one_hot']),
                     tags=['default'])
    job_mgr.add_jobs(class_task_infos, config,
                     'RF-D-class_fewshot',
                     RFInterfaceWrapper(**DefaultParams.RF_SKL_D, val_metric_name=val_metric_name), tags=tags)
    job_mgr.add_jobs(class_task_infos, config,
                     'ExtraTrees-D-class_fewshot',
                     ExtraTreesInterfaceWrapper(tfms=['ordinal_encoding'], permute_ordinal_encoding=True), tags=tags)
    job_mgr.add_jobs(class_task_infos, config,
                     'XGB-TD-class_fewshot',
                     XGBInterfaceWrapper(**DefaultParams.XGB_TD_CLASS, val_metric_name=val_metric_name), tags=tags)
    job_mgr.add_jobs(class_task_infos, config,
                     'XGB-PBB-class_fewshot',
                     XGBInterfaceWrapper(**DefaultParams.XGB_PBB_CLASS, val_metric_name=val_metric_name), tags=tags)
    job_mgr.add_jobs(class_task_infos, config,
                     'LGBM-TD-class_fewshot',
                     LGBMInterfaceWrapper(**DefaultParams.LGBM_TD_CLASS, val_metric_name=val_metric_name), tags=tags)
    job_mgr.add_jobs(class_task_infos, config,  # following https://arxiv.org/pdf/2411.04324
                     'LGBM-fewshot-class_fewshot',
                     LGBMInterfaceWrapper(n_estimators=100, extra_trees=True, num_leaves=4, lr=0.05, min_data_in_leaf=1,
                                          colsample_bytree=0.5, subsample=0.5, bagging_freq=1,
                                          min_data_per_group=1, cat_l2=0, cat_smooth=0, max_cat_to_onehot=100,
                                          val_metric_name=val_metric_name), tags=tags)
    job_mgr.add_jobs(class_task_infos, config,
                     'CatBoost-TD-class_fewshot',
                     CatBoostInterfaceWrapper(**DefaultParams.CB_TD_CLASS, val_metric_name=val_metric_name), tags=tags)
    job_mgr.add_jobs(class_task_infos, config,  # todo: disable label smoothing?
                     'RealMLP-TD-class_fewshot',
                     NNInterfaceWrapper(**DefaultParams.RealMLP_TD_CLASS, val_metric_name=val_metric_name), tags=tags)
    job_mgr.add_jobs(class_task_infos, config,  # todo: disable label smoothing?
                     'RealMLP-TD-tabarena-class_fewshot',
                     NNInterfaceWrapper(**(DefaultParams.RealMLP_TD_CLASS | dict(use_ls=False, n_ens=8)),
                                        val_metric_name=val_metric_name), tags=tags)
    job_mgr.add_jobs(class_task_infos, config,  #  refit not implemented
                     'TabM-D-class_fewshot',
                     TabMInterfaceWrapper(**DefaultParams.TABM_D_CLASS, val_metric_name=val_metric_name), tags=tags)
    job_mgr.add_jobs(class_task_infos, config,
                     'RealTabPFN-2.5-class_fewshot',
                     TabPFN2InterfaceWrapper(tabpfn_version='2.5'),
                     tags=tags)
    job_mgr.add_jobs(class_task_infos, config,
                     'TabICLv2-class_fewshot',
                     TabICLInterfaceWrapper(),
                     tags=tags)

    # ----- regression -----

    job_mgr.add_jobs(reg_task_infos, config,
                     'Dummy-reg_fewshot',
                     DummyInterfaceWrapper(tfms=['ordinal_encoding']), tags=tags)
    job_mgr.add_jobs(reg_task_infos, config,
                     'Linear-reg_fewshot',
                     LinearModelInterfaceWrapper(tfms=['mean_center', 'l2_normalize', 'one_hot']), tags=tags)
    job_mgr.add_jobs(reg_task_infos, config,
                     'RF-D-reg_fewshot',
                     RFInterfaceWrapper(**DefaultParams.RF_SKL_D), tags=tags)
    job_mgr.add_jobs(reg_task_infos, config,
                     'ExtraTrees-D-reg_fewshot',
                     ExtraTreesInterfaceWrapper(tfms=['ordinal_encoding'], permute_ordinal_encoding=True), tags=tags)
    job_mgr.add_jobs(reg_task_infos, config,
                     'XGB-TD-reg_fewshot',
                     XGBInterfaceWrapper(**DefaultParams.XGB_TD_REG), tags=tags)
    job_mgr.add_jobs(reg_task_infos, config,
                     'XGB-PBB-reg_fewshot',
                     XGBInterfaceWrapper(**DefaultParams.XGB_PBB_CLASS), tags=tags)
    job_mgr.add_jobs(reg_task_infos, config,
                     'LGBM-TD-reg_fewshot',
                     LGBMInterfaceWrapper(**DefaultParams.LGBM_TD_REG), tags=tags)
    job_mgr.add_jobs(reg_task_infos, config,  # following https://arxiv.org/pdf/2411.04324
                     'LGBM-fewshot-reg_fewshot',
                     LGBMInterfaceWrapper(n_estimators=100, extra_trees=True, num_leaves=4, lr=0.05, min_data_in_leaf=1,
                                          colsample_bytree=0.5, subsample=0.5, bagging_freq=1,
                                          min_data_per_group=1, cat_l2=0, cat_smooth=0, max_cat_to_onehot=100),
                     tags=tags)
    job_mgr.add_jobs(reg_task_infos, config,
                     'CatBoost-TD-reg_fewshot',
                     CatBoostInterfaceWrapper(**DefaultParams.CB_TD_REG), tags=tags)
    job_mgr.add_jobs(reg_task_infos, config,
                     'RealMLP-TD-reg_fewshot',
                     NNInterfaceWrapper(**DefaultParams.RealMLP_TD_REG), tags=tags)
    job_mgr.add_jobs(reg_task_infos, config,
                     'RealMLP-TD-tabarena-reg_fewshot',
                     NNInterfaceWrapper(**(DefaultParams.RealMLP_TD_REG | dict(n_ens=8))), tags=tags)
    job_mgr.add_jobs(reg_task_infos, config,  #  refit not implemented
                     'TabM-D-reg_fewshot',
                     TabMInterfaceWrapper(**DefaultParams.TABM_D_REG), tags=tags)
    job_mgr.add_jobs(reg_task_infos, config,
                     'RealTabPFN-2.5-reg_fewshot',
                     TabPFN2InterfaceWrapper(tabpfn_version='2.5'),
                     tags=tags)
    job_mgr.add_jobs(reg_task_infos, config,
                     'TabICLv2-reg_fewshot',
                     TabICLInterfaceWrapper(),
                     tags=tags)

    job_mgr.run_jobs(scheduler)

if __name__ == '__main__':
    run_fewshot_experiments()
