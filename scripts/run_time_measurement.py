import random
import time
import torch

import numpy as np
import sklearn

from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskInfo, TaskCollection
from pytabkit.models import utils
from pytabkit.models.data.splits import RandomSplitter
from pytabkit.models.sklearn.sklearn_base import AlgInterfaceEstimator
from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier, CatBoost_TD_Classifier, \
    LGBM_TD_Classifier, \
    XGB_TD_Classifier, LGBM_D_Classifier, CatBoost_D_Classifier, XGB_D_Classifier, LGBM_HPO_Classifier, \
    CatBoost_HPO_Classifier, \
    XGB_HPO_Classifier, RealMLP_HPO_Classifier, XGB_PBB_D_Classifier, RF_SKL_D_Classifier, MLP_SKL_D_Classifier, \
    MLP_SKL_D_Regressor, \
    RF_SKL_D_Regressor, RealMLP_HPO_Regressor, XGB_HPO_Regressor, CatBoost_HPO_Regressor, LGBM_HPO_Regressor, \
    XGB_D_Regressor, \
    CatBoost_D_Regressor, LGBM_D_Regressor, RealMLP_TD_Regressor, RealMLP_TD_S_Regressor, RealMLP_TD_S_Classifier, \
    XGB_TD_Regressor, \
    CatBoost_TD_Regressor, LGBM_TD_Regressor, MLP_RTDL_D_Classifier, Resnet_RTDL_D_Classifier, MLP_RTDL_D_Regressor, \
    Resnet_RTDL_D_Regressor, TabR_S_D_Classifier, TabR_S_D_Regressor, MLP_RTDL_HPO_Classifier, \
    MLP_RTDL_HPO_Regressor, XGB_HPO_TPE_Regressor, LGBM_HPO_TPE_Regressor, \
    CatBoost_HPO_TPE_Regressor, XGB_HPO_TPE_Classifier, LGBM_HPO_TPE_Classifier, CatBoost_HPO_TPE_Classifier


def measure_times(paths: Paths, alg_name: str, estimator: AlgInterfaceEstimator, coll_name: str, device: str,
                  rerun: bool = False, n_predict_reps: int = 20) -> None:
    task_infos = TaskCollection.from_name(coll_name, paths).load_infos(paths)
    times_list = []
    for task_info in task_infos:
        file_path = paths.times_alg_task(alg_name=alg_name, task_desc=task_info.task_desc) / 'times.yaml'
        if utils.existsFile(file_path) and not rerun:
            times_list.append(utils.deserialize(file_path, use_yaml=True))
            # print(f'Results exist already')
            continue

        print(f'Measuring time for alg {alg_name} on task {task_info.task_desc}: ', end='')
        estimator: AlgInterfaceEstimator = sklearn.base.clone(estimator)
        estimator.device = device

        task = task_info.load_task(paths)
        ds = task.ds
        seed = task_info.n_samples
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        trainval_test_split = RandomSplitter(seed).split_ds(ds)
        trainval_ds, test_ds = trainval_test_split.get_sub_ds(0), trainval_test_split.get_sub_ds(1)
        train_val_split = RandomSplitter(seed + 1, first_fraction=0.75).split_ds(trainval_ds)
        val_idxs = train_val_split.get_sub_idxs(1).numpy()
        x_trainval = trainval_ds.without_labels().to_df()
        y_trainval = trainval_ds.tensors['y'].numpy().squeeze(-1)
        x_test = test_ds.without_labels().to_df()

        start_time = time.time()
        estimator.fit(x_trainval, y_trainval, val_idxs=val_idxs)
        end_time = time.time()
        fit_time = end_time - start_time

        start_time = time.time()
        for i in range(n_predict_reps):
            estimator.predict(x_test)
        end_time = time.time()
        predict_time = (end_time - start_time) / n_predict_reps

        times = {'fit_time': fit_time, 'predict_time': predict_time}
        utils.serialize(file_path, times, use_yaml=True)
        times_list.append(times)

        print(f'{fit_time=:g}s, {predict_time=:g}s')

    avg_fit_time = np.mean([times['fit_time'] for times in times_list])
    avg_predict_time = np.mean([times['predict_time'] for times in times_list])
    print(f'Average times for {alg_name} on {coll_name}: {avg_fit_time=:g}s, {avg_predict_time=:g}s')


def measure_times_cpu_class(n_threads: int, rerun: bool = False):
    paths = Paths.from_env_variables()
    estimators = {
        'LGBM-TD_CPU': LGBM_TD_Classifier(n_threads=n_threads, verbosity=-1),
        'CatBoost-TD_CPU': CatBoost_TD_Classifier(n_threads=n_threads),
        'XGB-TD_CPU': XGB_TD_Classifier(n_threads=n_threads),
        'RealMLP-TD_CPU': RealMLP_TD_Classifier(n_threads=n_threads),
        'RealMLP-TD-S_CPU': RealMLP_TD_S_Classifier(n_threads=n_threads),
        'LGBM-D_CPU': LGBM_D_Classifier(n_threads=n_threads, verbosity=-1),
        'CatBoost-D_CPU': CatBoost_D_Classifier(n_threads=n_threads),
        'XGB-D_CPU': XGB_D_Classifier(n_threads=n_threads),
        'RF-SKL-D_CPU': RF_SKL_D_Classifier(n_threads=n_threads),
        'MLP-SKL-D_CPU': MLP_SKL_D_Classifier(n_threads=n_threads),
        'MLP-RTDL-D_CPU': MLP_RTDL_D_Classifier(n_threads=n_threads),
        'ResNet-RTDL-D_CPU': Resnet_RTDL_D_Classifier(n_threads=n_threads),
        'XGB-PBB-D_CPU': XGB_PBB_D_Classifier(n_threads=n_threads),
        'RealMLP-HPO-2_CPU': RealMLP_HPO_Classifier(n_threads=n_threads, n_hyperopt_steps=2),
        'MLP-RTDL-HPO-2_CPU': MLP_RTDL_HPO_Classifier(n_threads=n_threads, n_hyperopt_steps=2),
        'XGB-HPO-TPE_CPU': XGB_HPO_TPE_Classifier(n_threads=n_threads),
        'LGBM-HPO-TPE_CPU': LGBM_HPO_TPE_Classifier(n_threads=n_threads, verbosity=-1),
        'CatBoost-HPO-TPE_CPU': CatBoost_HPO_TPE_Classifier(n_threads=n_threads),
        'XGB-HPO-2_CPU': XGB_HPO_Classifier(n_threads=n_threads, n_hyperopt_steps=2),
        'LGBM-HPO-2_CPU': LGBM_HPO_Classifier(n_threads=n_threads, verbosity=-1, n_hyperopt_steps=2),
        'CatBoost-HPO-2_CPU': CatBoost_HPO_Classifier(n_threads=n_threads, n_hyperopt_steps=2),
        'TabR-S-D_CPU': TabR_S_D_Classifier(n_threads=n_threads),

        'LGBM-D_val-ce_CPU': LGBM_D_Classifier(n_threads=n_threads, val_metric_name='cross_entropy', verbosity=-1),
        'XGB-D_val-ce_CPU': XGB_D_Classifier(n_threads=n_threads, val_metric_name='cross_entropy'),
        'CatBoost-D_val-ce_CPU': CatBoost_D_Classifier(n_threads=n_threads, val_metric_name='cross_entropy'),
        'LGBM-TD_val-ce_CPU': LGBM_TD_Classifier(n_threads=n_threads, val_metric_name='cross_entropy', verbosity=-1),
        'XGB-TD_val-ce_CPU': XGB_TD_Classifier(n_threads=n_threads, val_metric_name='cross_entropy'),
        'CatBoost-TD_val-ce_CPU': CatBoost_TD_Classifier(n_threads=n_threads, val_metric_name='cross_entropy'),
        'XGB-PBB-D_val-ce_CPU': XGB_PBB_D_Classifier(n_threads=n_threads, val_metric_name='cross_entropy'),
        'RealMLP-TD_val-ce_no-ls_CPU': RealMLP_TD_Classifier(val_metric_name='cross_entropy',
                                                             use_ls=False, n_threads=n_threads),
        'RealMLP-TD-S_val-ce_no-ls_CPU': RealMLP_TD_S_Classifier(val_metric_name='cross_entropy',
                                                                 use_ls=False, n_threads=n_threads),
        'RealMLP-TD_no-ls_CPU': RealMLP_TD_Classifier(device='cpu',
                                                      use_ls=False, n_threads=n_threads),
        'RealMLP-TD-S_no-ls_CPU': RealMLP_TD_S_Classifier(device='cpu',
                                                          use_ls=False, n_threads=n_threads),
        'RealMLP-TD_val-ce_CPU': RealMLP_TD_Classifier(val_metric_name='cross_entropy',
                                                       n_threads=n_threads),
        'RealMLP-TD-S_val-ce_CPU': RealMLP_TD_S_Classifier(val_metric_name='cross_entropy',
                                                           n_threads=n_threads),
        'MLP-RTDL-D_val-ce_CPU': MLP_RTDL_D_Classifier(val_metric_name='cross_entropy',
                                                       n_threads=n_threads),
        'ResNet-RTDL-D_val-ce_CPU': Resnet_RTDL_D_Classifier(val_metric_name='cross_entropy',
                                                             n_threads=n_threads),
        'TabR-S-D_val-ce_CPU': TabR_S_D_Classifier(val_metric_name='cross_entropy',
                                                   n_threads=n_threads),

        'MLP-RTDL-D_rssc_CPU': MLP_RTDL_D_Classifier(n_threads=n_threads,
                                                     tfms=['median_center', 'robust_scale', 'smooth_clip']),
        'ResNet-RTDL-D_rssc_CPU': Resnet_RTDL_D_Classifier(n_threads=n_threads,
                                                           tfms=['median_center', 'robust_scale', 'smooth_clip']),
        'TabR-S-D_rssc_CPU': TabR_S_D_Classifier(n_threads=n_threads,
                                                 tfms=['median_center', 'robust_scale', 'smooth_clip']),
    }

    for alg_name, estimator in estimators.items():
        measure_times(paths, alg_name=alg_name, estimator=estimator, coll_name='meta-train-class', device='cpu',
                      rerun=rerun)


def measure_times_cpu_reg(n_threads: int, rerun: bool = False):
    paths = Paths.from_env_variables()
    estimators = {
        'LGBM-TD_CPU': LGBM_TD_Regressor(n_threads=n_threads, verbosity=-1),
        'CatBoost-TD_CPU': CatBoost_TD_Regressor(n_threads=n_threads),
        'XGB-TD_CPU': XGB_TD_Regressor(n_threads=n_threads),
        'RealMLP-TD_CPU': RealMLP_TD_Regressor(n_threads=n_threads),
        'RealMLP-TD-S_CPU': RealMLP_TD_S_Regressor(n_threads=n_threads),
        'LGBM-D_CPU': LGBM_D_Regressor(n_threads=n_threads, verbosity=-1),
        'CatBoost-D_CPU': CatBoost_D_Regressor(n_threads=n_threads),
        'XGB-D_CPU': XGB_D_Regressor(n_threads=n_threads),
        'RF-SKL-D_CPU': RF_SKL_D_Regressor(n_threads=n_threads),
        'MLP-SKL-D_CPU': MLP_SKL_D_Regressor(n_threads=n_threads),
        'MLP-RTDL-D_CPU': MLP_RTDL_D_Regressor(n_threads=n_threads),
        'ResNet-RTDL-D_CPU': Resnet_RTDL_D_Regressor(n_threads=n_threads),
        'RealMLP-HPO-2_CPU': RealMLP_HPO_Regressor(n_threads=n_threads, n_hyperopt_steps=2),
        'MLP-RTDL-HPO-2_CPU': MLP_RTDL_HPO_Regressor(n_threads=n_threads, n_hyperopt_steps=2),
        'XGB-HPO-2_CPU': XGB_HPO_Regressor(n_threads=n_threads, n_hyperopt_steps=2),
        'LGBM-HPO-2_CPU': LGBM_HPO_Regressor(n_threads=n_threads, verbosity=-1, n_hyperopt_steps=2),
        'CatBoost-HPO-2_CPU': CatBoost_HPO_Regressor(n_threads=n_threads, n_hyperopt_steps=2),
        'XGB-HPO-TPE_CPU': XGB_HPO_TPE_Regressor(n_threads=n_threads),
        'LGBM-HPO-TPE_CPU': LGBM_HPO_TPE_Regressor(n_threads=n_threads, verbosity=-1),
        'CatBoost-HPO-TPE_CPU': CatBoost_HPO_TPE_Regressor(n_threads=n_threads),
        'TabR-S-D_CPU': TabR_S_D_Regressor(n_threads=n_threads),
    }

    for alg_name, estimator in estimators.items():
        measure_times(paths, alg_name=alg_name, estimator=estimator, coll_name='meta-train-reg', device='cpu',
                      rerun=rerun)


def measure_times_gpu_class(n_threads: int, rerun: bool = False):
    paths = Paths.from_env_variables()
    # todo: add XGB-GPU and CatBoost-GPU?
    estimators = {
        'MLP-TD_GPU': RealMLP_TD_Classifier(device='cuda:0', n_threads=n_threads),
        'MLP-TD-S_GPU': RealMLP_TD_S_Classifier(device='cuda:0', n_threads=n_threads),
        'MLP-HPO-2_GPU': RealMLP_HPO_Classifier(device='cuda:0', n_threads=n_threads, n_hyperopt_steps=2),
    }

    import torch
    # have torch cuda initialization before running the first NN
    _ = torch.zeros(1, device='cuda:0')

    for alg_name, estimator in estimators.items():
        measure_times(paths, alg_name=alg_name, estimator=estimator, coll_name='meta-train-class', device='cuda:0',
                      rerun=rerun)


def measure_times_gpu_reg(n_threads: int, rerun: bool = False):
    paths = Paths.from_env_variables()
    estimators = {
        'MLP-TD_GPU': RealMLP_TD_Regressor(device='cuda:0', n_threads=n_threads),
        'MLP-TD-S_GPU': RealMLP_TD_S_Regressor(device='cuda:0', n_threads=n_threads),
        'MLP-HPO-2_GPU': RealMLP_HPO_Regressor(device='cuda:0', n_threads=n_threads, n_hyperopt_steps=2),
    }

    import torch
    # have torch cuda initialization before running the first NN
    _ = torch.zeros(1, device='cuda:0')

    for alg_name, estimator in estimators.items():
        measure_times(paths, alg_name=alg_name, estimator=estimator, coll_name='meta-train-reg', device='cuda:0',
                      rerun=rerun)


if __name__ == '__main__':
    # may take a day or so on a good CPU
    n_threads = 32
    measure_times_cpu_class(n_threads=n_threads, rerun=False)
    measure_times_cpu_reg(n_threads=n_threads, rerun=False)
    # measure_times_gpu_class(n_threads=n_threads, rerun=False)  # not used in the paper
    # measure_times_gpu_reg(n_threads=n_threads, rerun=False)  # not used in the paper
