import time

import numpy as np
import torch

from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskPackage, TaskDescription
from pytabkit.bench.scheduling.resources import NodeResources
from pytabkit.models import utils
from pytabkit.models.sklearn.default_params import DefaultParams
from pytabkit.models.training.logging import StdoutLogger
from pytabkit.bench.alg_wrappers.interface_wrappers import NNInterfaceWrapper, MLPRTDLInterfaceWrapper, ResNetRTDLInterfaceWrapper, \
    TabRInterfaceWrapper
from pytabkit.bench.alg_wrappers.extra_interface_wrappers import IterativeImportanceNNInterfaceWrapper, \
    IterativeWeightNNInterfaceWrapper, IterativeReinitNNInterfaceWrapper
from pytabkit.models.training.metrics import Metrics


def run_example(paths: Paths):
    start_time = time.time()
    use_gpu = torch.cuda.is_available()

    wrapper = NNInterfaceWrapper(**utils.join_dicts(DefaultParams.RealMLP_TD_REG))

    task_info = TaskDescription('uci-reg', 'parkinson_motor').load_info(paths)

    print('n_samples:', task_info.n_samples)
    print('n_cont:', task_info.tensor_infos['x_cont'].get_n_features())
    print('x_cat cat sizes:', task_info.tensor_infos['x_cat'].get_cat_sizes())
    print('n_classes:', task_info.tensor_infos['y'].get_cat_sizes())
    if task_info.tensor_infos['y'].get_cat_sizes() > 0:
        class_frequencies = torch.bincount(task_info.load_task(paths).ds.tensors['y'].squeeze(-1))
        print(f'class frequencies: {class_frequencies.numpy()}')

    is_nn = (isinstance(wrapper, NNInterfaceWrapper) or isinstance(wrapper, MLPRTDLInterfaceWrapper)
             or isinstance(wrapper, ResNetRTDLInterfaceWrapper)
             or isinstance(wrapper, IterativeImportanceNNInterfaceWrapper)
             or isinstance(wrapper, IterativeWeightNNInterfaceWrapper)
             or isinstance(wrapper, IterativeReinitNNInterfaceWrapper)
             or isinstance(wrapper, TabRInterfaceWrapper))
    use_gpu = use_gpu and is_nn

    print(f'Running on task {task_info.task_desc}')
    if is_nn:
        split_infos = task_info.get_random_splits(10)[0:1]
        task_package = TaskPackage(task_info, split_infos=split_infos, n_cv=1, n_refit=0, paths=paths, rerun=False,
                                   alg_name='test', save_y_pred=False)
    else:
        split_infos = task_info.get_random_splits(10)[1:2]
        task_package = TaskPackage(task_info, split_infos=split_infos, n_cv=1, n_refit=0, paths=paths, rerun=True,
                                   alg_name='test', save_y_pred=False)
    logger = StdoutLogger(verbosity_level=2)
    metric_name = Metrics.default_eval_metric_name(task_info.task_type)
    required_resources = wrapper.get_required_resources(task_package)
    print(f'Predicted time usage in s: {required_resources.time_s:g}')
    print(f'Predicted CPU RAM usage in GB: {required_resources.cpu_ram_gb:g}')
    print(f'Requested n_threads: {required_resources.n_threads:g}')

    # metric_name = '1-auroc'
    gpu_usages = np.array([1.0]) if use_gpu and is_nn else np.array([], dtype=np.float32)
    gpu_rams_gb = np.array([5.0]) if use_gpu and is_nn else np.array([], dtype=np.float32)
    tmp_folders = [paths.results_alg_task_split(task_package.task_info.task_desc,
                                                alg_name=task_package.alg_name, n_cv=task_package.n_cv,
                                                split_type=split_info.split_type,
                                                split_id=split_info.id) / 'tmp' for split_info in
                   task_package.split_infos]
    result_managers = wrapper.run(task_package, logger,
                                  assigned_resources=NodeResources(node_id=0, n_threads=16.0, cpu_ram_gb=2.0,
                                                                   gpu_usages=gpu_usages,
                                                                   gpu_rams_gb=gpu_rams_gb,
                                                                   physical_core_usages=np.array([0.0])),
                                  tmp_folders=tmp_folders)

    for rm in result_managers:
        print(rm.metrics_dict)
        print(rm.other_dict)

    result_pairs = [('val', [rm.metrics_dict['cv']['val']['1']['0'][metric_name] for rm in result_managers])]
    for is_cv in [True, False] if task_package.n_refit > 0 else [True]:
        cv_str = 'cv' if is_cv else 'refit'
        max_n_models = task_package.n_cv if is_cv else task_package.n_refit
        for n_models in {1, max_n_models}:  # use a set in case max_n_models == 1
            try:
                name = 'test-' + cv_str + '-' + str(n_models)
                results = [rm.metrics_dict[cv_str]['test'][str(n_models)][str(start_idx)][metric_name]
                           for rm in result_managers for start_idx in range(1 if n_models > 1 else max_n_models)]
                result_pairs.append((name, results))
            except KeyError as e:
                print(e)
                pass  # might happen if wrapper is not a randomized alg and therefore does not do ensembling
    for name, results in result_pairs:
        print(f'Mean {name} error: {np.mean(results):g} +- {np.std(results) / np.sqrt(len(results)):g}')

    # for rm in rms:
    #     print('val:', rm.val_dict)
    #     print('test:', rm.test_dict)

    print(f'Time: {time.time() - start_time:g} s')


if __name__ == '__main__':
    run_example(Paths.from_env_variables())
