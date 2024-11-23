from pathlib import Path

from sklearn.datasets import make_classification
import torch

from pytabkit import XGB_TD_Classifier
from pytabkit.bench.alg_wrappers.interface_wrappers import XGBInterfaceWrapper
from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskDescription, TaskInfo, Task, TaskCollection
from pytabkit.bench.run.task_execution import TabBenchJobManager, RunConfig
from pytabkit.bench.scheduling.execution import RayJobManager
from pytabkit.bench.scheduling.schedulers import SimpleJobScheduler
from pytabkit.models import utils
from pytabkit.models.data.data import TensorInfo, DictDataset
from pytabkit.models.sklearn.default_params import DefaultParams


# Running this test before the sklearn tests can cause an error in the pickling test for NNs using skorch:
# _pickle.PicklingError: Can't pickle <built-in function print>: it's not the same object as builtins.print
# The error occurs when ray.init() and FunctionProcess() are both used.

# def test_bench_simple(tmp_path: Path):
#     paths = Paths(base_folder=str(tmp_path/'tab_bench_data'))
#
#     # ----- import dataset -----
#
#     n_samples = 1000
#
#     X, Y = make_classification(
#         n_samples=n_samples,
#         random_state=1
#     )
#     x_cont = torch.as_tensor(X, dtype=torch.float32)
#     x_cat = torch.zeros(n_samples, 0, dtype=torch.long)
#     print(f'{Y.shape=}')
#     y = torch.as_tensor(Y, dtype=torch.long)
#     tensors = dict(x_cont=x_cont, x_cat=x_cat, y=y[:, None])
#     tensor_infos = dict(x_cont=TensorInfo(feat_shape=[x_cont.shape[1]]), x_cat=TensorInfo(feat_shape=[0]),
#                         y=TensorInfo(cat_sizes=[2]))
#     ds = DictDataset(tensors, tensor_infos)
#
#     task_desc = TaskDescription('custom-class', 'ds_custom')
#     task_info = TaskInfo.from_ds(task_desc=task_desc, ds=ds)
#     task = Task(task_info=task_info, ds=ds)
#     task.save(paths)
#     TaskCollection.from_source('custom-class', paths).save(paths)
#
#
#     # ----- run benchmark -----
#     job_mgr = TabBenchJobManager(paths)
#     scheduler = SimpleJobScheduler(RayJobManager())
#     config_10_1_0 = RunConfig(n_tt_splits=2, n_cv=1, n_refit=0, save_y_pred=False)
#     task_infos = TaskCollection.from_name('custom-class', paths).load_infos(paths)
#
#     ds_x, ds_y = task_infos[0].load_task(paths).ds.split_xy()
#     # xgb = XGBInterfaceWrapper(**utils.join_dicts(DefaultParams.XGB_D, dict(n_estimators=2)))
#     xgb = XGB_TD_Classifier(n_estimators=2)
#     xgb.fit(ds_x.to_df(), ds_y.to_df())
#
#     job_mgr.add_jobs(task_infos, config_10_1_0,
#                      'XGB-D-class',
#                      XGBInterfaceWrapper(**utils.join_dicts(DefaultParams.XGB_D, dict(n_estimators=2))),
#                      tags=['default'], rerun=False)
#
#     job_mgr.run_jobs(scheduler)

