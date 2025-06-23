import copy
import time
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import sklearn
import torch

from pytabkit.bench.alg_wrappers.interface_wrappers import RandomParamsNNInterfaceWrapper, \
    RandomParamsXGBInterfaceWrapper, LoadResultsWrapper, NNInterfaceWrapper, XGBInterfaceWrapper
from pytabkit.bench.data.common import SplitType
from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskCollection
from pytabkit.bench.run.results import ResultManager
from pytabkit.bench.run.task_execution import TabBenchJobManager, RunConfig, run_alg_selection
from pytabkit.bench.scheduling.execution import RayJobManager
from pytabkit.bench.scheduling.schedulers import SimpleJobScheduler
from pytabkit.models import utils
from pytabkit.models.data.data import TaskType
from pytabkit.models.data.splits import SplitInfo
from pytabkit.models.sklearn.default_params import DefaultParams
from pytabkit.models.training.metrics import Metrics


class ProbclassExperiments:
    def __init__(self, paths: Paths, n_tt_splits: int, n_cv: int, n_hpo_steps: int, hpo_models: List[str],
                 default_models: List[str]):
        self.paths = paths
        self.n_tt_splits = n_tt_splits
        self.n_cv = n_cv
        self.n_hpo_steps = n_hpo_steps
        self.hpo_models = hpo_models
        self.default_models = default_models

        self.job_mgr = None
        self.scheduler = None
        self.config = None
        self.task_infos = None
        self.val_metric_names = None
        self.calib_options = None
        self.hpo_names = None

    def setup(self):
        # don't do this in the constructor so we have a new job_mgr etc. every time  (to be safe)
        self.job_mgr = TabBenchJobManager(paths)
        self.scheduler = SimpleJobScheduler(RayJobManager())
        metrics = Metrics(metric_names=[
            'cross_entropy', 'brier',
            'n_cross_entropy', 'n_brier',
            'logloss-clip1e-06',
            'smece', 'ece-15', 'rmsce-15', 'mce-15',
            'class_error', '1-mcc', '1-auroc-ovr',
            'ref-ll-ts', 'ref-br-ts', 'cal-ll-ts', 'cal-br-ts',
        ],
            val_metric_name='logloss',  # probably unused anyway
            task_type=TaskType.CLASSIFICATION)
        self.config = RunConfig(n_tt_splits=self.n_tt_splits, n_cv=self.n_cv, n_refit=0, save_y_pred=True,
                                metrics=metrics, train_fraction=0.8)
        self.task_infos = TaskCollection.from_name('talent-class-small', paths).load_infos(paths)
        self.val_metric_names = ['cross_entropy', 'brier', 'class_error', '1-auroc-ovr', 'ref-ll-ts', 'ref-br-ts']
        self.hpo_names = copy.copy(self.hpo_models)
        if n_cv != 1:
            self.hpo_names = [bn + f'-cv{n_cv}' for bn in self.hpo_names]
        self.calib_options = {'ts-mix': dict(calibration_method='temp-scaling', calibrate_with_mixture=True)}

    def run_hpo_configs(self, n_hpo_steps: Optional[int] = None, rerun: bool = False):
        # for RealMLP:
        # 10 steps with 2 splits and n_cv=1: 2h5m
        # 2 steps with 2 splits and n_cv=5: 48m
        # 10 steps with 2 splits and n_cv=5: 4h6m
        # -> run 50 steps with 10 splits and n_cv=1: a bit more than 50h
        # for XGB:
        # 10 steps with 2 splits and n_cv=5: 10h22m (but waiting long for results on volkert, otherwise more like 6h30m)
        # 20 steps with 2 splits and n_cv=1: 3h
        # -> run 50 steps with 10 splits and n_cv=1: 37h
        self.setup()

        if n_hpo_steps is None:
            n_hpo_steps = self.n_hpo_steps

        # tag = f'paper_hpo-cv{n_cv}' if n_cv != 1 else 'paper_hpo'
        tag = 'paper_hpo'
        cv_str = f'-cv{n_cv}' if n_cv != 1 else ''
        for step_idx in range(n_hpo_steps):
            for base_name in self.hpo_names:
                if base_name.startswith('RealMLP-HPO'):
                    self.job_mgr.add_jobs(self.task_infos, self.config,
                                          f'RealMLP-HPO{cv_str}_step-{step_idx}',
                                          RandomParamsNNInterfaceWrapper(model_idx=step_idx, hpo_space_name='probclass',
                                                                         val_metric_names=self.val_metric_names),
                                          tags=[tag + '_' + base_name], rerun=rerun)
                elif base_name.startswith('XGB-HPO'):
                    self.job_mgr.add_jobs(self.task_infos, self.config,
                                          f'XGB-HPO{cv_str}_step-{step_idx}',
                                          RandomParamsXGBInterfaceWrapper(model_idx=step_idx,
                                                                          hpo_space_name='probclass',
                                                                          n_estimators=1000,
                                                                          early_stopping_rounds=1000,
                                                                          val_metric_names=self.val_metric_names),
                                          tags=[tag + '_' + base_name], rerun=rerun)
                elif base_name.startswith('MLP-HPO'):
                    self.job_mgr.add_jobs(self.task_infos, self.config,
                                          f'MLP-HPO{cv_str}_step-{step_idx}',
                                          RandomParamsNNInterfaceWrapper(model_idx=step_idx,
                                                                         hpo_space_name='probclass-mlp',
                                                                         val_metric_names=self.val_metric_names),
                                          tags=[tag + '_' + base_name], rerun=rerun)

        self.job_mgr.run_jobs(self.scheduler)

    def run_hpo_alg_selection(self, rerun: bool = False):
        tag = 'paper'
        self.setup()

        for base_name in self.hpo_names:
            for val_metric_name in self.val_metric_names:
                alg_names = [f'{base_name}_step-{i}_val-{val_metric_name}' for i in range(self.n_hpo_steps)]
                run_alg_selection(paths, self.config, self.task_infos,
                                  f'{base_name}-{self.n_hpo_steps}_val-{val_metric_name}',
                                  alg_names, val_metric_name, tags=[tag + '_' + base_name], rerun=rerun)

    def run_hpo_calibration_configs(self, rerun: bool = False):
        tag = 'paper'
        self.setup()

        for base_name in self.hpo_names:
            for calib_name, calib_params in self.calib_options.items():
                for val_metric_name in self.val_metric_names:
                    alg_name = f'{base_name}-{self.n_hpo_steps}_val-{val_metric_name}'
                    self.job_mgr.add_jobs(self.task_infos, self.config,
                                          f'{alg_name}_{calib_name}',
                                          LoadResultsWrapper(alg_name=alg_name, **calib_params),
                                          tags=[tag + '_' + base_name], rerun=rerun)

        self.job_mgr.run_jobs(self.scheduler)

    def run_step_calibration_configs(self, rerun: bool = False):
        # took 1h10m for 20 steps and 2 tt splits of RealMLP-HPO
        tag = 'paper_hpo-calib'
        self.setup()

        for calib_name, calib_params in self.calib_options.items():
            for val_metric_name in self.val_metric_names:
                for step_idx in range(self.n_hpo_steps):
                    for base_name in self.hpo_names:
                        alg_name = f'{base_name}_step-{step_idx}_val-{val_metric_name}'
                        self.job_mgr.add_jobs(self.task_infos, self.config,
                                              f'{alg_name}_{calib_name}',
                                              LoadResultsWrapper(alg_name=alg_name, **calib_params),
                                              tags=[tag + '_' + base_name], rerun=rerun)

        self.job_mgr.run_jobs(self.scheduler)

    def run_default_configs(self, rerun: bool = False):
        tag = 'paper'
        cv_str = f'-cv{n_cv}' if n_cv != 1 else ''
        self.setup()
        val_metric_names = self.val_metric_names + ['ref-ll-ts-cv5', 'ref-ll-is']

        for base_name in self.default_models:
            if base_name.startswith('RealMLP-TD'):
                self.job_mgr.add_jobs(self.task_infos, self.config,
                                      f'RealMLP-TD{cv_str}',
                                      NNInterfaceWrapper(**utils.join_dicts(DefaultParams.RealMLP_TD_CLASS, dict(
                                          use_ls=False, val_metric_names=val_metric_names,
                                      ))),
                                      tags=[tag + '_' + base_name], rerun=rerun)
            elif base_name.startswith('XGB-D'):
                self.job_mgr.add_jobs(self.task_infos, self.config,
                                      f'XGB-D{cv_str}',
                                      XGBInterfaceWrapper(**DefaultParams.XGB_D,
                                                          val_metric_names=val_metric_names),
                                      tags=[tag + '_' + base_name], rerun=rerun)
            elif base_name.startswith('MLP-D'):
                self.job_mgr.add_jobs(self.task_infos, self.config,
                                      f'MLP-D{cv_str}',
                                      NNInterfaceWrapper(**DefaultParams.VANILLA_MLP_CLASS,
                                                         val_metric_names=val_metric_names),
                                      tags=[tag + '_' + base_name], rerun=rerun)

        self.job_mgr.run_jobs(self.scheduler)

    def run_default_calibration_configs(self, rerun: bool = False):
        tag = 'paper'
        self.setup()

        val_metric_names = self.val_metric_names + ['ref-ll-ts-cv5', 'ref-ll-is']

        for base_name in self.default_models:
            for calib_name, calib_params in self.calib_options.items():
                for val_metric_name in val_metric_names:
                    alg_name = f'{base_name}_val-{val_metric_name}'
                    self.job_mgr.add_jobs(self.task_infos, self.config,
                                          f'{alg_name}_{calib_name}',
                                          LoadResultsWrapper(alg_name=alg_name, **calib_params),
                                          tags=[tag + '_' + base_name], rerun=rerun)

        self.job_mgr.run_jobs(self.scheduler)

    @staticmethod
    def get_extended_calib_methods() -> Dict[str, Dict[str, Any]]:
        return {
            'ts': dict(calibration_method='temp-scaling'),
            'ts-mix': dict(calibration_method='temp-scaling', calibrate_with_mixture=True),
            'ag-ts': dict(calibration_method='autogluon-ts'),
            'ag-ts-mix': dict(calibration_method='autogluon-ts', calibrate_with_mixture=True),
            'ag-inv-ts': dict(calibration_method='autogluon-inv-ts'),
            'ag-inv-ts-mix': dict(calibration_method='autogluon-inv-ts', calibrate_with_mixture=True),
            'torchunc-ts': dict(calibration_method='torchunc-ts'),
            'torchunc-ts-mix': dict(calibration_method='torchunc-ts', calibrate_with_mixture=True),
            'torchcal-ts': dict(calibration_method='torchcal-ts'),
            'torchcal-ts-mix': dict(calibration_method='torchcal-ts', calibrate_with_mixture=True),
            'guo-ts': dict(calibration_method='guo-ts'),
            'guo-ts-mix': dict(calibration_method='guo-ts', calibrate_with_mixture=True),
            'ir': dict(calibration_method='isotonic'),
            'ir-mix': dict(calibration_method='isotonic', calibrate_with_mixture=True),
        }

    def run_calibration_benchmark(self, rerun: bool = False):
        tag = 'paper_calib-bench'
        self.setup()

        alg_name = f'XGB-D_val-class_error'

        calib_methods = self.get_extended_calib_methods()

        for calib_name, calib_params in calib_methods.items():
            self.job_mgr.add_jobs(self.task_infos, self.config,
                                  f'{alg_name}_calib-bench_{calib_name}',
                                  LoadResultsWrapper(alg_name=alg_name, **calib_params),
                                  tags=[tag], rerun=rerun)

        self.job_mgr.run_jobs(self.scheduler)

    def run_calibration_timing(self, rerun: bool = False):
        import probmetrics.calibrators
        from probmetrics.distributions import CategoricalLogits

        self.setup()

        results_list = []

        csv_path = paths.base() / 'calib_times' / 'times.csv'
        if utils.existsFile(csv_path) and not rerun:
            return

        alg_name = f'XGB-D_val-class_error'
        calib_methods = self.get_extended_calib_methods()

        for i, task_info in enumerate(self.task_infos):
            print(f'Running calibration timing on {task_info.task_desc} ({i+1}/{len(self.task_infos)})')
            ds = task_info.load_task(self.paths).ds
            y_full = ds.tensors['y'].squeeze(-1)
            random_splits = task_info.get_random_splits(self.n_tt_splits, train_fraction=self.config.train_fraction,
                                                        trainval_fraction=self.config.trainval_fraction)
            for split_idx in range(self.n_tt_splits):
                random_split: SplitInfo = random_splits[split_idx]
                trainval_split = random_split.splitter.split_ds(ds)
                trainval_idxs = trainval_split.get_sub_idxs(0)
                trainval_ds = trainval_split.get_sub_ds(0)
                sub_splits = random_split.get_sub_splits(trainval_ds, n_splits=self.n_cv, is_cv=True)

                path = self.paths.results_alg_task_split(task_info.task_desc, alg_name, n_cv=self.n_cv,
                                                         split_type=SplitType.RANDOM, split_id=split_idx)
                rm = ResultManager.load(path, load_other=False, load_preds=True)
                y_logits_torch = torch.as_tensor(rm.y_preds_cv, dtype=torch.float32)

                for cv_idx in range(self.n_cv):
                    sub_split = sub_splits[cv_idx]
                    val_idxs = trainval_idxs[sub_split.get_sub_idxs(0)]
                    y_val = y_full[val_idxs]
                    y_pred_val = CategoricalLogits(y_logits_torch[cv_idx, val_idxs])

                    for calib_name, calib_params in calib_methods.items():
                        cal = probmetrics.calibrators.get_calibrator(**calib_params)
                        if i == 0 and split_idx == 0 and cv_idx == 0:
                            # dry run to avoid measuring import times
                            cal_tmp = sklearn.base.clone(cal)
                            cal_tmp.fit_torch(y_pred_val, y_val)

                        start_time = time.time()
                        cal.fit_torch(y_pred_val, y_val)
                        end_time = time.time()
                        results_list.append(dict(
                            alg_name=alg_name,
                            calib_name=calib_name,
                            task=str(task_info.task_desc),
                            n_val=len(val_idxs),
                            tt_split_idx=split_idx,
                            cv_split_idx=cv_idx,
                            time=end_time - start_time))

        results_df = pd.DataFrame(results_list)
        utils.ensureDir(csv_path)
        results_df.to_csv(csv_path)


if __name__ == '__main__':
    n_hpo_steps = 30
    n_tt_splits = 5
    n_cv = 1
    paths = Paths.from_env_variables()
    exp = ProbclassExperiments(paths=paths, n_tt_splits=n_tt_splits, n_cv=n_cv, n_hpo_steps=n_hpo_steps,
                               hpo_models=['MLP-HPO', 'XGB-HPO', 'RealMLP-HPO'],
                               default_models=['MLP-D', 'XGB-D', 'RealMLP-TD'])

    exp.run_default_configs()
    exp.run_default_calibration_configs()

    # took 9h for 20 steps with 5 splits for MLP-HPO
    # for RealMLP + XGB-HPO: 9h45m + 1h34m + ...
    # 30 hpo steps with 5 splits for MLP + RealMLP + XGB: 9h + 9h45m + 1h34m + 17h52m = 20h19m + 17h52m = 38h11m
    exp.run_hpo_configs()

    exp.run_hpo_alg_selection()
    exp.run_hpo_calibration_configs()
    exp.run_calibration_timing()
    exp.run_calibration_benchmark()

    # not used in the paper
    # exp.run_step_calibration_configs()
