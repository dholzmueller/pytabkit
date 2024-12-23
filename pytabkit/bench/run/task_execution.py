import shutil
import traceback
from typing import List, Optional

import numpy as np

from pytabkit.bench.alg_wrappers.general import AlgWrapper
from pytabkit.bench.data.common import SplitType
from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskPackage, TaskInfo
from pytabkit.bench.run.results import save_summaries, ResultManager
from pytabkit.bench.scheduling.schedulers import BaseJobScheduler
from pytabkit.models import utils
from pytabkit.models.training.logging import StdoutLogger
import glob
import math

from pytabkit.bench.scheduling.jobs import AbstractJob
from pytabkit.bench.scheduling.resources import NodeResources
from pytabkit.models.alg_interfaces.base import RequiredResources


class TabBenchJob(AbstractJob):
    """
    Internal helper class implementing AbstractJob for running tabular benchmarking jobs with our scheduling code.
    """

    def __init__(self, alg_name: str, alg_wrapper: AlgWrapper, task_package: TaskPackage, paths: Paths):
        """
        :param alg_name: Unique name of the method (for saving results).
        :param alg_wrapper: Wrapper implementing the ML method.
        :param task_package: Task package containing information on dataset and splits.
        :param paths: Data path configuration.
        """
        self.alg_name = alg_name
        self.alg_wrapper = alg_wrapper
        self.task_package = task_package
        self.paths = paths

    def get_group(self) -> str:
        """
        :return: Group name, in this case just the name of the AlgWrapper class.
        """
        return self.alg_wrapper.__class__.__name__

    def __call__(self, assigned_resources: NodeResources) -> bool:
        """
        Run the experiment with the given resources.

        :param assigned_resources: Assigned resources.
        :return: False if the job completed more quickly because results were partially already saved.
        """
        task_desc = self.task_package.task_info.task_desc
        print(f'Running {self.alg_name} on {len(self.task_package.split_infos)} splits of dataset {task_desc} '
              f'with {assigned_resources.get_n_threads()} threads'
              , flush=True)
        logger = StdoutLogger()
        # check whether any data directories exist, i.e. whether data is already available
        dirs_exist = [utils.existsDir(self.paths.results_alg_task_split(task_desc, self.alg_name,
                                                                        self.task_package.n_cv, split_info.split_type,
                                                                        split_info.id))
                      for split_info in self.task_package.split_infos]
        # check whether the run is a normal run which does not have unusually short runtime due to pre-computed data
        finished_normally = self.task_package.rerun or not any(dirs_exist)
        # create tmp_folders for saving temporary data in case the run is interrupted and needs to be restarted
        tmp_folders = [self.paths.results_alg_task_split(task_desc,
                                                         alg_name=self.task_package.alg_name,
                                                         n_cv=self.task_package.n_cv,
                                                         split_type=split_info.split_type,
                                                         split_id=split_info.id) / 'tmp' for split_info in
                       self.task_package.split_infos]
        result_managers = self.alg_wrapper.run(self.task_package, logger, assigned_resources, tmp_folders)
        for rm, split_info in zip(result_managers, self.task_package.split_infos):
            rm.save(self.paths.results_alg_task_split(task_desc, self.alg_name, self.task_package.n_cv,
                                                      split_info.split_type, split_info.id))

        # delete tmp_folders to save disk space
        for tmp_folder in tmp_folders:
            if utils.existsDir(tmp_folder):
                shutil.rmtree(tmp_folder)

        print(f'Finished running {self.alg_name} on {len(self.task_package.split_infos)} splits of dataset {task_desc}',
              flush=True)
        return finished_normally

    def get_required_resources(self) -> RequiredResources:
        return self.alg_wrapper.get_required_resources(self.task_package)

    def get_desc(self) -> str:
        split_ids = [split_info.id for split_info in self.task_package.split_infos]
        split_str = f'splits {sorted(split_ids)}'
        if len(split_ids) == 1:
            split_str = f'split {split_ids[0]}'
        elif all([split_id == split_ids[0] + i for i, split_id in enumerate(split_ids)]):
            # we have a range
            split_str = f'splits {split_ids[0]}-{split_ids[-1]}'
        return self.alg_name + f' on {split_str} of task {self.task_package.task_info.task_desc}'


class RunConfig:
    """
    This class stores some benchmark settings that a method can be run with.
    """

    def __init__(self, n_tt_splits: int, n_cv: int = 1, n_refit: int = 0, use_default_split: bool = False,
                 trainval_fraction: float = 0.8,
                 save_y_pred: bool = False, min_split_idx: int = 0):
        """
        :param n_tt_splits: Number of trainval-test-splits to evaluate the method with.
        :param n_cv: Number of cross-validation folds. If n_cv=1, use a single random split.
        :param n_refit: Number of models that should be refitted (and ensembled) on the training and validation set.
        :param use_default_split: Whether the default split of the datasets should be used.
        :param trainval_fraction: Fraction in (0, 1) of the data that should be used for training and validation set.
        The rest will be used for the test set.
        :param save_y_pred: Whether the predictions on the whole dataset should be saved
        (can use a considerable amount of disk storage, e.g. 3 GB
        for running a single method on meta-train and meta-test benchmarks).
        :param min_split_idx: Minimum index of the split that should be used.
        Can be set larger than zero if only a sub-range of the splits should be run.
        """
        self.n_tt_splits = n_tt_splits
        self.n_cv = n_cv
        self.n_refit = n_refit
        self.use_default_split = use_default_split
        self.trainval_fraction = trainval_fraction
        self.save_y_pred = save_y_pred
        self.min_split_idx = min_split_idx


class TabBenchJobManager:
    """
    This class can be used to add and run jobs for tabular benchmarks.
    """

    def __init__(self, paths: Paths):
        """
        :param paths: Data path configuration.
        """
        self.paths = paths
        self.jobs = []
        self.save_args = []

    def add_jobs(self, task_infos: List[TaskInfo], run_config: RunConfig, alg_name: str, alg_wrapper: AlgWrapper,
                 tags: Optional[List[str]] = None, rerun: bool = False) -> None:
        """
        Add jobs for the given method with the given run configuration on all task infos
        where results are not already available (except if rerun=True).
        Will also store the algorithm configuration and copy the current source files
        to the corresponding algorithm folder.
        :param task_infos: List of TaskInfo objects representing the datasets on which the method should be run.
        :param run_config: Run configuration.
        :param alg_name: Name of the method, should be unique (is used for storing and printing the results)
        :param alg_wrapper: Wrapper implementing the ML method.
        :param tags: List of tags associated to the method (can be used for selecting a subset of methods later).
        :param rerun: If True, run all combinations even if there are already computed results stored for it.
        (For large reruns, we rather recommend renaming the old method with rename_alg.py
        and then running the jobs again with the new name and rerun=False.
        This avoids problems if the rerun crashes and preserves the old results for comparison.)
        """
        # todo: update after updating project structure
        if tags is None:
            tags = ['default']

        task_packages = []
        for task_info in task_infos:
            if run_config.use_default_split:
                tt_split_infos = task_info.get_default_splits(run_config.n_tt_splits)
            else:
                tt_split_infos = task_info.get_random_splits(run_config.n_tt_splits, run_config.trainval_fraction)
            tt_split_infos = tt_split_infos[run_config.min_split_idx:]

            if not rerun:
                # filter out splits where results have already been computed
                tt_split_infos = [split_info for split_info in tt_split_infos
                                  if not utils.existsFile(
                        self.paths.results_alg_task_split(task_info.task_desc, alg_name, run_config.n_cv,
                                                          split_info.split_type, split_info.id) / 'metrics.yaml')]

            n_tt_splits = len(tt_split_infos)
            if n_tt_splits == 0:
                continue

            max_n_vectorized = alg_wrapper.get_max_n_vectorized(task_info)
            n_splits_per_package = min(n_tt_splits,
                                       max(1, max_n_vectorized // max(run_config.n_cv, run_config.n_refit)))
            n_packages_per_task = math.ceil(n_tt_splits / n_splits_per_package)
            # distribute load more evenly across packages
            # (e.g. have split sizes (4, 4, 4) instead of (5, 5, 2) for n_tt_splits=12)
            n_splits_per_package = math.ceil(n_tt_splits / n_packages_per_task)

            batch_idxs = [n_splits_per_package * i for i in range((n_tt_splits - 1) // n_splits_per_package + 1)] \
                         + [n_tt_splits]

            for start, stop in zip(batch_idxs[:-1], batch_idxs[1:]):
                task_packages.append(TaskPackage(task_info, split_infos=tt_split_infos[start:stop],
                                                 n_cv=run_config.n_cv, n_refit=run_config.n_refit,
                                                 paths=self.paths, rerun=rerun, alg_name=alg_name,
                                                 save_y_pred=run_config.save_y_pred))

        for tp in task_packages:
            self.jobs.append(TabBenchJob(alg_name=alg_name, alg_wrapper=alg_wrapper, task_package=tp, paths=self.paths))

        if len(task_packages) > 0:
            # store alg info because something is actually being run
            # todo: this might not work on Windows
            # copy python files
            py_files = glob.glob('scripts/*.py') + glob.glob('pytabkit/**/*.py', recursive=True)
            utils.serialize(self.paths.algs() / alg_name / 'wrapper.pkl', alg_wrapper)
            extended_config = utils.join_dicts(alg_wrapper.config,
                                               {'alg_name': alg_name,
                                                'wrapper_class_name': alg_wrapper.__class__.__name__})
            utils.serialize(self.paths.algs() / alg_name / 'extended_config.yaml', extended_config, use_yaml=True)
            utils.serialize(self.paths.algs() / alg_name / 'tags.yaml', tags, use_yaml=True)
            for py_file in py_files:
                utils.copyFile(py_file, self.paths.algs() / alg_name / 'src' / py_file)

        rerun_summary = True  # always create the summary since a part of the results might have changed.
        self.save_args.append((self.paths, task_infos, alg_name, run_config.n_cv, rerun_summary))

    def run_jobs(self, scheduler: BaseJobScheduler) -> None:
        """
        Runs the added jobs with the given scheduler.
        After all jobs are done, creates the result summaries for faster loading of results.
        :param scheduler: Scheduler for running the jobs.
        """
        print(f'Starting scheduler')
        scheduler.add_jobs(self.jobs)
        scheduler.run()

        for args in self.save_args:
            try:
                save_summaries(*args)
            except Exception as e:
                traceback.print_exc()


def run_alg_selection(paths: Paths, config: RunConfig, task_infos: List[TaskInfo],
                      target_alg_name: str, alg_names: List[str], val_metric_name: str, tags: List[str] = ['paper'],
                      rerun: bool = False):
    n_cv = config.n_cv
    split_type = SplitType.DEFAULT if config.use_default_split else SplitType.RANDOM
    assert n_cv == 1  # not implemented otherwise
    assert len(alg_names) > 0
    assert config.n_refit == 0  # not implemented otherwise

    for task_info in task_infos:
        task_desc = task_info.task_desc
        for split_id in range(config.n_tt_splits):
            target_path = paths.results_alg_task_split(task_desc, target_alg_name, n_cv, split_type, split_id)
            if utils.existsFile(target_path / 'metrics.yaml') and not rerun:
                continue

            print(f'Running algorithm selection for {target_alg_name} on split {split_id} of task {task_desc}')
            best_alg_name = None
            best_val_score = np.inf
            best_alg_idx = None

            # find best alg
            for i, alg_name in enumerate(alg_names):
                rm = ResultManager.load(paths.results_alg_task_split(task_desc, alg_name, n_cv, split_type, split_id),
                                        only_metrics=True)
                val_score = rm.metrics_dict['cv']['val']['1']['0'][val_metric_name]

                if val_score < best_val_score or best_alg_name is None:
                    best_val_score = val_score
                    best_alg_name = alg_name
                    best_alg_idx = best_alg_idx

            # load full results of best alg and save them to target directory
            rm = ResultManager.load(paths.results_alg_task_split(task_desc, best_alg_name, n_cv, split_type, split_id))
            rm.other_dict['cv']['fit_params'] = dict(best_alg_idx=best_alg_idx,
                                                     sub_fit_params=rm.other_dict['cv']['fit_params'])
            rm.save(target_path)

    # save alg in algs folder
    py_files = glob.glob('scripts/*.py') + glob.glob('pytabkit/**/*.py', recursive=True)
    # utils.serialize(paths.algs() / target_alg_name / 'wrapper.pkl', alg_wrapper)
    extended_config = dict(sub_algs=alg_names)
    utils.serialize(paths.algs() / target_alg_name / 'extended_config.yaml', extended_config, use_yaml=True)
    utils.serialize(paths.algs() / target_alg_name / 'tags.yaml', tags, use_yaml=True)
    for py_file in py_files:
        utils.copyFile(py_file, paths.algs() / target_alg_name / 'src' / py_file)

    # save summaries
    print(f'Saving summaries')
    save_summaries(paths, task_infos, target_alg_name, n_cv=n_cv, rerun=True)
