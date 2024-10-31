import distutils.command.build_ext
from typing import List, Dict, Any, Tuple, Optional, Callable, Union

import numpy as np

from pytabkit.bench.data.common import SplitType
from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskCollection, TaskInfo
from pytabkit.models import utils
from pytabkit.models.training.metrics import Metrics


class AlgFilter:
    def __call__(self, alg_name: str, tags: List[str], alg_config: Dict[str, Any]) -> bool:
        raise NotImplementedError()


class FunctionAlgFilter(AlgFilter):
    def __init__(self, f):
        self.f = f

    def __call__(self, alg_name: str, tags: List[str], alg_config: Dict[str, Any]) -> bool:
        return self.f(alg_name, tags, alg_config)


class EvalModeSelector:  # base class
    def select_eval_modes(self, eval_modes: List[Tuple[str, str, str]]) -> List[Tuple[str, Tuple[str, str, str]]]:
        # gets a list of (cv_type, n_models, start_idx) tuples, returns a sublist of them
        # but with a suffix-str in for each element
        raise NotImplementedError()

    def select(self, alg_name: str, task_results: List) -> Tuple[List[str], List[List]]:
        # task results should be indexed by [task_idx][split_idx]['cv'/'refit'][str(n_models)][str(start_idx)]
        # returns a list of alg names and a list new_alg_task_results indexed by [task_idx][split_idx]

        # determine all combinations that occur in all task results
        sets = [set((cv_type, n_models, start_idx)
                    for cv_type, d1 in split_dict.items()
                    for n_models, d2 in d1.items()
                    for start_idx, d3 in d2.items())
                for task_result in task_results for split_dict in task_result]
        eval_modes = list(set.intersection(*sets))

        # select using function overridden in subclass
        selected = self.select_eval_modes(eval_modes)

        # select elements for selected eval modes
        new_alg_names = []
        new_alg_task_results = []
        for suffix, (cv_type, n_models, start_idx) in selected:
            new_alg_names.append(alg_name + suffix)
            new_alg_task_results.append([[split[cv_type][n_models][start_idx] for split in task_result]
                                         for task_result in task_results])
        return new_alg_names, new_alg_task_results


class DefaultEvalModeSelector(EvalModeSelector):
    def select_eval_modes(self, eval_modes: List[Tuple[str, str, str]]) -> List[Tuple[str, Tuple[str, str, str]]]:
        # out of different numbers of ensemble members,
        # select only the largest ensemble/bagging combinations and single ensemble member
        result = []
        # if ('refit', '1', '0') in eval_modes:
        #     # refit with 1 model, standard
        #     result.append(('', ('refit', '1', '0')))

        for name, val in [('bag', 'cv'), ('ens', 'refit')]:
            modes = [mode for mode in eval_modes if mode[0] == val]
            if len(modes) > 0:
                # maximize n_models
                idx = np.argmax([int(mode[1]) for mode in modes])
                idx_min = np.argmin([int(mode[1]) for mode in modes])
                mode = modes[idx]
                result.append((f' [{name}-{mode[1]}]', mode))
                if idx_min != idx:
                    result.append((f' [{name}-{modes[idx_min][1]}]', modes[idx_min]))

        return result


class AlgTaskTable:
    def __init__(self, alg_names: List[str], task_infos: List[TaskInfo], alg_task_results: List[List[Any]]):
        self.alg_names = alg_names
        self.task_infos = task_infos
        self.alg_task_results = alg_task_results

    def map(self, f):
        return AlgTaskTable(self.alg_names, self.task_infos,
                            [[[f(r) for r in splits] for splits in task_results]
                             for task_results in self.alg_task_results])

    def filter_n_splits(self, n_splits: int) -> 'AlgTaskTable':
        """
        Limits the number of split results to n_splits
        and removes all algs where there exists a task with less than n_splits split results.
        :param n_splits:
        :return:
        """
        alg_valid = [all(len(split_results) >= n_splits for split_results in task_results)
                     for task_results in self.alg_task_results]
        alg_names = [alg_name for is_valid, alg_name in zip(alg_valid, self.alg_names) if is_valid]
        alg_task_results = [[split_results[:n_splits] for split_results in task_results]
                            for is_valid, task_results in zip(alg_valid, self.alg_task_results) if is_valid]
        return AlgTaskTable(alg_names, self.task_infos, alg_task_results)

    def to_array(self) -> np.ndarray:
        return np.asarray(self.alg_task_results)

    def rename_algs(self, f: Callable[[str], str]) -> 'AlgTaskTable':
        return AlgTaskTable(alg_names=[f(an) for an in self.alg_names], task_infos=self.task_infos,
                            alg_task_results=self.alg_task_results)

    def filter_algs(self, alg_names: List[str]) -> 'AlgTaskTable':
        return AlgTaskTable(alg_names=[an for an in self.alg_names if an in alg_names], task_infos=self.task_infos,
                            alg_task_results=[tr for tr, an in zip(self.alg_task_results, self.alg_names)
                                              if an in alg_names])


class MultiResultsTable:
    def __init__(self, train_table: AlgTaskTable, val_table: AlgTaskTable, test_table: AlgTaskTable,
                 alg_tags: List[List[str]], alg_configs: List[Dict[str, Any]]):
        # val_table.alg_task_table and test_table.alg_task_table are indexed by
        # [alg_idx][task_idx][split_idx]['cv'/'refit'][str(n_models)][str(start_idx)][metric_name]
        self.train_table = train_table
        self.val_table = val_table
        self.test_table = test_table
        self.alg_tags = alg_tags
        self.alg_configs = alg_configs

    def get_test_results_table(self, eval_mode_selector: EvalModeSelector, val_metric_name: Optional[str] = None,
                               test_metric_name: Optional[str] = None,
                               alg_group_dict: Optional[Dict[str, AlgFilter]] = None,
                               val_test_groups: Optional[Dict[str, Dict[str, str]]] = None,
                               use_validation_errors: bool = False,
                               use_train_errors: bool = False) \
            -> AlgTaskTable:
        """
        :param eval_mode_selector:
            Decides how to select results from the different available ensembled/bagged results and how to name them
        :param val_metric_name: Name of the validation metric (used for optimizing over multiple algorithms)
        :param test_metric_name: Name of the test metric
        :param alg_group_dict: Optional dictionary of name: alg_filter.
            For each such pair, an additional algorithm with the given name will be added to the resulting table.
            Its results are computed as follows: On each split of each task,
            out of all the algorithms where the alg_filter returns True, the one with the best validation error is chosen,
            and then its test error is used.
        :param val_test_groups: Similar to alg_group_dict,
            but allows to use a different alg for the test score associated with the one with the best validation error.
            Specifically, for name: pairs in val_test_groups.items(),
            the best validation error among the keys of pairs will be determined,
            and then the test score of the value associated to this best key will be returned.
        :param use_validation_errors: If True, use validation errors instead of test errors.
        :param use_train_errors: If True, use train errors instead of test errors.
        :return:
        """
        # the selector assigns new alg names (e.g. with [ens-5] for an ensemble)
        # but the alg_group selects based on configs and new names
        assert not (use_train_errors and use_validation_errors)

        # extract only default metric values from self.val_table
        val_metric_name = val_metric_name or Metrics.default_eval_metric_name(self.val_table.task_infos[0].task_type)
        test_metric_name = test_metric_name or Metrics.default_eval_metric_name(self.val_table.task_infos[0].task_type)

        if '1-r2' in [val_metric_name, test_metric_name]:
            for table in [self.val_table, self.test_table, self.train_table]:
                table.alg_task_results = utils.map_nested(table.alg_task_results, lambda metrics_dict: utils.join_dicts(metrics_dict, {'1-r2': metrics_dict['nrmse']**2}), dim=6)

        # tables indexed by [alg_idx][task_idx][split_idx]['cv'/'refit'][str(n_models)][str(start_idx)][metric_name]
        val_results = utils.select_nested(self.val_table.alg_task_results, val_metric_name, dim=6)
        if use_validation_errors:
            test_results = val_results
        elif use_train_errors:
            test_results = utils.select_nested(self.train_table.alg_task_results, val_metric_name, dim=6)
        else:
            test_results = utils.select_nested(self.test_table.alg_task_results, test_metric_name, dim=6)

        # take mean over all single model validation scores in cross-validation
        # now indexed by [alg_idx][task_idx][split_idx]
        # print(np.asarray(val_results[0][0][0]['cv']['1'].values()))
        val_results = utils.map_nested(val_results, lambda dct: np.mean(np.asarray(list(dct['cv']['1'].values()))),
                                       dim=3)

        # create new test table by selecting for eval modes (multiple eval modes can be selected for an alg_name)
        # hence the table can get longer
        new_alg_names = []
        new_alg_task_results = []

        # Meaning: new_alg_names[new_alg_idxs[i]] is first algorithm corresponding to self.val_table.alg_names[i]
        new_alg_idxs = []

        for alg_name, task_results in zip(self.test_table.alg_names, test_results):
            # generates a list of alg names and of alg_task_results
            an, atr = eval_mode_selector.select(alg_name, task_results)
            if len(an) == 0:
                raise RuntimeError(f'No eval mode selected from alg {alg_name}')
            new_alg_idxs.append(len(new_alg_names))
            new_alg_names.extend(an)
            new_alg_task_results.extend(atr)

        # test_results_table.alg_task_results is indexed by [alg_idx][task_idx][split_idx]
        test_results_table = AlgTaskTable(new_alg_names, self.test_table.task_infos, new_alg_task_results)

        if val_test_groups is None:
            val_test_groups = dict()

        if alg_group_dict is not None:
            more_val_test_groups = {key: {alg_name: alg_name
                                          for alg_name, alg_tags, alg_config in
                                          zip(self.val_table.alg_names, self.alg_tags, self.alg_configs)
                                          if filter(alg_name, alg_tags, alg_config)}
                                    for key, filter in alg_group_dict.items()}

            val_test_groups = utils.join_dicts(val_test_groups, more_val_test_groups)

        # add algorithms optimized over a group, selecting the one with the best validation score
        # (or one associated to the best one)
        group_names = []
        group_task_results = []
        for group_name, val_test_dict in val_test_groups.items():
            if len(val_test_dict) == 0:
                continue  # could happen if the alg_filter does not apply to anything
            all_alg_names = self.val_table.alg_names
            val_alg_names = list(val_test_dict.keys())
            val_alg_idxs = [all_alg_names.index(alg_name) if alg_name in all_alg_names else None
                            for alg_name in val_alg_names]
            test_alg_idxs = [all_alg_names.index(val_test_dict[alg_name])
                             if val_test_dict[alg_name] in all_alg_names else None
                             for alg_name in val_alg_names]
            # print(f'{group_name=}, {val_alg_idxs=}, {test_alg_idxs=}')
            if None in (val_alg_idxs + test_alg_idxs):
                continue  # not all algs found

            max_n_splits = np.min([len(splits)
                                   for i in (val_alg_idxs + test_alg_idxs)
                                   for splits in val_results[i]])
            # shape: n_algs x n_tasks x max_n_splits
            cut_splits = [[splits[:max_n_splits] for splits in val_results[i]]
                          for i in val_alg_idxs]
            # shape: n_tasks x max_n_splits
            best_idxs = np.argmin(np.asarray(cut_splits), axis=0)
            test_atr = test_results_table.alg_task_results

            group_names.append(group_name)
            group_task_results.append(
                [[test_atr[new_alg_idxs[test_alg_idxs[best_idxs[task_idx, split_idx]]]][task_idx][split_idx]
                  for split_idx in range(best_idxs.shape[1])]
                 for task_idx in range(best_idxs.shape[0])])
        test_results_table = AlgTaskTable(test_results_table.alg_names + group_names, test_results_table.task_infos,
                                          test_results_table.alg_task_results + group_task_results)

        # # add alg groups - on each task, alg groups take the alg from the group with the best val error
        # # (val error is always minimized here, not maximized)
        # if alg_group_dict is not None:
        #     group_names = []
        #     group_task_results = []
        #     for key, alg_filter in alg_group_dict.items():
        #         alg_idxs = [i for i in range(len(self.val_table.alg_names))
        #                     if alg_filter(self.val_table.alg_names[i], self.alg_tags[i], self.alg_configs[i])]
        #         if len(alg_idxs) == 0:
        #             continue
        #         max_n_splits = np.min([len(splits)
        #                                for i in alg_idxs
        #                                for splits in val_results[i]])
        #         # shape: n_algs x n_tasks x max_n_splits
        #         cut_splits = [[splits[:max_n_splits] for splits in val_results[i]]
        #                       for i in alg_idxs]
        #         # shape: n_tasks x max_n_splits
        #         best_idxs = np.argmin(np.asarray(cut_splits), axis=0)
        #         test_atr = test_results_table.alg_task_results
        #
        #         group_names.append(key)
        #         group_task_results.append(
        #             [[test_atr[new_alg_idxs[alg_idxs[best_idxs[task_idx, split_idx]]]][task_idx][split_idx]
        #               for split_idx in range(best_idxs.shape[1])]
        #              for task_idx in range(best_idxs.shape[0])])
        #     test_results_table = AlgTaskTable(test_results_table.alg_names + group_names, test_results_table.task_infos,
        #                                       test_results_table.alg_task_results + group_task_results)

        return test_results_table

    @staticmethod
    def load(task_collection: TaskCollection, n_cv: int, paths: Paths, alg_filter: Optional[AlgFilter] = None,
             split_type=SplitType.RANDOM, max_n_splits: Optional[int] = None, max_n_algs: Optional[int] = None):
        # load only summaries (faster)
        alg_names = [alg_path.name for alg_path in paths.result_summaries().iterdir()]
        # now only keep algs where all tasks from task_collection have been evaluated
        alg_names = [an for an in alg_names if np.all([utils.existsDir(paths.summary_alg_task(task_desc, an, n_cv))
                                                       for task_desc in task_collection.task_descs])]

        print('computed alg names')

        alg_tags = [utils.deserialize(paths.algs() / alg_name / 'tags.yaml', use_yaml=True) for alg_name in alg_names]
        alg_configs = [utils.deserialize(paths.algs() / alg_name / 'extended_config.yaml', use_yaml=True)
                       for alg_name in alg_names]

        if alg_filter is None:
            alg_filter = lambda an, tags, aw: True

        alg_dict = {an: (tags, config) for an, tags, config in zip(alg_names, alg_tags, alg_configs)
                    if alg_filter(an, tags, config)}
        if max_n_algs is not None and max_n_algs >= 0:
            alg_dict = {key: value for i, (key, value) in enumerate(alg_dict.items()) if i < max_n_algs}
        alg_names = list(alg_dict.keys())
        alg_tags = [alg_dict[an][0] for an in alg_names]
        alg_configs = [alg_dict[an][1] for an in alg_names]

        task_infos = task_collection.load_infos(paths)

        # val_metric_name = Metrics.default_metric_name(task_infos[0].task_type)

        # indexed by
        # [alg_idx][task_idx]['cv'/'refit']['train'/'val'/'test'][str(n_models)][str(start_idx)][metric_name][split_idx]
        alg_task_results = [[utils.deserialize(paths.summary_alg_task(task_desc, alg_name, n_cv)
                                               / f'metrics.msgpack.gz', use_msgpack=True, compressed=True)[split_type]
                             for task_desc in task_collection.task_descs]
                            for alg_name in alg_names]

        # swap split_idx dimension to after task_idx, now indexed by
        # [alg_idx][task_idx][split_idx]['cv'/'refit']['train'/'val'/'test'][str(n_models)][str(start_idx)][metric_name]
        alg_task_results = utils.shift_dim_nested(alg_task_results, 7, 2)

        if max_n_splits is not None and max_n_splits >= 1:
            alg_task_results = utils.map_nested(alg_task_results,
                                                lambda lst: lst[:max_n_splits] if len(lst) > max_n_splits else lst, 2)

        def select_valtest(dct: Dict, name: str):
            # helper function because for the 'refit' results,
            # we have to take the validation results from the 'cv' part
            # because 'refit' did not have a validation set
            if name != 'val':
                return {key: value[name] for key, value in dct.items()}
            else:
                return {key: dct['cv']['val'] for key in dct}

        tables = {name: AlgTaskTable(alg_names=alg_names, task_infos=task_infos,
                               alg_task_results=utils.map_nested(alg_task_results,
                                                                 lambda dct: select_valtest(dct, name), dim=3))
                  for name in ['train', 'val', 'test']}
        # does not work since 'refit' does not have 'val'
        # tables = [AlgTaskTable(alg_names=alg_names, task_infos=task_infos,
        #                        alg_task_results=utils.select_nested(alg_task_results, name, dim=4))
        #           for name in ['val', 'test']]
        return MultiResultsTable(train_table=tables['train'], val_table=tables['val'], test_table=tables['test'],
                                 alg_tags=alg_tags, alg_configs=alg_configs)


class TableAnalyzer:
    def __init__(self, post_f: Optional[Callable[[float], float]] = None):
        self.post_f = post_f or (lambda x: x)

    def _print_table(self, alg_names: List[str], means, stds=None, is_higher_better: bool = False,
                     perm: Optional[np.ndarray] = None):
        means = np.asarray(means)
        if perm is None:
            perm = np.argsort(means)
            if is_higher_better:
                perm = perm[::-1]
        means = means[perm]
        alg_names = [alg_names[i] for i in perm]
        if stds is None:
            str_table = [[an + ': ', f'{self.post_f(m):6.4f}'] for an, m in zip(alg_names, means)]
        else:
            stds = np.asarray(stds)[perm]
            str_table = [[an + ': ', f'{self.post_f(m):6.4f} ',
                          f'[{self.post_f(m - 2 * s):6.4f}, {self.post_f(m + 2 * s):6.4f}]']
                         for an, m, s in zip(alg_names, means, stds)]

        print(utils.pretty_table_str(str_table))

    def print_analysis(self, alg_task_table: AlgTaskTable):
        raise NotImplementedError()


class TaskWeighting:
    def __init__(self, task_infos: List[TaskInfo], separate_task_names: Optional[List[str]]):
        """
        Computes a weighting of tasks, downweighting tasks that have similar tasks.
        :param task_infos: Task infos.
        :param separate_task_names: Names of tasks that should not be grouped together with other tasks
        """
        self.task_infos = task_infos
        separate_task_names = separate_task_names or []
        task_names = [task_info.task_desc.task_name.split('_')[0] for task_info in task_infos]
        task_prefixes = [task_name if task_name in separate_task_names else task_name.split('_')[0]
                         for task_name in task_names]
        self.prefix_counts = {}
        for prefix in task_prefixes:
            if prefix in self.prefix_counts:
                self.prefix_counts[prefix] += 1
            else:
                self.prefix_counts[prefix] = 1
        self.task_weights = np.asarray([1.0 / self.prefix_counts[prefix] for prefix in task_prefixes])
        self.task_weights /= np.sum(self.task_weights)

    def get_n_groups(self) -> int:
        return len(self.prefix_counts)

    def get_task_weights(self) -> np.ndarray:
        return self.task_weights


class MeanTableAnalyzer(TableAnalyzer):
    def __init__(self, f=None, use_weighting=False, separate_task_names: Optional[List[str]] = None, post_f=None):
        super().__init__(post_f=post_f)
        self.f = f
        self.use_weighting = use_weighting
        self.separate_task_names = separate_task_names

    def print_analysis(self, alg_task_table: AlgTaskTable) -> None:
        if self.use_weighting:
            task_weights = TaskWeighting(alg_task_table.task_infos, self.separate_task_names).get_task_weights()
            # task_weights = get_task_weights(alg_task_table.task_infos)
        else:
            n = len(alg_task_table.task_infos)
            task_weights = np.ones(n) / n
        if self.f is not None:
            alg_task_table = alg_task_table.map(self.f)
        alg_task_results = alg_task_table.alg_task_results
        # if self.f is not None:
        #     alg_task_results = [[[self.f(x) for x in c] for c in b] for b in alg_task_results]

        means = [np.dot(task_weights, [np.mean(splits) for splits in task_results])
                 for task_results in alg_task_results]
        stds = [np.sqrt(np.dot(task_weights ** 2, [np.std(splits) ** 2 / len(splits) for splits in task_results]))
                for task_results in alg_task_results]

        self._print_table(alg_task_table.alg_names, means, stds)

    def get_means(self, alg_task_table: AlgTaskTable) -> List[float]:
        if self.use_weighting:
            separate_task_names = ['facebook_comment_volume', 'facebook_live_sellers_thailand_shares']
            task_weights = TaskWeighting(alg_task_table.task_infos, separate_task_names).get_task_weights()
        else:
            n = len(alg_task_table.task_infos)
            task_weights = np.ones(n) / n
        if self.f is not None:
            alg_task_table = alg_task_table.map(self.f)
        alg_task_results = alg_task_table.alg_task_results
        return [self.post_f(np.dot(task_weights, [np.mean(splits) for splits in task_results]))
                for task_results in alg_task_results]

    def get_intervals(self, alg_task_table: AlgTaskTable, std_factor: float = 2.0) -> List[Tuple[float, float]]:
        # e.g. if std_factor=2, then the +-2 sigma interval will be used
        if self.use_weighting:
            separate_task_names = ['facebook_comment_volume', 'facebook_live_sellers_thailand_shares']
            task_weights = TaskWeighting(alg_task_table.task_infos, separate_task_names).get_task_weights()
        else:
            n = len(alg_task_table.task_infos)
            task_weights = np.ones(n) / n
        if self.f is not None:
            alg_task_table = alg_task_table.map(self.f)
        alg_task_results = alg_task_table.alg_task_results
        means = [np.dot(task_weights, [np.mean(splits) for splits in task_results])
                 for task_results in alg_task_results]
        stds = [np.sqrt(np.dot(task_weights ** 2, [np.std(splits) ** 2 / len(splits) for splits in task_results]))
                for task_results in alg_task_results]
        post_intervals = [(self.post_f(mean - std_factor * std), self.post_f(mean + std_factor * std))
                          for mean, std in zip(means, stds)]
        return post_intervals


class ArrayTableAnalyzer(TableAnalyzer):
    """
    Intermediate class that analyzes using the same number of splits for each method
    """

    def __init__(self, f=None, use_weighting=False, separate_task_names: Optional[List[str]] = None, post_f=None):
        super().__init__(post_f=post_f)
        self.f = f
        self.use_weighting = use_weighting
        self.separate_task_names = separate_task_names

    def _is_higher_better(self) -> bool:
        # can be overridden if necessary
        return False

    def _process_losses(self, loss_arr: np.ndarray, val_loss_arr: Optional[np.ndarray]) \
            -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        # optional second tuple can be the permutation of configurations that should be used for displaying them
        raise NotImplementedError()

    def print_analysis(self, alg_task_table: AlgTaskTable, val_table: Optional[AlgTaskTable] = None) -> None:
        if self.use_weighting:
            task_weights = TaskWeighting(alg_task_table.task_infos, self.separate_task_names).get_task_weights()
            # task_weights = get_task_weights(alg_task_table.task_infos)
        else:
            n = len(alg_task_table.task_infos)
            task_weights = np.ones(n) / n
        if self.f is not None:
            alg_task_table = alg_task_table.map(self.f)
            if val_table is not None:
                val_table = val_table.map(self.f)
        alg_task_results = alg_task_table.alg_task_results
        # if self.f is not None:
        #     alg_task_results = [[[self.f(x) for x in c] for c in b] for b in alg_task_results]

        min_n_splits = np.min([len(splits) for task_results in alg_task_results for splits in task_results])

        loss_arr = np.asarray([[splits[:min_n_splits] for splits in task_results] for task_results in alg_task_results])
        val_loss_arr = None
        if val_table is not None:
            val_loss_arr = np.asarray(
                [[splits[:min_n_splits] for splits in task_results] for task_results in val_table.alg_task_results])
        results_arr = self._process_losses(loss_arr, val_loss_arr)
        perm = None
        if isinstance(results_arr, Tuple):
            results_arr, perm = results_arr

        means = np.mean(results_arr, axis=-1) @ task_weights

        # todo: could implement better confidence intervals from plotting code
        stds = np.sqrt((np.std(results_arr, axis=-1) ** 2 / results_arr.shape[-1]) @ (task_weights ** 2))

        self._print_table(alg_task_table.alg_names, means, stds, is_higher_better=self._is_higher_better(), perm=perm)


class WinsTableAnalyzer(ArrayTableAnalyzer):
    def _process_losses(self, loss_arr: np.ndarray, val_loss_arr: Optional[np.ndarray]) \
            -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        return (loss_arr == np.min(loss_arr, axis=0, keepdims=True)).astype(np.float32)

    def _is_higher_better(self) -> bool:
        return True


def get_ranks(values: np.ndarray) -> np.ndarray:
    # computes ranks across the first axis
    return np.sum(values[:, None] > values[None, :], axis=1) + 1
    # ranks_per_method = []
    # for i in range(values.shape[0]):
    #     ranks_per_method.append(np.sum((values[i, None] > values).astype(np.int32), axis=0) + 1)
    # return np.stack(ranks_per_method, axis=0)


class RankTableAnalyzer(ArrayTableAnalyzer):
    def _process_losses(self, loss_arr: np.ndarray, val_loss_arr: Optional[np.ndarray]) \
            -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        return get_ranks(loss_arr)


class NormalizedLossTableAnalyzer(ArrayTableAnalyzer):
    def _process_losses(self, loss_arr: np.ndarray, val_loss_arr: Optional[np.ndarray]) \
            -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        min_arr = np.min(loss_arr, axis=0, keepdims=True)
        max_arr = np.max(loss_arr, axis=0, keepdims=True)
        return (loss_arr - min_arr) / (max_arr - min_arr + 1e-30)


class GreedyAlgSelectionTableAnalyzer(ArrayTableAnalyzer):
    """
    Greedy selection of a portfolio of methods
    such that the addition improves the best performance in the portfolio the most
    """
    def _process_losses(self, loss_arr: np.ndarray, val_loss_arr: Optional[np.ndarray]) \
            -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        # val_loss_arr = loss_arr  # todo
        assert val_loss_arr is not None
        n_algs = loss_arr.shape[0]
        non_selected_algs = np.arange(n_algs)
        # alg_selected = np.zeros(loss_arr.shape[0], dtype=np.bool_)

        perm = []

        for i in range(loss_arr.shape[0]):
            # losses are updated, tracking the loss of the alg after optimizing over best models and the given one
            # find best model
            best_non_selected_idx = np.argmin(np.mean(val_loss_arr, axis=(1, 2))[non_selected_algs])
            best_idx = non_selected_algs[best_non_selected_idx]

            perm.append(best_idx)
            non_selected_algs = np.concatenate(
                [non_selected_algs[:best_non_selected_idx], non_selected_algs[best_non_selected_idx + 1:]], axis=0)

            for alg_idx in non_selected_algs:
                is_better = val_loss_arr[best_idx] <= val_loss_arr[alg_idx]
                val_loss_arr[alg_idx] = np.where(is_better, val_loss_arr[best_idx], val_loss_arr[alg_idx])
                loss_arr[alg_idx] = np.where(is_better, loss_arr[best_idx], loss_arr[alg_idx])

        return loss_arr, np.asarray(perm, dtype=np.int32)


def alg_results_str(alg_task_table: AlgTaskTable, alg_name: str):
    alg_task_results = alg_task_table.alg_task_results
    if alg_name not in alg_task_table.alg_names:
        alg_name = alg_name + ' [bag-1]'
    # todo: could throw an exception
    alg_idx = alg_task_table.alg_names.index(alg_name)
    task_results = alg_task_results[alg_idx]
    means = [np.mean(splits) for splits in task_results]
    stds = [np.std(splits) / np.sqrt(len(splits)) for splits in task_results]
    task_names = [str(task_info.task_desc) for task_info in alg_task_table.task_infos]
    str_table = [[f'Task ', 'Error', 'Interval']]
    for name, mean, std in zip(task_names, means, stds):
        str_table.append([f'{name}: ', f'{mean:6.4f} ', f'[{mean - 2 * std:6.4f}, {mean + 2 * std:6.4f}]'])
    return utils.pretty_table_str(str_table)


def alg_comparison_str(alg_task_table: AlgTaskTable, alg_names: List[str]):
    alg_task_results = alg_task_table.alg_task_results
    alg_names = [an if an in alg_task_table.alg_names else an + ' [bag-1]' for an in alg_names]
    # todo: could throw an exception
    alg_idxs = [alg_task_table.alg_names.index(alg_name) for alg_name in alg_names]
    means = [[np.mean(splits) for splits in alg_task_results[alg_idx]] for alg_idx in alg_idxs]
    task_names = [str(task_info.task_desc) for task_info in alg_task_table.task_infos]
    str_table = [[f'Task '] + [f'Alg {i + 1} ' for i in range(len(alg_names))]]
    for i, name in enumerate(task_names):
        str_table.append([f'{name}: '] + [f'{alg_means[i]:6.4f} ' for alg_means in means])
    str_table.append([''] * 3)
    min_means = [np.min([alg_means[i] for alg_means in means]) for i in range(len(task_names))]
    n_wins_list = [sum([int(alg_means[i] == min_means[i]) for i in range(len(task_names))]) for alg_means in means]
    str_table.append(['Wins:'] + [str(n_wins) for n_wins in n_wins_list])
    return utils.pretty_table_str(str_table)

# CLI:
# task collection
# n_cv (default=1?)
# preference regarding is_cv and ensembling?
# optionally whether default splits should be used or not?
# tags (connect by and or or?)
