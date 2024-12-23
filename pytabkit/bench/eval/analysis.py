from typing import Optional, Callable, Tuple, Dict, List, Union

import numpy as np
import scipy

from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskCollection
from pytabkit.bench.eval.evaluation import FunctionAlgFilter, MultiResultsTable, DefaultEvalModeSelector, TaskWeighting, \
    get_ranks
from pytabkit.models import utils
from pytabkit.models.data.nested_dict import NestedDict


class ResultsTables:
    def __init__(self, paths: Paths):
        self.paths = paths
        self.tables = NestedDict()

    def get(self, coll_name: str, n_cv: int = 1, tag: str = 'paper') -> MultiResultsTable:
        idxs = (coll_name, n_cv, tag)
        if idxs in self.tables:
            return self.tables[idxs]
        else:
            # load table from disk
            task_collection = TaskCollection.from_name(coll_name, self.paths)
            alg_filter = FunctionAlgFilter(lambda an, tags, config, my_tag=tag: my_tag in tags)
            table = MultiResultsTable.load(task_collection, n_cv=n_cv, paths=self.paths, alg_filter=alg_filter)
            self.tables[idxs] = table
            return table


def _get_t_mean_confidence_interval_single(values: np.ndarray) -> Tuple[float, float]:
    # following https://www.geeksforgeeks.org/how-to-calculate-confidence-intervals-in-python/
    # see also https://stats.stackexchange.com/questions/358408/confidence-interval-for-the-mean-normal-distribution-or-students-t-distributi
    # and http://stla.github.io/stlapblog/posts/ModelReduction.html
    sem = scipy.stats.sem(values)
    if sem == 0.0:
        mean = np.mean(values)
        return mean, mean
    else:
        interval = scipy.stats.t.interval(confidence=0.95, df=len(values) - 1, loc=np.mean(values),
                                          scale=sem)
        return interval[0], interval[1]


def get_t_mean_confidence_interval(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # takes the confidence intervals across the last dimension,
    # the other dimensions are considered to be batch dimensions
    if len(values.shape) == 1:
        lower, upper = _get_t_mean_confidence_interval_single(values)
        return np.asarray(lower), np.asarray(upper)

    pairs = [get_t_mean_confidence_interval(values[i]) for i in range(values.shape[0])]
    lower = np.asarray([pair[0] for pair in pairs])
    upper = np.asarray([pair[1] for pair in pairs])
    return lower, upper


def get_benchmark_results(paths: Paths, table: MultiResultsTable, coll_name: str,
                          use_relative_score: bool = True, return_percentages: bool = True,
                          val_metric_name: Optional[str] = None, test_metric_name: Optional[str] = None,
                          rel_alg_name: str = 'BestModel', use_ranks: bool = False,
                          use_normalized_errors: bool = False,
                          use_grinnorm_errors: bool = False,
                          use_task_mean: bool = True,
                          use_geometric_mean: bool = True, shift_eps: float = 1e-2,
                          filter_alg_names_list: Optional[List[str]] = None,
                          simplify_name_fn: Optional[Callable[[str], str]] = None,
                          n_splits: int = 10, use_validation_errors: bool = False) -> \
        Tuple[
            Dict[str, Union[float, np.ndarray]], Dict[str, Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]]]:
    # returns means and confidence intervals for each alg_name (converted using get_display_name())
    # relative confidence intervals for arithmetic mean are a bit wrong
    # because the uncertainty in the divisor is not incorporated

    f = (lambda x: np.log(x + shift_eps)) if use_geometric_mean else (lambda x: x)
    post_f = (lambda x: np.exp(x)) if use_geometric_mean else (lambda x: x)

    if simplify_name_fn is None:
        simplify_name_fn = get_simplified_name

    task_collection = TaskCollection.from_name(coll_name, paths)
    task_infos = task_collection.load_infos(paths)
    task_type_name = 'class' if task_infos[0].tensor_infos['y'].is_cat() else 'reg'
    opt_groups = get_opt_groups(task_type_name)
    alg_group_dict = {'BestModel': (lambda an, tags, config: not an.startswith('Ensemble')), **{
        f'BestModel{group_name}': (lambda an, tags, config, ans=alg_names: an in ans)
        for group_name, alg_names in opt_groups.items()
    }}
    test_table = table.get_test_results_table(DefaultEvalModeSelector(), alg_group_dict=alg_group_dict,
                                              test_metric_name=test_metric_name,
                                              val_metric_name=val_metric_name,
                                              use_validation_errors=use_validation_errors)
    test_table = test_table.rename_algs(simplify_name_fn)
    # print(f'{test_table.alg_names=}')
    # print(f'{filter_alg_names_list=}')
    if filter_alg_names_list is not None:
        test_table = test_table.filter_algs(filter_alg_names_list)

    # new code
    test_table = test_table.filter_n_splits(n_splits)
    # shape: [n_algs, n_tasks, n_splits]
    errors = test_table.to_array()
    if use_ranks:
        errors = get_ranks(errors)
    elif use_normalized_errors:
        min_arr = np.min(errors, axis=0, keepdims=True)
        max_arr = np.max(errors, axis=0, keepdims=True)
        errors = (errors - min_arr) / (max_arr - min_arr + 1e-30)
        errors = np.clip(errors, 0.0, 1.0)
    elif use_grinnorm_errors:
        assert task_type_name in ['class', 'reg']
        min_arr = np.min(errors, axis=0, keepdims=True)
        max_arr = np.quantile(errors, 1.0 if task_type_name == 'class' else 0.9, axis=0, keepdims=True)
        errors = (errors - min_arr) / (max_arr - min_arr + 1e-30)
        if task_type_name == 'reg':
            errors = np.clip(errors, 0.0, 1.0)
        else:
            errors = np.clip(errors, 0.0, np.inf)

    idx_best = test_table.alg_names.index(rel_alg_name) if use_relative_score else 0

    use_task_weighting = coll_name.startswith('meta-train') or coll_name.startswith('uci')
    if use_task_weighting:
        separate_task_names = ['facebook_comment_volume', 'facebook_live_sellers_thailand_shares']
        task_weights = TaskWeighting(test_table.task_infos, separate_task_names).get_task_weights()
    else:
        n_tasks = len(test_table.task_infos)
        task_weights = np.ones(n_tasks) / n_tasks

    f_errors = f(errors)
    mean_f_errors = np.mean(f_errors, axis=-1)
    if use_task_mean:
        mean_f_errors = mean_f_errors @ task_weights
    mean_scores = post_f(mean_f_errors)

    if not use_task_mean:
        assert not use_relative_score
    if return_percentages:
        assert use_relative_score

    base_f_errors = f_errors[idx_best, None] if use_relative_score else np.zeros_like(f_errors)
    # mean_base_f_errors = np.mean(base_f_errors, axis=-1) @ task_weights
    rel_f_errors = f_errors - base_f_errors
    mean_rel_f_errors = np.mean(rel_f_errors, axis=-1)
    if use_task_mean:
        mean_rel_f_errors = mean_rel_f_errors @ task_weights

    # # unbiased estimate of variance of mean estimator
    # variances_algs_tasks = np.var(rel_f_errors, axis=-1) / (n_splits - 1)
    # variances_algs = variances_algs_tasks @ (task_weights ** 2)
    # stds_algs = np.sqrt(variances_algs)
    # lower_rel_mean_f_errors = mean_rel_f_errors - 1.96 * stds_algs
    # upper_rel_mean_f_errors = mean_rel_f_errors + 1.96 * stds_algs

    if use_task_mean:
        # take the mean over tasks first, then do the confidence interval for
        rel_f_errors = np.einsum('ats,t->as', rel_f_errors, task_weights)
    lower_rel_mean_f_errors, upper_rel_mean_f_errors = get_t_mean_confidence_interval(rel_f_errors)
    # lower_rel_mean_f_errors = []
    # upper_rel_mean_f_errors = []
    # for i in range(means_algs_splits.shape[0]):
    #     # following https://www.geeksforgeeks.org/how-to-calculate-confidence-intervals-in-python/
    #     # see also https://stats.stackexchange.com/questions/358408/confidence-interval-for-the-mean-normal-distribution-or-students-t-distributi
    #     # and http://stla.github.io/stlapblog/posts/ModelReduction.html
    #     means_splits = means_algs_splits[i]
    #     sem = scipy.stats.sem(means_splits)
    #     if sem == 0.0:
    #         mean = np.mean(means_splits)
    #         interval = [mean, mean]
    #     else:
    #         interval = scipy.stats.t.interval(confidence=0.95, df=len(means_splits) - 1, loc=np.mean(means_splits),
    #                                           scale=sem)
    #     lower_rel_mean_f_errors.append(interval[0])
    #     upper_rel_mean_f_errors.append(interval[1])
    # lower_rel_mean_f_errors = np.array(lower_rel_mean_f_errors)
    # upper_rel_mean_f_errors = np.array(upper_rel_mean_f_errors)
    # 2.5% and 97.5% quantiles for normal distribution
    lower_rel_mean_scores = post_f(lower_rel_mean_f_errors)
    upper_rel_mean_scores = post_f(upper_rel_mean_f_errors)
    rel_mean_scores = post_f(mean_rel_f_errors)

    # lower_f_errors = mean_f_errors - 1.96 * stds_algs
    # upper_f_errors = mean_f_errors + 1.96 * stds_algs
    # lower_scores = post_f(lower_f_errors)
    # upper_scores = post_f(upper_f_errors)

    def transform(scores: np.ndarray) -> np.ndarray:
        if use_relative_score and not use_geometric_mean:
            # we computed the arithmetic mean of the difference, need to normalize and add 1
            scores = scores / mean_scores[idx_best, None] + 1.0
        if return_percentages:
            scores = 100 * (scores - 1.0)
        return scores

    scores = transform(rel_mean_scores)
    lower_scores = transform(lower_rel_mean_scores)
    upper_scores = transform(upper_rel_mean_scores)

    scores_dict = {alg_name: score
                   for alg_name, score in zip(test_table.alg_names, scores)}
    intervals_dict = {alg_name: (lower, upper)
                      for alg_name, lower, upper in zip(test_table.alg_names, lower_scores, upper_scores)}
    # scores_dict = {display_name_fn(alg_name): score
    #                for alg_name, score in zip(test_table.alg_names, scores)}
    # intervals_dict = {display_name_fn(alg_name): (lower, upper)
    #                   for alg_name, lower, upper in zip(test_table.alg_names, lower_scores, upper_scores)}
    return scores_dict, intervals_dict


def get_opt_groups(task_type_name: str) -> Dict[str, List[str]]:
    """
    Generates a groups of methods that should be evaluated.

    :param task_type_name: 'class' or 'reg'
    :return: A dict of lists {alg_group_name: [alg_name_1, alg_name_2, ...]}
    """
    opt_groups = utils.join_dicts(get_ensemble_groups(task_type_name), {
        '_LGBM-HPO+TD': ['LGBM-HPO', f'LGBM-TD-{task_type_name}'],
        '_XGB-HPO+TD': ['XGB-HPO', f'XGB-TD-{task_type_name}'],
        '_CatBoost-HPO+TD': ['CatBoost-HPO', f'CatBoost-TD-{task_type_name}'],
        '_RealMLP-HPO+TD': ['RealMLP-HPO', f'RealMLP-TD-{task_type_name}'],
        '_MLP-HPO+TD': ['MLP-HPO', f'MLP-TD-{task_type_name}'],
        '-TD_val-ce': [f'RealMLP-TD-{task_type_name}_val-ce_no-ls', f'XGB-TD-{task_type_name}_val-ce',
                       f'LGBM-TD-{task_type_name}_val-ce', f'CatBoost-TD-{task_type_name}_val-ce'],
        '-D_val-ce': [f'MLP-PLR-D-{task_type_name}_val-ce', f'XGB-D-{task_type_name}_val-ce',
                      f'LGBM-D-{task_type_name}_val-ce', f'CatBoost-D-{task_type_name}_val-ce'],
    })

    for method in ['MLP-RTDL-D', 'ResNet-RTDL-D', 'MLP-PLR-D', 'FTT-D', 'TabR-S-D']:
        opt_groups[f'_{method}_prep'] = [f'{method}-{task_type_name}', f'{method}-{task_type_name}_rssc']

    return opt_groups


def get_ensemble_groups(task_type_name: str) -> Dict[str, List[str]]:
    """
    Generates a groups of methods that should be evaluated.

    :param task_type_name: 'class' or 'reg'
    :return: A dict of lists {alg_group_name: [alg_name_1, alg_name_2, ...]}
    """

    return {
        '_GBDTs-TD': [f'XGB-TD-{task_type_name}', f'LGBM-TD-{task_type_name}', f'CatBoost-TD-{task_type_name}'],
        '-TD': [f'XGB-TD-{task_type_name}', f'LGBM-TD-{task_type_name}', f'CatBoost-TD-{task_type_name}',
                f'RealMLP-TD-{task_type_name}'],
        '_GBDTs-HPO': ['XGB-HPO', 'LGBM-HPO', 'CatBoost-HPO'],
        # 'GBDTs-HPO_MLP-HPO': ['XGB-HPO', 'LGBM-HPO', 'CatBoost-HPO', 'MLP-HPO'],  # todo: duplicate
        '-HPO': ['XGB-HPO', 'LGBM-HPO', 'CatBoost-HPO', 'RealMLP-HPO'],
        '_MLP-TD_MLP-TD-S': [f'RealMLP-TD-{task_type_name}', f'RealMLP-TD-S-{task_type_name}'],
        '-D': ['XGB-D', 'LGBM-D', 'CatBoost-D', f'MLP-PLR-D-{task_type_name}'],
    }


def get_simplified_name(alg_name: str):
    alg_name = alg_name.replace(' [bag-1]', '')
    alg_name = alg_name.replace('-class', '').replace('-reg', '')
    # the rest is not happening in get_display_name after merging the names with the names from the runtimes
    # alg_name = alg_name.replace('RF-SKL', 'RF')
    # alg_name = alg_name.replace('-RTDL', '')
    # alg_name = alg_name.replace('_val-ce', '')
    # if alg_name == 'XGBoost-HPO':
    #     return 'XGB-HPO'
    # elif alg_name == 'Ensemble_GBDTs-TD_MLP-TD':
    #     return 'Ensemble_TD'
    # elif alg_name == 'Ensemble_GBDTs-HPO_MLP-HPO':
    #     return 'Ensemble_HPO'
    return alg_name


def get_display_name(alg_name: str) -> str:
    alg_name = alg_name.replace('BestModel', 'Best')
    # alg_name = alg_name.replace('_rssc', '')
    alg_name = alg_name.replace('_rssc', ' (RS+SC)')
    alg_name = alg_name.replace('_no-ls', ' (no LS)')
    alg_name = alg_name.replace('_val-ce', '')
    alg_name = alg_name.replace('RF-SKL', 'RF')
    alg_name = alg_name.replace('-RTDL', '')
    alg_name = alg_name.replace('_best-1-auc-ovr', '')
    if alg_name.endswith('_prep') and alg_name.startswith('Best_'):
        alg_name = alg_name[len('Best_'):-len('_prep')]
        alg_name = alg_name + ' (best of both)'
    return alg_name
