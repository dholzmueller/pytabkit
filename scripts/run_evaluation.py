import time
from typing import Optional

import numpy as np

import fire

from pytabkit.bench.data.common import SplitType
from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskDescription, TaskCollection
from pytabkit.bench.eval.analysis import get_opt_groups
from pytabkit.bench.eval.evaluation import MultiResultsTable, DefaultEvalModeSelector, MeanTableAnalyzer, \
    alg_results_str, \
    alg_comparison_str, WinsTableAnalyzer, RankTableAnalyzer, NormalizedLossTableAnalyzer, \
    GreedyAlgSelectionTableAnalyzer


def show_eval(coll_name: str = 'meta-train-class', n_cv: int = 1, show_alg_groups: bool = True,
              val_metric_name: str = None, metric_name: str = None, split_type: str = SplitType.RANDOM,
              use_task_weighting: Optional[bool] = None, shift_eps: float = 0.01,
              data_path: Optional[str] = None, alg_name: Optional[str] = None,
              alg_name_2: Optional[str] = None, tag: Optional[str] = None, max_n_splits: Optional[int] = None,
              max_n_algs: Optional[int] = None, show_val_results: bool = False, show_train_results: bool = False,
              algs_prefix: Optional[str] = None, algs_suffix: Optional[str] = None, algs_contains: Optional[str] = None,
              exclude_datasets: Optional[str] = None):
    """
    Prints evaluation tables on the selected datasets/algorithms.
    The following aggregate statistics will be printed, all of which are
    based on the specified metric and validation metric:

    - log shifted geometric mean test metric when greedily creating an algorithm portfolio based on the validation
      results. The algorithms are sorted by order of inclusion into the portfolio.
      The scores are the scores of selecting the best algorithm out of the portfolio up to this point
      on every dataset separately, based on the validation sets.
    - Win fraction: Fraction of datasets (may be weighted) on which this algorithm is the best one.
    - Arithmetic mean rank
    - Arithmetic mean normalized test metric: The best method is normalized to 0 and the worst one to 1.
    - Arithmetic mean test metric
    - Log shifted geometric mean test metric: mean(log(metric+shift_eps))
    - Shifted geometric mean test metric: exp(mean(log(metric+shift_eps)))

    :param coll_name: Name of the task collection, e.g., 'meta-train-class'
    :param n_cv: Number of cross-validation folds.
        Will only print results for algorithms that have been evaluated with this number of cross-validation folds.
    :param show_alg_groups: Whether to show aggregate algorithms,
        such as the one that picks the best method on the validation set out of the displayed methods.
    :param val_metric_name: Name of the validation metric, used for the algorithm groups. By default, the same value as
        metric_name will be used.
    :param metric_name: Name of the metric that should be displayed (default = classification error / RMSE).
    :param split_type: Type of the split, normally random_split.
    :param use_task_weighting: Whether to weight tasks for the evaluation. If false, uniform weights are used.
        If True, weights based on prefixes are used. By default, weights are used only for meta-train collections.
    :param shift_eps: Epsilon parameter used in the shifted geometric mean.
    :param data_path: Path to the data folder where results are saved.
        By default, this function will take the path from Paths.from_env_variables().
    :param alg_name: Algorithm for which results on individual datasets should be printed
    :param alg_name_2: Second algorithm for which results on individual datasets should be printed.
    :param tag: If specified, only print algorithms whose tags include the given tag.
    :param max_n_splits: If specified, only evaluate the given number of train-test splits.
    :param max_n_algs: Maximum number of methods that should be processed and displayed.
    This does not contain groups of methods (e.g. "all algs") that will be added on top later.
    :param show_val_results: Whether to show validation errors instead of test errors.
    :param show_train_results: Whether to show training errors instead of test errors.
    :param algs_prefix: If specified, only methods with this prefix will be displayed.
    :param algs_suffix: If specified, only methods with this suffix will be displayed.
    :param algs_contains: If specified, only methods containing this substring will be displayed.
    :param exclude_datasets: Optional comma-separated list of datasets that will be excluded from the analysis.
    :return:
    """
    print('start show eval')
    paths = Paths(data_path) if data_path is not None else Paths.from_env_variables()
    start_time = time.time()
    if '/' in coll_name:
        # use a single task
        parts = coll_name.split('/')
        if len(parts) != 2:
            print(f'Too many / in coll_name {coll_name}')
            return
        task_collection = TaskCollection(coll_name, [TaskDescription(*parts)])
    else:
        task_collection = TaskCollection.from_name(coll_name, paths)
    if exclude_datasets:
        exclude_names = exclude_datasets.split(',')
        task_collection = TaskCollection(task_collection.coll_name,
                                         [td for td in task_collection.task_descs if td.task_name not in exclude_names])
    print('load table')
    # table = MultiResultsTable.load_summaries(task_collection, n_cv=n_cv, paths=paths)
    if tag is None:
        alg_filter = None
    else:
        show_tags = tag.split(',') if isinstance(tag, str) else list(
            tag)  # commas are converted to tuples in the command line, apparently
        alg_filter = lambda an, tags, config: (np.any([show_tag in tags for show_tag in show_tags])
                                               and (algs_prefix is None or an.startswith(algs_prefix))
                                               and (algs_suffix is None or an.endswith(algs_suffix))
                                               and (algs_contains is None or algs_contains in an))
    table = MultiResultsTable.load(task_collection, n_cv=n_cv, paths=paths, max_n_algs=max_n_algs,
                                   split_type=split_type, alg_filter=alg_filter, max_n_splits=max_n_splits)
    print('process table')
    # alg_group_dict = {'all algs': (lambda an, tags, config: True)} if show_alg_groups else None
    task_type_name = 'class' if 'class' in coll_name else 'reg'
    opt_groups = get_opt_groups(task_type_name)
    alg_group_dict = {'BestModel': (lambda an, tags, config: not an.startswith('Ensemble')), **{
        f'BestModel{group_name}': (lambda an, tags, config, ans=alg_names: an in ans)
        for group_name, alg_names in opt_groups.items()
    }}
    if not show_alg_groups:
        alg_group_dict = None
    if alg_name is not None and alg_name_2 is not None and show_alg_groups:
        alg_group_dict['selected algs'] = (lambda an, tags, config, grp=[alg_name, alg_name_2]:
                                           np.any([g.startswith(an) for g in grp]))

    val_test_groups = {f'HPO-on-BestModel-TD-{task_type_name}': {f'{family}-TD-{task_type_name}': f'{family}-HPO'
                                                                 for family in ['XGB', 'LGBM', 'CatBoost', 'MLP']}
                       for task_type_name in ['class', 'reg']}

    if val_metric_name is None:
        val_metric_name = metric_name

    test_table = table.get_test_results_table(DefaultEvalModeSelector(), alg_group_dict=alg_group_dict,
                                              test_metric_name=metric_name, val_metric_name=val_metric_name,
                                              val_test_groups=val_test_groups, use_validation_errors=show_val_results,
                                              use_train_errors=show_train_results)
    val_table_single = table.get_test_results_table(DefaultEvalModeSelector(), alg_group_dict=dict(),
                                                    test_metric_name=metric_name, val_metric_name=val_metric_name,
                                                    val_test_groups=val_test_groups, use_validation_errors=True)
    test_table_single = table.get_test_results_table(DefaultEvalModeSelector(), alg_group_dict=dict(),
                                                     test_metric_name=metric_name, val_metric_name=val_metric_name,
                                                     val_test_groups=val_test_groups,
                                                     use_validation_errors=show_val_results,
                                                     use_train_errors=show_train_results)

    if len(test_table.alg_task_results) == 0:
        print(f'No results found')
        return

    subset = 'train' if show_train_results else ('val' if show_val_results else 'test')

    if use_task_weighting is None:
        use_task_weighting = coll_name.startswith('meta-train') or coll_name.startswith('uci')
    separate_task_names = ['facebook_comment_volume', 'facebook_live_sellers_thailand_shares']
    if n_cv == 1:
        # fails for n_cv > 1 because proper selection on the validation set is not implemented
        print(
            f'Greedy algorithm selection cumulative best log shifted geometric mean (err+{shift_eps:g}) {subset} error:')
        analyzer = GreedyAlgSelectionTableAnalyzer(use_weighting=use_task_weighting,
                                                   separate_task_names=separate_task_names,
                                                   f=lambda x: np.log(x + shift_eps))
        analyzer.print_analysis(test_table_single, val_table_single)
        print()
    print('Win fraction:')
    analyzer = WinsTableAnalyzer(use_weighting=use_task_weighting, separate_task_names=separate_task_names)
    analyzer.print_analysis(test_table)
    print()
    print('Arithmetic mean rank:')
    analyzer = RankTableAnalyzer(use_weighting=use_task_weighting, separate_task_names=separate_task_names)
    analyzer.print_analysis(test_table)
    print()
    print(f'Arithmetic mean normalized {subset} metric:')
    analyzer = NormalizedLossTableAnalyzer(use_weighting=use_task_weighting, separate_task_names=separate_task_names)
    analyzer.print_analysis(test_table)
    print()
    print(f'Arithmetic mean {subset} metric:')
    analyzer = MeanTableAnalyzer(use_weighting=use_task_weighting, separate_task_names=separate_task_names)
    analyzer.print_analysis(test_table)
    print()
    print(f'Shifted geometric mean (err+{shift_eps:g}) {subset} metric:')
    analyzer = MeanTableAnalyzer(f=lambda x: np.log(x + shift_eps), use_weighting=use_task_weighting,
                                 separate_task_names=separate_task_names,
                                 post_f=lambda x: np.exp(x))
    analyzer.print_analysis(test_table)
    print()
    print(f'Log shifted geometric mean (err+{shift_eps:g}) {subset} metric:')
    analyzer = MeanTableAnalyzer(f=lambda x: np.log(x + shift_eps), use_weighting=use_task_weighting,
                                 separate_task_names=separate_task_names)
    analyzer.print_analysis(test_table)
    print()
    # print('Mean modlog test error:')  # todo: name modlog is suboptimal, people could associate mod with modulo
    # analyzer = MeanTableAnalyzer(f=lambda x: np.log(x + 1e-3) - np.log(1e-3), use_weighting=use_task_weighting)
    # analyzer.print_analysis(test_table)
    if alg_name is not None:
        if alg_name_2 is None:
            print(f'Errors for alg {alg_name}:')
            print(alg_results_str(test_table, alg_name))
        else:
            print(f'Comparison: {alg_name} vs. {alg_name_2}')
            print(alg_comparison_str(test_table, [alg_name, alg_name_2]))
    print(f'Time for printing: {time.time() - start_time:g} s')


if __name__ == '__main__':
    fire.Fire(show_eval)
