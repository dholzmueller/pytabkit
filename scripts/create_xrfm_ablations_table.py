from typing import List, Optional

import numpy as np
from pytabkit.bench.run.results import ResultManager

from pytabkit.bench.data.common import SplitType
from pytabkit.bench.data.paths import Paths
from pytabkit.bench.eval.analysis import ResultsTables, get_benchmark_results
from pytabkit.bench.eval.tables import _get_table_str
from pytabkit.models import utils


def generate_xrfm_ablations_results_table(paths: Paths, tables: ResultsTables, filename: str, coll_name: str,
                                          test_metric_name: Optional[str] = None,
                                          val_metric_name: Optional[str] = None):
    table = tables.get(coll_name, tag='default')

    alg_display_names = {
        'xRFM-HPO-paper-large_new': 'AGOP',
        'xRFM-HPO-large-temptune_new': 'AGOP + TT',
        'xRFM-HPO-large-temptune-pca_new': 'PCA + TT',
        'xRFM-HPO-large-temptune-rf_new': 'RF + TT'
    }
    alg_names = list(alg_display_names.keys())

    means, intervals = get_benchmark_results(paths, table, coll_name=coll_name, use_relative_score=False,
                                             test_metric_name=test_metric_name, val_metric_name=val_metric_name,
                                             return_percentages=False, use_task_mean=False, use_geometric_mean=False,
                                             n_splits=1)

    alg_names = [an for an in alg_names if an in means]

    table_head = [['', r'\multicolumn{4}{c}{Splitting method}'],
                  ['Dataset'] + [alg_display_names[an] for an in alg_names]]
    table_body = []

    enumerated_task_infos = list(enumerate(table.test_table.task_infos))
    enumerated_task_infos.sort(key=lambda tup: tup[1].task_desc.task_name.lower())

    print(f'{coll_name=}')
    print(f'{list(means.keys())=}')

    def get_score_strings(scores: List[float], maximize: bool = False, use_int: bool = False) -> List[str]:
        best_row_score = np.max(scores) if maximize else np.min(scores)
        is_best_list = [score == best_row_score for score in scores]
        row_strs = []
        for is_best, row_score in zip(is_best_list, scores):
            cur_str = str(round(row_score)) if use_int else f'{row_score:5.4f}'
            if is_best:
                cur_str = r'\textbf{' + cur_str + r'}'
            row_strs.append(cur_str)
        return row_strs

    for task_idx, task_info in enumerated_task_infos:
        row_scores = [means[alg_name][task_idx] for alg_name in alg_names]
        table_body.append([task_info.task_desc.task_name] + get_score_strings(row_scores))

    # escape underscores for latex
    table_head = [[val.replace('_', r'\_') for val in row] for row in table_head]
    table_body = [[val.replace('_', r'\_') for val in row] for row in table_body]

    # generate bottom part, first average scores
    # indexed by [task][alg]
    scores_matrix = np.asarray(
        [[means[alg_name][task_idx] for alg_name in alg_names] for task_idx, _ in enumerated_task_infos])
    n_wins = (scores_matrix == np.min(scores_matrix, axis=1)[:, None]).astype(np.int32).sum(axis=0).tolist()
    table_foot = [['Number of wins:'] + get_score_strings(n_wins, maximize=True, use_int=True),
                  ['Shifted geometric mean:'] \
                  + get_score_strings(np.exp(np.mean(np.log(scores_matrix + 0.01), axis=0)).tolist()),
                  ['Arithmetic mean:'] \
                  + get_score_strings(np.mean(scores_matrix, axis=0).tolist())
                  ]
    # get runtimes
    mean_fit_times = []
    mean_eval_times = []
    for alg_name in alg_names:
        fit_times = []
        eval_times = []
        for task_idx, task_info in enumerated_task_infos:
            fit_time = 0.0
            eval_time = 0.0
            for hpo_step in range(30):
                path = paths.results_alg_task_split(task_desc=task_info.task_desc,
                                                    alg_name=alg_name + f'_step-{hpo_step}', n_cv=1,
                                                    split_type=SplitType.RANDOM, split_id=0)
                rm = ResultManager.load(path, load_preds=False)
                fit_time += rm.other_dict['cv']['fit_time_s']
                eval_time += rm.other_dict['cv']['eval_time_s']
            fit_times.append(fit_time)
            eval_times.append(eval_time)
        mean_fit_times.append(np.mean(fit_times))
        mean_eval_times.append(np.mean(eval_times))
    table_foot.append(['Average fit time [s]:'] + get_score_strings(mean_fit_times, use_int=True))
    table_foot.append(['Average eval time [s]:'] + get_score_strings(mean_eval_times, use_int=True))

    table_str = _get_table_str(table_head, table_body, table_foot)
    file_path = paths.plots() / filename
    utils.writeToFile(file_path, table_str)


if __name__ == '__main__':
    paths = Paths.from_env_variables()

    tables = ResultsTables(paths)

    for coll_name in ['meta-test-large-class', 'meta-test-large-reg']:
        generate_xrfm_ablations_results_table(paths, tables, f'individual_results_{coll_name}.tex',
                                              coll_name=coll_name)
