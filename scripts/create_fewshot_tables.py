from typing import List, Optional

import numpy as np
from pytabkit.bench.run.results import ResultManager

from pytabkit.bench.data.common import SplitType
from pytabkit.bench.data.paths import Paths
from pytabkit.bench.eval.analysis import ResultsTables, get_benchmark_results, get_simplified_name
from pytabkit.bench.eval.tables import _get_table_str
from pytabkit.models import utils


def print_fewshot_table(paths: Paths, tables: ResultsTables, regression: bool,
                        test_metric_name: Optional[str] = None,
                        val_metric_name: Optional[str] = None):
    coll_name = 'talent-reg-tiny' if regression else 'talent-class-tiny'
    table = tables.get(coll_name, tag='fewshot', n_cv=4)

    alg_display_names = {
        'RealTabPFN-2.5': 'RealTabPFN-2.5',
        'TabICLv2': 'TabICooL',
        'RF-D': 'RandomForest',
        'ExtraTrees-D': 'ExtraTrees',
        'RealMLP-TD-tabarena': 'RealMLP-TD (TabArena)',
        'RealMLP-TD': 'RealMLP-TD',
        'XGB-TD': 'XGBoost-TD',
        'XGB-PBB': 'XGBoost-PBB-D',
        'LGBM-TD': 'LightGBM-TD',
        'CatBoost-TD': 'CatBoost-TD',
        'LGBM-fewshot': 'LightGBM-Huertas',
        'TabM-D': 'TabM',
        'Dummy': 'Dummy',
    }

    means, intervals = get_benchmark_results(
        paths, table, coll_name=coll_name, use_relative_score=False,
        test_metric_name=test_metric_name, val_metric_name=val_metric_name,
        return_percentages=False, use_task_mean=True, use_geometric_mean=False,
        use_ranks=True,
        simplify_name_fn=lambda name: name.replace('-class', '').replace('-reg', '').replace('_fewshot', '').replace(' [bag-4]', ''),
        filter_alg_names_list=list(alg_display_names.keys()),
        n_splits=2)

    # print(f'{means=}')

    names_list = list(alg_display_names.keys())
    means_values = [means[name] for name in alg_display_names]
    perm = np.argsort(means_values)
    print(f'Average ranks on {coll_name}:')
    for idx in perm:
        print(f'{alg_display_names[names_list[idx]]}: {means_values[idx]:.3f}')


if __name__ == '__main__':
    paths = Paths.from_env_variables()

    tables = ResultsTables(paths)

    for regression, metric_name in [(False, '1-auc_ovr'), (True, 'rmse')]:
        print_fewshot_table(paths, tables, regression=regression,
                            val_metric_name=metric_name, test_metric_name=metric_name)
