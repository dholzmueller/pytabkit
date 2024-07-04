import numbers

import fire
import numpy as np

from pytabkit.bench.data.common import SplitType
from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskCollection
from pytabkit.bench.run.results import ResultManager
from pytabkit.models import utils


def analyze_hpo_best(alg_name: str, coll_name: str, n_splits: int = 10):
    print(f'Analyzing {coll_name}:')
    paths = Paths.from_env_variables()
    task_infos = TaskCollection.from_name(coll_name, paths).load_infos(paths)
    best_params = []
    for task_info in task_infos:
        for split_id in range(n_splits):
            results_path = paths.results_alg_task_split(task_info.task_desc, alg_name, n_cv=1,
                                                        split_type=SplitType.RANDOM, split_id=split_id)
            result_manager = ResultManager.load(results_path, only_metrics=False)
            fit_params = result_manager.other_dict['cv']['fit_params']

            best_params.append(fit_params['hyper_fit_params'] if 'hyper_fit_params' in fit_params
                               else fit_params['sub_fit_params'][0])

            # add keys from sub-dicts like in scikit-learn with __
            flattened_params = {}
            for key, value in best_params[-1].items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        flattened_params[f'{key}__{sub_key}'] = sub_value
            best_params[-1] = utils.join_dicts(best_params[-1], flattened_params)
            # print(best_params[-1])
            # print(result_manager.other_dict)
            # return

    param_names = sorted(list(best_params[0].keys()))

    for param_name in param_names:
        values = [config[param_name] for config in best_params]
        unique_values = []
        # do it manually so that it only requires equality comparison and not hashing or other comparisons
        for v in values:
            if v not in unique_values:
                unique_values.append(v)
        # print(f'Processing {param_name=} with {unique_values=}')

        if len(unique_values) == 1:
            continue  # a hyperparam that hasn't been tuned, most likely
        elif len(unique_values) <= 10:
            print(f'Frequencies of best values for hyperparameter {param_name}:')
            for value in unique_values:
                n_best = len([v for v in values if v == value])
                print(f'{value}: {n_best}')
            print()
        elif all(isinstance(v, numbers.Number) for v in unique_values):
            print(f'Quantiles of best values for hyperparameter {param_name}:')
            for q in np.linspace(0.0, 1.0, 11):
                print(f'alpha={q:g}: {np.quantile(values, q)}')
            print()
        else:
            print(f'No method for printing values of hyperparameter {param_name}')
            print()

    #
    # for act_name in ['relu', 'selu', 'mish']:
    #     n_best = len([config for config in best_params if config['act'] == act_name])
    #     print(f'Number of times that {act_name} was best: {n_best}')


if __name__ == '__main__':
    fire.Fire(analyze_hpo_best)
