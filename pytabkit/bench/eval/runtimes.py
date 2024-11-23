from typing import Dict

import numpy as np

from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskCollection
from pytabkit.models import utils


def get_avg_train_times(paths: Paths, coll_name: str, per_1k_samples: bool = False) -> Dict[str, float]:
    task_infos = TaskCollection.from_name(coll_name, paths).load_infos(paths)
    alg_names = [path.name for path in paths.times().iterdir()]
    result = dict()
    for alg_name in alg_names:
        file_paths = [paths.times_alg_task(alg_name, task_desc=task_info.task_desc) / 'times.yaml'
                      for task_info in task_infos]
        if all(utils.existsFile(file_path) for file_path in file_paths):
            single_times = [utils.deserialize(file_path, use_yaml=True)['fit_time'] for file_path in file_paths]
            if per_1k_samples:
                # use 0.6 since that is the fraction of training samples
                single_times = [single_time / ((0.6 * task_info.n_samples) / 1000)
                                for single_time, task_info in zip(single_times, task_infos)]
            mean_time = np.mean(single_times)
            result[alg_name] = mean_time
    return result


def get_avg_predict_times(paths: Paths, coll_name: str, per_1k_samples: bool = False) -> Dict[str, float]:
    task_infos = TaskCollection.from_name(coll_name, paths).load_infos(paths)
    alg_names = [path.name for path in paths.times().iterdir()]
    result = dict()
    for alg_name in alg_names:
        file_paths = [paths.times_alg_task(alg_name, task_desc=task_info.task_desc) / 'times.yaml'
                      for task_info in task_infos]
        if all(utils.existsFile(file_path) for file_path in file_paths):
            single_times = [utils.deserialize(file_path, use_yaml=True)['predict_time'] for file_path in file_paths]
            if per_1k_samples:
                # use 0.6 since that is the fraction of training samples
                single_times = [single_time / ((0.2 * task_info.n_samples) / 1000)
                                for single_time, task_info in zip(single_times, task_infos)]
            mean_time = np.mean(single_times)
            result[alg_name] = mean_time
    return result
