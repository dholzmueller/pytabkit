from pathlib import Path
from typing import Dict, List

from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskInfo
from pytabkit.models import utils


class ResultManager:
    """
    Stores experimental results and can save and load them.
    """

    def __init__(self):
        # indexing convention:
        # self.metrics_dict['cv'/'refit']['train'/'val'/'test'][str(n_models)][str(start_idx)][metric_name] = float
        self.metrics_dict = {}

        # indexed by ['cv'/'refit'], then for example fields like ['y_preds'], ['fit_params']
        # or ['sub_info'] for hyperopt sub-results
        self.other_dict = {}

    def add_results(self, is_cv: bool, results_dict: Dict) -> None:
        """
        Add a dictionary of results.
        :param is_cv: Whether these results are from cross-validation (True) or refitting (False).
        :param results_dict: Dictionary of results
        """
        cv_str = 'cv' if is_cv else 'refit'
        if cv_str not in self.metrics_dict:
            self.metrics_dict[cv_str] = {}
        if cv_str not in self.other_dict:
            self.other_dict[cv_str] = {}
        for key, value in results_dict.items():
            if key == 'metrics':
                self.metrics_dict[cv_str] = value
            else:
                self.other_dict[cv_str][key] = value

    def save(self, path: Path) -> None:
        utils.serialize(path / 'metrics.yaml', self.metrics_dict, use_yaml=True)
        utils.serialize(path / 'other.msgpack.gz', self.other_dict, use_msgpack=True, compressed=True)

    @staticmethod
    def load(path: Path, only_metrics: bool = False):
        """
        Load results.
        :param path: Data path.
        :param only_metrics: If True, only the metrics results are loaded
        (this can be much faster, especially if predictions are also stored).
        :return:
        """
        rm = ResultManager()
        rm.metrics_dict = utils.deserialize(path / 'metrics.yaml', use_yaml=True)
        if not only_metrics:
            rm.other_dict = utils.deserialize(path / 'other.msgpack.gz', use_msgpack=True, compressed=True)
        return rm


def save_summaries(paths: Paths, task_infos: List[TaskInfo], alg_name: str, n_cv: int, rerun=False) -> None:
    """
    Compress the results into result_summaries that can be loaded faster for evaluation.
    :param paths: Path configuration.
    :param task_infos: Task infos of tasks that should be summarized.
    :param alg_name: Name of the method whose results should be summarized.
    :param n_cv: Number of cross-validation splits for which the results should be summarized.
    :param rerun: Whether to re-compute the summaries even if summaries are already present.
    """
    for task_info in task_infos:
        task_desc = task_info.task_desc
        src_path = paths.results_alg_task(task_desc, alg_name, n_cv)
        dest_path = paths.summary_alg_task(task_desc, alg_name, n_cv)
        if not rerun and utils.existsDir(dest_path):
            continue

        # indexed by [split_type][split_idx]['cv'/'refit']['train'/'val'/'test'][str(n_models)][str(start_index)][metric_name]
        metrics_dict = {}
        for split_type_path in src_path.iterdir():
            split_type = split_type_path.name
            split_id_metrics_list = []
            split_id = 0
            while True:
                split_id_path = split_type_path / str(split_id)
                if not utils.existsDir(split_id_path):
                    break
                rm = ResultManager.load(split_id_path, only_metrics=True)
                split_id_metrics_list.append(rm.metrics_dict)
                split_id += 1
            if split_id >= 1:
                # there exists a split
                metrics_dict[split_type] = split_id_metrics_list
        if len(metrics_dict) > 0:
            # shift split_idx dimension to the end
            results_dict = utils.shift_dim_nested(metrics_dict, 1, 6)
            # print(f'{results_dict=}')
            # results_dict[split_type]['cv'/'refit']['train'/'val'/'test'][str(n_models)][str(start_idx)][metric_name][split_idx]
            utils.serialize(dest_path / 'metrics.msgpack.gz', results_dict, use_msgpack=True, compressed=True)


