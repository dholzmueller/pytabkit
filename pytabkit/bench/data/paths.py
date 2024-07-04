import os
import uuid
from pathlib import Path
from typing import Optional

from pytabkit.models import utils
import shutil


class TmpPathContextManager:
    """
    Helper class: Context manager for creating temporary paths.
    """
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self) -> Path:
        if utils.existsDir(self.path):
            raise RuntimeError('Temporary path already exists:', self.path)
        utils.create_dir(self.path)
        return self.path

    def __exit__(self, type, value, traceback):
        shutil.rmtree(self.path)


class Paths:
    """
    This class provides paths where data can be stored. Its base path can be configured.
    It requires one base folder, which will have several subfolders:
    algs, tasks, task_collections, results, result_summaries, eval, plots, tmp, ...
    by subclassing this class, specific folders can be re-located (e.g. put data on SSD)
    """

    def __init__(self, base_folder: str, tasks_folder: Optional[str] = None, results_folder: Optional[str] = None,
                 result_summaries_folder: Optional[str] = None, uci_download_folder: Optional[str] = None):
        self.base_path = Path(base_folder)
        self.tasks_path = Path(tasks_folder) if tasks_folder is not None else self.base_path / 'tasks'
        self.results_path = Path(results_folder) if results_folder is not None else self.base_path / 'results'
        self.result_summaries_path = Path(
            result_summaries_folder) if result_summaries_folder is not None else self.base_path / 'result_summaries'
        self.uci_download_path = Path(
            uci_download_folder) if uci_download_folder is not None else self.base_path / 'uci_download'

    @staticmethod
    def from_env_variables() -> 'Paths':
        """
        Construct a Paths object that is constructed from environment variables if they are set.
        Otherwise, the base folder will either be taken from custom_paths.py,
        if available, or set to './tab_bench_data'.
        :return: Paths object.
        """
        base_folder = os.environ.get('TAB_BENCH_DATA_BASE_FOLDER', None)
        if base_folder is None:
            try:
                from scripts import custom_paths
                base_folder = custom_paths.get_base_folder()
            except:
                base_folder = './tab_bench_data'
        tasks_folder = os.environ.get('TAB_BENCH_DATA_TASKS_FOLDER', None)
        results_folder = os.environ.get('TAB_BENCH_DATA_RESULTS_FOLDER', None)
        result_summaries_folder = os.environ.get('TAB_BENCH_DATA_RESULT_SUMMARIES_FOLDER', None)
        uci_download_folder = os.environ.get('TAB_BENCH_DATA_UCI_DOWNLOAD_FOLDER', None)
        return Paths(base_folder=base_folder, tasks_folder=tasks_folder, results_folder=results_folder,
                     result_summaries_folder=result_summaries_folder, uci_download_folder=uci_download_folder)

    def base(self) -> Path:
        return self.base_path

    def algs(self) -> Path:
        return self.base() / 'algs'

    def tasks(self) -> Path:
        return self.tasks_path

    def task_collections(self) -> Path:
        return self.base() / 'task_collections'

    def results(self) -> Path:
        return self.results_path

    def result_summaries(self) -> Path:
        return self.result_summaries_path

    def eval(self) -> Path:
        return self.base() / 'eval'

    def plots(self) -> Path:
        return self.base() / 'plots'

    def tmp(self) -> Path:
        return self.base() / 'tmp'

    def uci_download(self) -> Path:
        return self.uci_download_path

    def resources(self):
        return self.base() / 'resources'

    def times(self) -> Path:
        return self.base() / 'times'

    def new_tmp_folder(self) -> TmpPathContextManager:
        # https://stackoverflow.com/questions/2759644/python-multiprocessing-doesnt-play-nicely-with-uuid-uuid4
        return TmpPathContextManager(self.tmp() / str(uuid.UUID(bytes=os.urandom(16), version=4)))

    def results_alg_task(self, task_desc: 'TaskDescription', alg_name: str, n_cv: int) -> Path:
        return self.results() / alg_name / task_desc.task_source / task_desc.task_name / f'{n_cv}-fold'

    def summary_alg_task(self, task_desc: 'TaskDescription', alg_name: str, n_cv: int) -> Path:
        return self.result_summaries() / alg_name / task_desc.task_source / task_desc.task_name \
            / f'{n_cv}-fold'

    def results_alg_task_split(self, task_desc: 'TaskDescription', alg_name: str, n_cv: int, split_type: str,
                               split_id: int) -> Path:
        return self.results_alg_task(task_desc, alg_name, n_cv) / split_type / str(split_id)

    def tasks_task(self, task_desc: 'TaskDescription') -> Path:
        return self.tasks() / task_desc.task_source / task_desc.task_name

    def results_task(self, task_desc: 'TaskDescription') -> Path:
        return self.results() / task_desc.task_source / task_desc.task_name

    def resources_exp_it(self, exp_name: str, iteration: int) -> Path:
        return self.resources() / exp_name / str(iteration)

    def task_source(self, task_source_name: str) -> Path:
        return self.tasks() / task_source_name

    def times_alg_task(self, alg_name: str, task_desc: 'TaskDescription'):
        return self.times() / alg_name / task_desc.task_source / task_desc.task_name
