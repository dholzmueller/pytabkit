from typing import Dict, List, Optional

from pytabkit.bench.data.common import SplitType
from pytabkit.bench.data.paths import Paths
from pytabkit.models import utils
import numpy as np
import torch

from pytabkit.models.data.data import TensorInfo, TaskType, DictDataset
from pytabkit.models.data.splits import SplitInfo, RandomSplitter, IndexSplitter


# Should a Task/TaskInfo allow to configure the sizes of train/val/test?
# Disadvantages:
# - Might want to compare different train sizes on the same test set
# - How do we distinguish them in a TaskDescription?
# current solution is instead to set this in RunConfig
# alternatively, could consider encoding this in the split type,
# but this would only concern the fraction of test samples

# make default split simply an int so it can be serialized more easily?
# Do we ever need something other than an IndexSplitter?


class TaskDescription:
    """
    The minimal necessary information to identify a task, consisting of a task source and a task name.
    A task is a dataset with a specific target variable.
    """
    def __init__(self, task_source: str, task_name: str):
        """
        :param task_source: Name of the source where the task was retrieved from (see ``data.common.TaskSource``)
        :param task_name: Name of the task (dataset).
        """
        self.task_source = task_source
        self.task_name = task_name

    def load_info(self, paths: Paths) -> 'TaskInfo':
        """
        Load the associated TaskInfo object.

        :param paths: Path configuration.
        :return: Task info object.
        """
        return TaskInfo.load(paths, self)

    def load_task(self, paths: Paths):
        """
        Load the associated Task object.

        :param paths: Path configuration.
        :return: Task object.
        """
        return self.load_info(paths).load_task(paths)

    def exists_task(self, paths: Paths):
        """
        Check if the task for this description is stored on disk.

        :param paths: Path configuration.
        :return: True iff it exists.
        """
        return utils.existsFile(paths.tasks_task(self) / 'info.yaml')

    def __str__(self):
        """
        :return: Description as a string ``f'{self.task_source}/{self.task_name}'``
        """
        return f'{self.task_source}/{self.task_name}'

    def to_dict(self) -> Dict:
        """
        Convert to a dictionary for saving.

        :return: Dictionary with 'task_source' and 'task_name' entries.
        """
        return {'task_source': self.task_source, 'task_name': self.task_name}

    @staticmethod
    def from_dict(data: Dict) -> 'TaskDescription':
        """
        Create from a dictionary.

        :param data: Dictionary.
        :return: TaskDescription object.
        """
        return TaskDescription(task_source=data['task_source'], task_name=data['task_name'])

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        if not isinstance(other, TaskDescription):
            return False
        return self.task_source == other.task_source and self.task_name == other.task_name


class TaskCollection:
    """
    Collection (list) of TaskDescription objects with its own name (can be the name of the task source).
    """
    # there should be a TaskCollection for every TaskSource with the same name
    # but there can be other collections with other names
    def __init__(self, coll_name: str, task_descs: List[TaskDescription]):
        """
        :param coll_name: Name of the task collection.
        :param task_descs: Task descriptions.
        """
        self.coll_name = coll_name
        self.task_descs = task_descs

    def save(self, paths: Paths):
        file = paths.task_collections() / f'{self.coll_name}.yaml'
        data = {'coll_name': self.coll_name, 'task_descs': [td.to_dict() for td in self.task_descs]}
        utils.serialize(file, data, use_yaml=True)

    def load_infos(self, paths: Paths) -> List['TaskInfo']:
        return [desc.load_info(paths) for desc in self.task_descs]

    @staticmethod
    def from_name(coll_name: str, paths: Paths) -> 'TaskCollection':
        file = paths.task_collections() / f'{coll_name}.yaml'
        data = utils.deserialize(file, use_yaml=True)
        task_descs = [TaskDescription.from_dict(d) for d in data['task_descs']]
        return TaskCollection(data['coll_name'], task_descs)

    @staticmethod
    def from_source(task_source: str, paths: Paths) -> 'TaskCollection':
        """
        Create a task collection with all tasks from a given task source
        (that have been imported/saved with this task source name).
        The task collection will have the same name as the source.
        :param task_source: Name of the task source.
        :param paths: Path configuration.
        :return: TaskCollection object.
        """
        path = paths.task_source(task_source)
        if not utils.existsDir(path):
            return TaskCollection(task_source, [])
        task_descs = [TaskDescription(task_source, p.name) for p in path.iterdir()]
        task_descs.sort(key=lambda task_desc: str(task_desc).lower())  # sort by name
        return TaskCollection(task_source, task_descs)


class TaskInfo:
    """
    Information about a task (without containing the dataset itself).
    """
    def __init__(self, task_desc: TaskDescription, n_samples: int, tensor_infos: Dict[str, TensorInfo],
                 default_split_idx: Optional[int], more_info_dict: Optional[Dict], max_n_trainval: Optional[int] = None):
        """
        :param task_desc: Task description.
        :param n_samples: Number of samples.
        :param tensor_infos: Information about the tensors (x_cat, x_cont, y).
        :param default_split_idx: If the dataset has a default split, this is the index of the first test sample.
            We assume that in this case, the training part is stored before the test part.
        :param more_info_dict: Dictionary with more information that can be stored,
            for example about the original OpenML dataset id.
        :param max_n_trainval: maximum number of samples used for training+validation in random splits.
            If None (default value), no maximum is imposed.
        """
        self.task_desc = task_desc
        self.n_samples = n_samples
        self.tensor_infos = tensor_infos
        self.task_type = TaskType.REGRESSION if tensor_infos['y'].is_cont() else TaskType.CLASSIFICATION
        self.default_split_idx = default_split_idx
        self.more_info_dict = more_info_dict or dict()
        self.max_n_trainval = max_n_trainval

    def get_n_classes(self) -> int:
        """
        :return: Number of classes for classification, or 0 for regression.
        """
        return self.tensor_infos['y'].get_cat_size_product()   # we take the product, but it should only be 1 element

    def load_task(self, paths: Paths) -> 'Task':
        """
        Load the associated task.
        :param paths: Path configuration.
        :return: Task object.
        """
        path = paths.tasks_task(self.task_desc)
        tensors = {}
        tensors['x_cont'] = torch.as_tensor(np.load(str(path / 'x_cont.npy'))).type(torch.float32)
        tensors['x_cat'] = torch.as_tensor(np.load(str(path / 'x_cat.npy'))).type(torch.long)
        tensors['y'] = torch.as_tensor(np.load(str(path / 'y.npy'))).type(
            torch.long if self.task_type == TaskType.CLASSIFICATION else torch.float32)
        ds = DictDataset(tensors=tensors, tensor_infos=self.tensor_infos)
        return Task(task_info=self, ds=ds)

    def get_ds_size_gb(self) -> float:
        """
        :return: Dataset size in gigabyte, when stored in torch Tensors
            (8 byte for categorical variables, 4 byte for continuous variables).
        """
        # need 8 byte for categorical variables (torch.long) but only 4 for continuous (torch.float32)
        return self.n_samples * sum([ti.get_n_features() * (8 if ti.is_cat() else 4)
                                     for ti in self.tensor_infos.values()]) / (1024**3)

    def save(self, paths: Paths):
        path = paths.tasks_task(self.task_desc)
        info_dict = {'task_desc': self.task_desc.to_dict(), 'n_samples': self.n_samples,
                     'tensor_infos': {key: value.to_dict() for key, value in self.tensor_infos.items()},
                     'default_split_idx': self.default_split_idx,
                     'more_info_dict': self.more_info_dict,
                     'max_n_trainval': self.max_n_trainval}
        utils.serialize(path / 'info.yaml', info_dict, use_yaml=True)

    @staticmethod
    def load(paths: Paths, task_desc: TaskDescription):
        info_dict = utils.deserialize(paths.tasks_task(task_desc) / 'info.yaml', use_yaml=True)
        return TaskInfo(task_desc=TaskDescription.from_dict(info_dict['task_desc']),
                        n_samples=info_dict['n_samples'],
                        tensor_infos={key: TensorInfo.from_dict(value)
                                      for key, value in info_dict['tensor_infos'].items()},
                        default_split_idx=info_dict['default_split_idx'],
                        more_info_dict=info_dict.get('more_info_dict', dict()),
                        max_n_trainval=info_dict.get('max_n_trainval', None))

    @staticmethod
    def from_ds(task_desc: TaskDescription, ds: DictDataset, default_split_idx: Optional[int] = None,
                more_info_dict: Optional[Dict] = None) -> 'TaskInfo':
        return TaskInfo(task_desc=task_desc, n_samples=ds.n_samples, tensor_infos=ds.tensor_infos,
                        default_split_idx=default_split_idx, more_info_dict=more_info_dict)

    def get_random_splits(self, n_splits, first_fraction=0.8) -> List[SplitInfo]:
        # use n_samples to generate alg_seed
        # in order to have the randomness also depend on the data set and not only on the split index
        return [SplitInfo(RandomSplitter(seed=i, first_fraction=first_fraction, max_n_first=self.max_n_trainval),
                          SplitType.RANDOM, id=i,
                          alg_seed=utils.combine_seeds(self.n_samples, i))
                for i in range(n_splits)]

    def get_default_splits(self, n_splits) -> List[SplitInfo]:
        if self.default_split_idx is None:
            return []
        else:
            return [SplitInfo(IndexSplitter(self.default_split_idx), SplitType.DEFAULT, id=i,
                              alg_seed=utils.combine_seeds(self.n_samples, i))
                    for i in range(n_splits)]


class Task:
    """
    Task (dataset with defined target variable),
    consisting of a task info and a dataset.
    """
    def __init__(self, task_info: TaskInfo, ds: DictDataset):
        self.task_info = task_info
        self.ds = ds  # data is on CPU here

    def save(self, paths: Paths):
        path = paths.tasks_task(self.task_info.task_desc)
        utils.ensureDir(path / 'x_cont.npy')
        np.save(str(path / 'x_cont.npy'), self.ds.tensors['x_cont'].type(torch.float32).numpy())
        np.save(str(path / 'x_cat.npy'), self.ds.tensors['x_cat'].type(torch.int32).numpy())
        np.save(str(path / 'y.npy'), self.ds.tensors['y'].type(
            torch.int32 if self.task_info.task_type == TaskType.CLASSIFICATION else torch.float32).numpy())
        self.task_info.save(paths)


class TaskPackage:
    """
    Combines information about how to run a task on a benchmark.
    """
    def __init__(self, task_info: TaskInfo, split_infos: List[SplitInfo], n_cv: int, n_refit: int, paths: Paths,
                 rerun: bool, alg_name: str, save_y_pred: bool):
        self.task_info = task_info
        self.split_infos = split_infos
        self.n_cv = n_cv
        self.n_refit = n_refit
        self.paths = paths
        self.rerun = rerun
        self.alg_name = alg_name
        self.save_y_pred = save_y_pred







