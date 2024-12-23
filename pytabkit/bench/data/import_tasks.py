from typing import Union, Optional, List, Dict

import sklearn.model_selection

import torch
from pathlib import Path
import numpy as np
import pandas as pd

from pytabkit.bench.data.common import TaskSource
from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskDescription, TaskInfo, Task, TaskCollection
from pytabkit.models import utils
from pytabkit.models.data.data import TaskType, DictDataset, TensorInfo


def download_if_not_exists(url: str, dest: str):
    import requests
    """
    Simple function for downloading a file from an url if no file at the destination path exists.
    :param url: URL of the file to download.
    :param dest: Path where to save the downloaded file.
    """
    # following https://dzone.com/articles/simple-examples-of-downloading-files-using-python
    utils.ensureDir(dest)
    if not utils.existsFile(dest):
        print('Downloading ' + url, flush=True)
        # file = requests.get(url)
        # open(dest, 'wb').write(file.content)
        r = requests.get(url, stream=True)
        with open(dest, 'wb') as f:
            print('Progress (dot = 1 MB): ', end='', flush=True)
            for ch in r.iter_content(chunk_size=1024**2):
                print('.', end='', flush=True)
                f.write(ch)
            print(flush=True)


def extract_categories(X):
    n_cols = X.shape[1]
    n_samples = X.shape[0]
    is_categorical = np.asarray([np.allclose(np.abs(X[:, i]), 1.0) for i in range(n_cols)])

    cat_idx_groups = []
    i = 0
    while i < n_cols:
        if not is_categorical[i]:
            i += 1
            continue
        compat_signs = []
        while i < n_cols:
            signs = X[:, i] > 0
            if np.any([np.any(np.logical_and(signs, cs)) for cs in compat_signs]):
                break
            compat_signs.append(signs)
            i += 1

        cat_idx_groups.append(list(np.arange(i - len(compat_signs), i)))

    cont_idxs = list(np.argwhere(~is_categorical)[:, 0])
    X_conts = X[:, cont_idxs] if len(cont_idxs) > 0 else np.zeros(shape=(n_samples, 0), dtype=np.float32)
    signs = X > 0
    # for binary categorical variables, shift by 1 since the category 0 is reserved for missing values
    X_cats = [np.sum(signs[:, g] * np.arange(1, len(g) + 1), axis=1) + (1 if len(g) == 1 else 0) for g in cat_idx_groups]
    X_cats = np.stack(X_cats, axis=1).astype(np.int32) if len(X_cats) > 0 else np.zeros(shape=(n_samples, 0), dtype=np.int32)
    # binary categorical variables need to be shifted one more since here
    # "-1" is not already the missing variable category
    cat_sizes = [len(group) + 1 + (1 if len(group) == 1 else 0) for group in cat_idx_groups]
    return X_conts, X_cats, cat_sizes


def check_zero_hot(uci_base_path):
    uci_base = Path(uci_base_path)
    uci_paths = [uci_base / 'bin-class-data',
                   uci_base / 'multi-class-data',
                   uci_base / 'regression-data']

    for path in uci_paths:
        ds_names = [file.stem for file in path.iterdir() if file.is_file()]
        ds_names.sort()
        for ds_name in ds_names:
            print('Processing dataset', ds_name)
            ds_path = path / (ds_name + '.csv')

            data = np.genfromtxt(ds_path, delimiter=',')
            X = data[:, 1:]
            X_cont, X_cat, cat_sizes = extract_categories(X)
            if np.any(np.logical_and(np.min(X_cat, axis=0) == 0, np.max(X_cat, axis=0) >= 2)):
                print('This dataset has a zero-hot encoding')


def convert_to_class_numbers(y):
    y = np.rint(y)
    y_target = np.zeros(y.shape, dtype=np.int32)
    classes = np.unique(y)
    n_classes = len(classes)
    for i, c in enumerate(classes):
        y_target[y == c] = i
    return y_target, n_classes


def import_from_csv(ds_path: Union[Path, str], task_type: TaskType, task_desc: TaskDescription, paths: Paths,
                    default_split_idx: Optional[int] = None, remove_duplicates: bool = False):
    data = np.genfromtxt(ds_path, delimiter=',')
    X = data[:, 1:]
    y = data[:, 0]
    x_cont, x_cat, cat_sizes = extract_categories(X)
    n_classes = 0

    if remove_duplicates:
        # check for duplicates
        df_cont = pd.DataFrame(x_cont)
        df_cat = pd.DataFrame(x_cat)
        df_combined = pd.concat([df_cont, df_cat], axis=1)  # Concatenate the two DataFrames along the column axis
        is_duplicated = df_combined.duplicated()
        if is_duplicated.any():
            print(f'Warning: Data set contains {is_duplicated.sum()} duplicate values! Removing duplicates...')
            not_duplicated_np = (~is_duplicated).values
            x_cont = x_cont[not_duplicated_np]
            x_cat = x_cat[not_duplicated_np]
            y = y[not_duplicated_np]

    # preprocess y
    if task_type == TaskType.CLASSIFICATION:
        y, n_classes = convert_to_class_numbers(y)
    elif task_type == TaskType.REGRESSION:
        # normalize y
        y = (y - np.mean(y, axis=-1)) / (np.std(y, axis=-1) + 1e-30)
    ds = DictDataset({'x_cont': torch.as_tensor(x_cont, dtype=torch.float32),
                      'x_cat': torch.as_tensor(x_cat, dtype=torch.long),
                      'y': torch.as_tensor(y[:, None])},
                     {'x_cont': TensorInfo(feat_shape=[x_cont.shape[-1]]),
                      'x_cat': TensorInfo(cat_sizes=cat_sizes),
                      'y': TensorInfo(cat_sizes=[n_classes])})
    task_info = TaskInfo.from_ds(task_desc, ds, default_split_idx=default_split_idx)
    task = Task(task_info, ds)
    task.save(paths)


def import_uci_tasks(paths: Paths, remove_duplicates: bool = False, rerun=False):
    uci_base = Path(paths.uci_download())
    uci_matches = [(TaskSource.UCI_BIN_CLASS, uci_base / 'bin-class-data'),
                   (TaskSource.UCI_MULTI_CLASS, uci_base / 'multi-class-data'),
                   (TaskSource.UCI_REGRESSION, uci_base / 'regression-data')]

    for src, path in uci_matches:
        print('Processing task source', src)
        ds_names = [file.stem for file in path.iterdir() if file.is_file()]
        ds_names.sort()
        task_type = TaskType.CLASSIFICATION if 'class' in src else TaskType.REGRESSION
        for ds_name in ds_names:
            task_desc = TaskDescription(task_source=src, task_name=ds_name)
            if (not rerun) and task_desc.exists_task(paths):
                continue
            print('Processing dataset', ds_name)
            ds_path = path / (ds_name + '.csv')

            import_from_csv(ds_path=ds_path, task_type=task_type, task_desc=task_desc, paths=paths,
                            remove_duplicates=remove_duplicates)
        TaskCollection.from_source(src, paths).save(paths)
        print()


def get_openml_task_ids(suite_id: Union[str, int]) -> List[int]:
    import openml
    suite = openml.study.get_suite(suite_id)
    return suite.tasks


class PandasTask:
    def __init__(self, x_df, y_df, cat_indicator: List[bool], task_type: str, more_info: Dict):
        if len(x_df.columns) != len(cat_indicator):
            raise ValueError('x.shape[1] != len(category_indicator)')

        self.x_df = x_df  # should be (sparse) pd.DataFrame
        # should be (sparse) pd.Series  (i.e. a single column of a DataFrame)
        self.y_df = y_df if task_type == TaskType.REGRESSION else y_df.astype('category')
        if pd.api.types.is_sparse(self.y_df):
            self.y_df = self.y_df.sparse.to_dense()

        # this is a fix because category_indicator[0] was False for the dataset MIP-2016-regression
        # despite the column being categorical (dtype=object)
        self.cat_indicator = [v or not pd.api.types.is_numeric_dtype(x_df[x_df.columns[i]])
                          for i, v in enumerate(cat_indicator)]
        self.cont_indicator = [not b for b in self.cat_indicator]
        self.task_type = task_type
        self.more_info_dict = more_info  # could be passed along to TaskInfo

    def get_n_classes(self):
        if self.task_type == TaskType.REGRESSION:
            return 0
        else:
            self.y_df = self.y_df.cat.remove_unused_categories()
            return len(self.y_df.cat.categories)

    def get_n_samples(self):
        return len(self.x_df)

    def deduplicate(self):
        is_duplicated = self.x_df.duplicated()
        if is_duplicated.any():
            print(f'Warning: Data set contains {is_duplicated.sum()} duplicate values! Removing duplicates...')
            self.x_df = self.x_df.loc[~is_duplicated]
            self.y_df = self.y_df[~is_duplicated]

    def limit_n_classes(self, max_n_classes: int):
        n_classes = self.get_n_classes()
        if n_classes <= max_n_classes:
            return

        vc = self.y_df.value_counts()
        # use mergesort to make it more deterministic
        perm = np.argsort(vc, kind='mergesort')
        cats = vc.axes[0]
        largest_classes = [cats[i] for i in perm[-max_n_classes:]]
        other_classes = [cats[i] for i in perm[:-max_n_classes]]
        to_keep = self.y_df.isin(largest_classes)
        self.x_df = self.x_df.loc[to_keep, :]
        self.y_df = self.y_df[to_keep]
        self.y_df = self.y_df.cat.remove_categories(other_classes)

    def subsample(self, max_size: int):
        if self.x_df.shape[0] > max_size:
            gen = np.random.default_rng(seed=0)
            perm = gen.permutation(self.x_df.shape[0])
            idxs = perm[:max_size]
            self.x_df = self.x_df.iloc[idxs]
            self.y_df = self.y_df.iloc[idxs]

    def remove_missing_cont(self):
        if not np.any(self.cont_indicator):
            return  # no continuous columns

        not_nan_rows = self.x_df.loc[:, self.cont_indicator].notna().all(axis=1)
        self.x_df = self.x_df.loc[not_nan_rows, :]
        self.y_df = self.y_df[not_nan_rows]

    def normalize_regression_y(self):
        if self.task_type == TaskType.REGRESSION and len(self.y_df) >= 2:
            y_np = np.asarray(self.y_df)
            self.y_df.loc[:] = (y_np - np.mean(y_np)) / (np.std(y_np) + 1e-30)

    def get_task(self, task_desc: TaskDescription) -> Task:
        x_cont = np.array(self.x_df.loc[:, self.cont_indicator], dtype=np.float32)

        x_cat_columns = []
        cat_sizes = []
        for i, is_cat in enumerate(self.cat_indicator):
            if is_cat:
                col = self.x_df[self.x_df.columns[i]].astype('category')
                col = col.cat.remove_unused_categories()
                # detect missing values
                col = col.cat.remove_categories([s for s in ['', '?'] if s in col.cat.categories])
                # don't use asarray to make sure that the array is not read-only
                col = np.array(col.cat.codes, dtype=np.int32)
                col += 1  # category 0 is used for missing value
                x_cat_columns.append(col)
                cat_sizes.append(1 + np.max(col))

        if len(x_cat_columns) > 0:
            x_cat = np.stack(x_cat_columns, axis=1)
        else:
            x_cat = np.zeros(shape=(len(self.x_df), 0), dtype=np.int32)

        if self.task_type == TaskType.CLASSIFICATION:
            self.y_df = self.y_df.cat.remove_unused_categories()
            y = np.array(self.y_df.cat.codes, dtype=np.int32)
            # y, n_classes = convert_to_class_numbers(y)
        else:
            y = np.array(self.y_df, dtype=np.float32)

        ds = DictDataset({'x_cont': torch.as_tensor(x_cont), 'x_cat': torch.as_tensor(x_cat),
                          'y': torch.as_tensor(y[:, None])},
                         {'x_cont': TensorInfo(feat_shape=[x_cont.shape[-1]]),
                          'x_cat': TensorInfo(cat_sizes=cat_sizes),
                          'y': TensorInfo(cat_sizes=[self.get_n_classes()])})
        task_info = TaskInfo.from_ds(task_desc, ds, more_info_dict=self.more_info_dict)
        return Task(task_info, ds)

    @staticmethod
    def from_openml_task_id(task_id: int):
        import openml
        task = openml.tasks.get_task(task_id, download_data=False)
        dataset = openml.datasets.get_dataset(task.dataset_id, download_data=False)
        x_df, y_df, cat_indicator, names = dataset.get_data(target=task.target_name, dataset_format='dataframe')

        if task.task_type_id == openml.tasks.TaskType.SUPERVISED_CLASSIFICATION:
            task_type = TaskType.CLASSIFICATION
        elif task.task_type_id == openml.tasks.TaskType.SUPERVISED_REGRESSION:
            task_type = TaskType.REGRESSION
        else:
            raise RuntimeError(f'Unknown OpenML Task Type: {task.task_type}')

        more_info_dict = dict(openml_task_id=task_id, openml_dataset_id=task.dataset_id)

        return PandasTask(x_df, y_df, cat_indicator, task_type, more_info=more_info_dict)


def set_openml_cache_dir(dir_name: Union[str, Path]):
    import openml
    if 'set_root_cache_directory' in dir(openml.config):
        # newer openml versions
        openml.config.set_root_cache_directory(str(dir_name))
    elif 'set_cache_directory' in dir(openml.config):
        # older openml versions
        openml.config.set_cache_directory(str(dir_name))


def get_openml_ds_names(task_ids: List[int]):
    import openml
    names = []
    for i, task_id in enumerate(task_ids):
        task = openml.tasks.get_task(task_id, download_data=False)
        dataset = openml.datasets.get_dataset(task.dataset_id, download_data=False)
        names.append(dataset.name)

    return names


def import_openml(task_ids: List[int], task_source_name: str, paths: Paths, cache_dir: Union[str, Path] = None,
                  normalize_y: bool = False, min_n_samples: int = 1, max_n_classes: int = 100000,
                  min_n_classes: int = 0, remove_missing_cont: bool = True, remove_duplicates: bool = False,
                  exclude_ds_names: Optional[List[str]] = None, max_n_samples: Optional[int] = None,
                  include_only_ds_names: Optional[List[str]] = None, rerun: bool = False,
                  ignore_above_n_classes: int = 100000):
    print(f'Processing task source {task_source_name}')
    import openml

    for i, task_id in enumerate(task_ids):
        with paths.new_tmp_folder() as tmp_folder:
            set_openml_cache_dir(cache_dir or tmp_folder)
            task = openml.tasks.get_task(task_id, download_data=False)
            dataset = openml.datasets.get_dataset(task.dataset_id, download_data=False)
            print(f'Processing task {dataset.name} for OpenML task source {task_source_name} [{i+1}/{len(task_ids)}]')
            if dataset.name in (exclude_ds_names or []) or \
                    (include_only_ds_names is not None and dataset.name not in include_only_ds_names):
                print('Task was manually excluded')
                continue
            task_desc = TaskDescription(task_source_name, dataset.name)
            if (not rerun) and task_desc.exists_task(paths):
                continue

            pd_task = PandasTask.from_openml_task_id(task_id)
            if remove_missing_cont:
                pd_task.remove_missing_cont()
            if remove_duplicates:
                pd_task.deduplicate()
            if max_n_samples is not None:
                pd_task.subsample(max_n_samples)
            if normalize_y:
                pd_task.normalize_regression_y()
            if pd_task.get_n_classes() > ignore_above_n_classes:
                print(f'Ignoring task with {pd_task.get_n_classes()} > {ignore_above_n_classes} classes')
                continue
            if pd_task.get_n_classes() > max_n_classes:
                print(f'Only keeping the most frequent {max_n_classes} out of {pd_task.get_n_classes()} classes')
                pd_task.limit_n_classes(max_n_classes)
            if pd_task.get_n_samples() < min_n_samples:
                print(f'Too few samples ({pd_task.get_n_samples()} < {min_n_samples}), ignoring task')
                continue
            if pd_task.get_n_classes() < min_n_classes:
                print(f'Too few classes, ignoring task')
                continue
            pd_task.get_task(task_desc).save(paths)

    TaskCollection.from_source(task_source_name, paths).save(paths)
    print(f'Finished importing OpenML tasks {task_source_name}')
    print()


if __name__ == '__main__':
    # import time
    # paths = Paths.from_env_variables()
    # start_time = time.time()
    # with paths.new_tmp_folder() as tmp_folder:
    #     pass
    # print(f'Time: {time.time() - start_time:g} s')
    task_ids = get_openml_task_ids(271)
    import_openml(task_ids[1:2], 'test', Paths('test'))
    pass

