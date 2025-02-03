from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pytabkit.bench.data.import_tasks import PandasTask
from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskDescription, TaskCollection
from pytabkit.models import utils
from pytabkit.models.data.data import TaskType


def import_talent_benchmark(paths: Paths, talent_folder: str, source_name: str, allow_regression: bool = True,
                            allow_classification: bool = True,
                            normalize_y: bool = False, min_n_samples: int = 1, max_n_classes: int = 100000,
                            min_n_classes: int = 0, remove_missing_cont: bool = True, remove_duplicates: bool = False,
                            max_n_samples: Optional[int] = None, ignore_above_n_classes: int = 100000,
                            dry_run: bool = False):
    talent_folder = Path(talent_folder)
    dataset_folders = [dataset_folder for dataset_folder in talent_folder.iterdir()]
    for i, dataset_folder in enumerate(dataset_folders):
        dataset_name = dataset_folder.name

        info = utils.deserialize(dataset_folder / 'info.json', use_json=True)
        if dry_run:
            train_size = info.get("train_size", None)
            n_samples = info['train_size'] + info['val_size'] + info['test_size']
            if train_size >= 100_000:
                print(f'{dataset_name}: {train_size=}')
            if n_samples >= 100_000:
                print(f'{dataset_name}: {n_samples=}')
            continue

        print(f'Importing dataset {dataset_name} [{i + 1}/{len(dataset_folders)}]')
        # can be 'regression', 'multiclass', 'binclass'
        task_type = info['task_type']
        print(f'{task_type=}')
        assert task_type in ['regression', 'multiclass', 'binclass']

        if task_type == 'regression' and not allow_regression:
            print(f'Skipping regression datasets')
            continue
        elif task_type != 'regression' and not allow_classification:
            print(f'Skipping classification datasets')
            continue

        # can be 1 for regression
        n_classes = info.get('n_classes', info.get('num_classes', None))

        print(f'{n_classes=}')

        y = np.concatenate(
            [np.load(dataset_folder / f'y_{part}.npy', allow_pickle=True) for part in ['train', 'val', 'test']], axis=0)
        n_samples = y.shape[0]

        # print(f'{y[:5]=}')

        # print(f'{y.shape=}, {y.dtype=}')

        if len(y.shape) == 2 and y.shape[1] == 1:
            y = y[:, 0]

        y_df = pd.Series(y)

        if task_type == 'regression':
            y_df = y_df.astype(np.float32)
        else:
            y_df = y_df.astype('category')
            if np.any(y_df.isnull()):
                raise ValueError(f'Missing values in class labels not allowed')

        x_dfs = []

        if utils.existsFile(dataset_folder / 'N_train.npy'):
            N = np.concatenate(
                [np.load(dataset_folder / f'N_{part}.npy', allow_pickle=True) for part in ['train', 'val', 'test']],
                axis=0)
            # print(f'{N.shape=}, {N.dtype=}')
            df = pd.DataFrame(N, columns=[f'cont_{i}' for i in range(N.shape[1])]).astype(np.float32)
            # print(df.head())
            # print(f'{df.columns=}')
            x_dfs.append(df)
            # print(N.flatten()[0])
            # if np.any(np.isnan(N)):
            if np.any(df.isnull()):
                print(f'Contains missing numerical values! ##########################################')
        else:
            N = np.zeros(shape=(n_samples, 0), dtype=np.float32)

        if utils.existsFile(dataset_folder / 'C_train.npy'):
            C = np.concatenate(
                [np.load(dataset_folder / f'C_{part}.npy', allow_pickle=True) for part in ['train', 'val', 'test']],
                axis=0)
            # print(f'{C.shape=}, {C.dtype=}')
            df = pd.DataFrame(C, columns=[f'cat_{i}' for i in range(C.shape[1])]).astype('category')
            # print(f'{df.columns=}')
            x_dfs.append(df)
            if np.any(df.isnull()):
                print(f'Contains missing categorical values! ##########################################')
        else:
            C = np.zeros(shape=(n_samples, 0), dtype=np.int32)

        if len(x_dfs) == 1:
            x_df = x_dfs[0]
        elif len(x_dfs) == 2:
            x_df = pd.concat(x_dfs, axis='columns')
        else:
            raise ValueError(f'Expected len(x_dfs) in [1, 2], but got {len(x_dfs)=}')

        cat_columns = x_df.select_dtypes(include='category').columns.tolist()
        cat_indicator = [column in cat_columns for column in x_df.columns]

        task_type = TaskType.REGRESSION if task_type == 'regression' else TaskType.CLASSIFICATION

        # task_source_name = 'talent-reg' if task_type == TaskType.REGRESSION else 'talent-class'
        task_desc = TaskDescription(source_name, dataset_name)

        pd_task = PandasTask(x_df, y_df, cat_indicator, task_type, more_info=info)
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

    if not dry_run:
        TaskCollection.from_source(source_name, paths).save(paths)
