from typing import Optional

import fire
import openml

from pytabkit.bench.data.import_tasks import set_openml_cache_dir, PandasTask
from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskCollection


def check_missing_values(openml_cache_dir: Optional[str] = None):
    paths = Paths.from_env_variables()
    for coll_name in ['meta-test-class', 'meta-test-reg']:
        task_infos = TaskCollection.from_name(coll_name, paths).load_infos(paths)
        # task_infos = [task_info for task_info in task_infos if task_info.n_samples < 5000]
        task_infos_no_missing_numeric = []
        task_infos_no_missing = []
        for task_info in task_infos:
            openml_task_id = task_info.more_info_dict['openml_task_id']
            with paths.new_tmp_folder() as tmp_folder:
                set_openml_cache_dir(openml_cache_dir or tmp_folder)
                task = openml.tasks.get_task(openml_task_id, download_data=False)
                dataset = openml.datasets.get_dataset(task.dataset_id, download_data=False)
                print(f'Analyzing {dataset.name}:')
                pd_task = PandasTask.from_openml_task_id(openml_task_id)
                has_column_nan = pd_task.x_df.isna().any()
                has_numeric_nan = has_column_nan[pd_task.cont_indicator].any(axis=None)
                has_categorical_nan = has_column_nan[pd_task.cat_indicator].any(axis=None)
                print(f'{has_numeric_nan=}, {has_categorical_nan=}')
                if not has_numeric_nan:
                    task_infos_no_missing_numeric.append(task_info)
                    if not has_categorical_nan:
                        task_infos_no_missing.append(task_info)
                # task = openml.tasks.get_task(openml_task_id, download_data=False)
                # dataset = openml.datasets.get_dataset(task.dataset_id, download_data=False)
                # x_df, y_df, cat_indicator, names = dataset.get_data(target=task.target_name, dataset_format='dataframe')
                # has_column_nan = x_df.isna().any()

        TaskCollection(coll_name + '-no-missing-numeric',
                       [task_info.task_desc for task_info in task_infos_no_missing_numeric]).save(paths)
        TaskCollection(coll_name + '-no-missing',
                       [task_info.task_desc for task_info in task_infos_no_missing]).save(paths)


if __name__ == '__main__':
    fire.Fire(check_missing_values)
    pass
