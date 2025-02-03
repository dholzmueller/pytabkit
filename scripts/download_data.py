from typing import Optional

import fire

from pytabkit.bench.data.common import TaskSource
from pytabkit.bench.data.get_uci import download_all_uci
from pytabkit.bench.data.import_talent_benchmark import import_talent_benchmark
from pytabkit.bench.data.import_tasks import import_uci_tasks, get_openml_task_ids, import_openml, get_openml_ds_names
from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskCollection, TaskDescription, TaskInfo


def run_import(openml_cache_dir: str = None, import_meta_train: bool = False, import_meta_test: bool = False,
               import_openml_class_bin_extra: bool = False,
               import_grinsztajn: bool = False, import_grinsztajn_medium: bool = False,
               import_tabzilla_hard: bool = False, import_automl_class_small: bool = False,
               import_talent_class_small: bool = False, import_talent_reg_small: bool = False,
               talent_folder: Optional[str] = None):
    paths = Paths.from_env_variables()
    min_n_samples = 1000

    if import_meta_train:
        print(f'Importing meta-train')
        # import UCI
        download_all_uci(paths)
        import_uci_tasks(paths)

        # generate task collections
        uci_multi_class_descs = TaskCollection.from_source(TaskSource.UCI_MULTI_CLASS, paths).task_descs
        uci_bin_class_descs = TaskCollection.from_source(TaskSource.UCI_BIN_CLASS, paths).task_descs
        uci_multi_class_task_names = [td.task_name for td in uci_multi_class_descs]
        uci_class_descs = uci_multi_class_descs + [td for td in uci_bin_class_descs
                                                   if td.task_name not in uci_multi_class_task_names]
        uci_class_descs = [td for td in uci_class_descs if td.load_info(paths).n_samples >= min_n_samples]
        TaskCollection('meta-train-class', uci_class_descs).save(paths)

        uci_reg_descs = TaskCollection.from_source(TaskSource.UCI_REGRESSION, paths).task_descs
        uci_reg_descs = [td for td in uci_reg_descs if td.load_info(paths).n_samples >= min_n_samples]
        TaskCollection('meta-train-reg', uci_reg_descs).save(paths)

    # maybe could use faster pyarrow backend for pandas if v2 is available?
    # pd.options.mode.dtype_backend = "pyarrow"

    if import_meta_test or import_openml_class_bin_extra or import_automl_class_small:
        # import AutoML Benchmark and CTR-23 benchmark
        # could also import the TabZilla suite
        # https://www.openml.org/search?type=study&study_type=task&id=379&sort=tasks_included
        # but the selection criteria for this one are based a lot on the performance of different algorithms
        automl_class_task_ids = get_openml_task_ids(271)
        automl_reg_task_ids = get_openml_task_ids(269)
        ctr23_reg_task_ids = get_openml_task_ids(353)
        sarcos_duplicated_task_id = 361254
        sarcos_deduplicated_task_id = 361011
        if sarcos_duplicated_task_id in ctr23_reg_task_ids:
            # use the version of sarcos without the duplicated test set
            print(f'Using a different version of the sarcos data set for the CTR-23 benchmark')
            ctr23_reg_task_ids.remove(sarcos_duplicated_task_id)
            ctr23_reg_task_ids.append(sarcos_deduplicated_task_id)
        all_reg_task_ids = list(set(automl_reg_task_ids + ctr23_reg_task_ids))  # todo
        automl_class_ds_names = get_openml_ds_names(automl_class_task_ids)
        automl_reg_ds_names = get_openml_ds_names(automl_reg_task_ids)
        ctr23_reg_ds_names = get_openml_ds_names(ctr23_reg_task_ids)

        def check_task(td: TaskDescription, min_n_samples: Optional[int] = None,
                       max_one_hot_size: Optional[int] = None) -> bool:
            task_info = td.load_info(paths)
            if min_n_samples is not None and task_info.n_samples < min_n_samples:
                print(f'Ignoring task {str(td)} because it has too few samples')
                return False
            n_cont = task_info.tensor_infos['x_cont'].get_n_features()
            cat_sizes = task_info.tensor_infos['x_cat'].get_cat_sizes().numpy()
            # ignore 'missing' categories
            # todo: is this really the way we should handle this?
            d_one_hot = n_cont + sum([1 if cs == 3 else cs - 1 for cs in cat_sizes])
            if max_one_hot_size is not None and d_one_hot > max_one_hot_size:
                print(f'Ignoring task {str(td)} because it is too high-dimensional after one-hot encoding')
                return False
            return True

        if import_meta_test:
            print(f'Importing meta-test')
            # treat dionis separately because we want to subsample it to 100k instead of 500k samples for speed and RAM reasons
            automl_class_task_ids_not_dionis = [id for id, name in zip(automl_class_task_ids, automl_class_ds_names)
                                                if name != 'dionis']
            automl_class_task_ids_dionis = [id for id, name in zip(automl_class_task_ids, automl_class_ds_names)
                                            if name == 'dionis']
            assert len(automl_class_task_ids_dionis) == 1
            assert len(automl_class_task_ids_not_dionis) == len(automl_class_task_ids) - 1

            import_openml(automl_class_task_ids_not_dionis, TaskSource.OPENML_CLASS, paths, openml_cache_dir,
                          max_n_samples=500_000, rerun=False)
            import_openml(automl_class_task_ids_dionis, TaskSource.OPENML_CLASS, paths, openml_cache_dir,
                          max_n_samples=100_000, rerun=True)

            import_openml(all_reg_task_ids, TaskSource.OPENML_REGRESSION, paths, openml_cache_dir, normalize_y=True,
                          max_n_samples=500000, rerun=False)

            class_descs = TaskCollection.from_source(TaskSource.OPENML_CLASS, paths).task_descs

            # generate task collections
            exclude_automl_class = ['kr-vs-kp', 'wilt', 'ozone-level-8hr', 'first-order-theorem-proving',
                                    'GesturePhaseSegmentationProcessed', 'PhishingWebsites', 'wine-quality-white',
                                    'nomao',
                                    'bank-marketing', 'adult']
            filtered_class_descs = [td for td in class_descs if td.task_name not in exclude_automl_class
                                    and td.task_name in automl_class_ds_names
                                    and check_task(td, min_n_samples=min_n_samples, max_one_hot_size=10000)]
            TaskCollection('meta-test-class', filtered_class_descs).save(paths)

            # we exclude Brazilian_houses because there is already brazilian_houses in ctr-23,
            # and Brazilian_houses includes three features that should not be used for predicting the target,
            # while brazilian_houses should not contain them
            exclude_automl_reg = ['wine_quality', 'abalone', 'OnlineNewsPopularity', 'Brazilian_houses']
            exclude_ctr23_reg = ['abalone', 'physiochemical_protein', 'naval_propulsion_plant', 'superconductivity',
                                 'white_wine', 'red_wine', 'grid_stability']
            reg_descs = TaskCollection.from_source(TaskSource.OPENML_REGRESSION, paths).task_descs
            filtered_reg_descs = [td for td in reg_descs if td.task_name not in exclude_automl_reg + exclude_ctr23_reg
                                  and td.task_name in automl_reg_ds_names + ctr23_reg_ds_names
                                  and check_task(td, min_n_samples=min_n_samples, max_one_hot_size=10000)]
            TaskCollection('meta-test-reg', filtered_reg_descs).save(paths)

        if import_openml_class_bin_extra:
            print(f'Importing openml-class-bin-extra')
            # also import binary version of multiclass tasks
            # requires that meta_test has already been imported
            class_descs = TaskCollection.from_source(TaskSource.OPENML_CLASS, paths).task_descs
            multiclass_names = [td.task_name for td in class_descs if td.load_info(paths).get_n_classes() > 2]
            # print(f'{multiclass_names=}')
            import_openml(automl_class_task_ids, TaskSource.OPENML_CLASS_BIN_EXTRA, paths, openml_cache_dir,
                          max_n_classes=2, include_only_ds_names=multiclass_names)

        if import_automl_class_small:
            print(f'Importing automl-class-small')
            import_openml(automl_class_task_ids, TaskSource.AUTOML_CLASS_SMALL, paths, openml_cache_dir,
                          ignore_above_n_classes=50, min_n_samples=1000, max_n_samples=100_000)
            descs = TaskCollection.from_source(TaskSource.AUTOML_CLASS_SMALL, paths).task_descs
            filtered_descs = [td for td in descs if check_task(td, max_one_hot_size=1000)]
            TaskCollection('automl-class-small-filtered', filtered_descs).save(paths)

    if import_grinsztajn:
        print(f'Importing grinsztain benchmark')
        import_grinsztajn_datasets(openml_cache_dir)

    if import_grinsztajn_medium:
        print(f'Importing grinsztain medium benchmark')
        import_grinsztajn_medium_datasets(openml_cache_dir)

    if import_tabzilla_hard:
        print(f'Importing TabZilla hard benchmark')
        import_tabzilla_hard_datasets(openml_cache_dir)

    if import_talent_class_small:
        if talent_folder is None:
            raise ValueError(f'Please specify talent_folder to import datasets from the TALENT benchmark')
        import_talent_benchmark(paths, talent_folder=talent_folder, source_name='talent-class-small',
                                allow_regression=False,
                                min_n_samples=1000, max_n_samples=100_000, ignore_above_n_classes=100)
        task_infos = TaskCollection.from_source('talent-class-small', paths).load_infos(paths)
        bin_task_descs = [ti.task_desc for ti in task_infos if ti.get_n_classes() == 2]
        multi_task_descs = [ti.task_desc for ti in task_infos if ti.get_n_classes() != 2]
        TaskCollection('talent-bin-class-small', bin_task_descs).save(paths)
        TaskCollection('talent-multi-class-small', multi_task_descs).save(paths)
        above10k_descs = [ti.task_desc for ti in task_infos if ti.n_samples >= 10_000]
        below10k_descs = [ti.task_desc for ti in task_infos if ti.n_samples < 10_000]
        TaskCollection('talent-class-small-above10k', above10k_descs).save(paths)
        TaskCollection('talent-class-small-below10k', below10k_descs).save(paths)

        talent_reg_tabpfn_task_descs = [ti.task_desc for ti in task_infos if
                                          ti.get_n_classes() <= 10 and ti.n_samples <= 10_000 and ti.tensor_infos[
                                              'x_cont'].get_n_features() + ti.tensor_infos[
                                              'x_cat'].get_n_features() <= 500]

        TaskCollection('talent-class-tabpfn', talent_reg_tabpfn_task_descs).save(paths)

    if import_talent_reg_small:
        if talent_folder is None:
            raise ValueError(f'Please specify talent_folder to import datasets from the TALENT benchmark')
        import_talent_benchmark(paths, talent_folder=talent_folder, source_name='talent-reg-small',
                                allow_regression=True, allow_classification=False,
                                min_n_samples=1000, max_n_samples=100_000)

        task_infos = TaskCollection.from_source('talent-reg-small', paths).load_infos(paths)
        talent_reg_tabpfn_task_descs = [ti.task_desc for ti in task_infos if
                                          ti.n_samples <= 10_000 and ti.tensor_infos[
                                              'x_cont'].get_n_features() + ti.tensor_infos[
                                              'x_cat'].get_n_features() <= 500]

        TaskCollection('talent-reg-tabpfn', talent_reg_tabpfn_task_descs).save(paths)


def import_grinsztajn_datasets(openml_cache_dir: str = None):
    # import data sets from the benchmark of Grinsztajn et al.
    paths = Paths.from_env_variables()
    import_openml(get_openml_task_ids(334), 'grinsztajn-cat-class', paths, openml_cache_dir,
                  max_n_samples=500000,
                  rerun=False)
    import_openml(get_openml_task_ids(335), 'grinsztajn-cat-reg', paths, openml_cache_dir,
                  normalize_y=True, max_n_samples=500000,
                  rerun=False)
    import_openml(get_openml_task_ids(336), 'grinsztajn-num-reg', paths, openml_cache_dir,
                  normalize_y=True, max_n_samples=500000,
                  rerun=False)
    import_openml(get_openml_task_ids(337), 'grinsztajn-num-class', paths, openml_cache_dir,
                  max_n_samples=500000,
                  rerun=False)

    import_openml(get_openml_task_ids(334), 'grinsztajn-cat-class-15k', paths, openml_cache_dir,
                  max_n_samples=15_000,
                  rerun=False)
    import_openml(get_openml_task_ids(335), 'grinsztajn-cat-reg-15k', paths, openml_cache_dir,
                  normalize_y=True, max_n_samples=15_000,
                  rerun=False)
    import_openml(get_openml_task_ids(336), 'grinsztajn-num-reg-15k', paths, openml_cache_dir,
                  normalize_y=True, max_n_samples=15_000,
                  rerun=False)
    import_openml(get_openml_task_ids(337), 'grinsztajn-num-class-15k', paths, openml_cache_dir,
                  max_n_samples=15_000,
                  rerun=False)


def import_grinsztajn_medium_datasets(openml_cache_dir: str = None):
    paths = Paths.from_env_variables()
    for bench_name, bench_id_cat, bench_id_num in [('grinsztajn-class', 334, 337), ('grinsztajn-reg', 335, 336)]:
        task_ids_cat = get_openml_task_ids(bench_id_cat)
        task_ids_num = get_openml_task_ids(bench_id_num)
        task_ids = task_ids_cat + [task_id for task_id in task_ids_num if task_id not in task_ids_cat]
        import_openml(task_ids, bench_name, paths, openml_cache_dir,
                      max_n_samples=500_000,  # normalize_y=(bench_name=='grinsztajn-reg'),
                      rerun=False)
        task_infos = TaskCollection.from_source(bench_name, paths).load_infos(paths)
        for task_info in task_infos:
            # use 13333 so the 75%-25% train-val split will use 10k training samples
            task_info.max_n_trainval = 13_333
            task_info.save(paths)

    tc_orig = TaskCollection.from_source('grinsztajn-class', paths)
    tc_orig.save(paths)
    # exclude eye_movements because it has a leak according to the TabR paper
    tc = TaskCollection('grinsztajn-class-filtered',
                        [task_desc for task_desc in tc_orig.task_descs if task_desc.task_name != 'eye_movements'])
    tc.save(paths)


def import_tabzilla_hard_datasets(openml_cache_dir: str = None):
    # import data sets from the benchmark of Grinsztajn et al.
    paths = Paths.from_env_variables()
    import_openml(get_openml_task_ids(379), 'tabzilla-hard-class', paths, openml_cache_dir,
                  rerun=False)


def split_meta_test(paths: Paths):
    for task_type in ['class', 'reg']:
        coll_name = f'meta-test-{task_type}'
        task_infos = TaskCollection.from_name(coll_name, paths).load_infos(paths)

        def is_ood(task_info: TaskInfo):
            if task_info.n_samples < 1500 or task_info.n_samples > 60000:
                return True
            n_features = (task_info.tensor_infos['x_cont'].get_n_features()
                          + task_info.tensor_infos['x_cat'].get_n_features())
            if n_features > 750:
                return True
            x_cat_info = task_info.tensor_infos['x_cat']
            if x_cat_info.get_n_features() > 0 and x_cat_info.get_cat_sizes().max().item() > 50:
                return True
            return False

        id_task_descs = [task_info.task_desc for task_info in task_infos if not is_ood(task_info)]
        ood_task_descs = [task_info.task_desc for task_info in task_infos if is_ood(task_info)]

        TaskCollection(f'{coll_name}-indist', id_task_descs).save(paths)
        TaskCollection(f'{coll_name}-oodist', ood_task_descs).save(paths)

        print(f'{len(id_task_descs)=}, {len(ood_task_descs)=}')


# could extend this for other task collections like openml-cc18, pmlb, uci121 or uci-small

if __name__ == '__main__':
    fire.Fire(run_import)
    # import_grinsztajn_datasets()
    # paths = Paths.from_env_variables()
    # split_meta_test(paths)

    # meta_train = TaskCollection.from_name('meta-train-class', paths).load_infos(paths)
    # only_bin_class = [info.task_desc for info in meta_train if info.get_n_classes() == 2]
    # only_multi_class = [info.task_desc for info in meta_train if info.get_n_classes() > 2]
    # TaskCollection('meta-train-bin-class', only_bin_class).save(paths)
    # TaskCollection('meta-train-multi-class', only_multi_class).save(paths)

    # print(get_openml_ds_names([361011]))
    # ctr23_reg_task_ids = get_openml_task_ids(353)
    # ctr23_reg_ds_names = get_openml_ds_names(ctr23_reg_task_ids)
    # for ds_name in ctr23_reg_ds_names:
    #     print(ds_name)

    # test brazilian houses data set
    # import openml
    # import pandas as pd
    # task = openml.tasks.get_task(361267, download_data=False)
    # dataset = openml.datasets.get_dataset(task.dataset_id, download_data=True)
    # df: pd.DataFrame = dataset.get_data()[0]
    # print(df.head())
    # print(dataset.dataset_id)

    # test sarcos dataset
    # import openml
    # task = openml.tasks.get_task(361011, download_data=False)
    # dataset = openml.datasets.get_dataset(task.dataset_id, download_data=False)
    # print(dataset.dataset_id)
