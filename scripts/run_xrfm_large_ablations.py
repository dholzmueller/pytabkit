import fire

from pytabkit.bench.alg_wrappers.interface_wrappers import RandomParamsxRFMInterfaceWrapper
from pytabkit.bench.run.task_execution import RunConfig, TabBenchJobManager, run_alg_selection

from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskCollection
from pytabkit.bench.scheduling.execution import RayJobManager
from pytabkit.bench.scheduling.schedulers import SimpleJobScheduler
from pytabkit.models.data.data import TaskType


def run_xrfm_large_ablations(hpo_space_name: str, n_hpo_steps: int = 30, rerun: bool = False):
    # todo: install xrfm directly from the repo
    # todo: set env variable for the tab_bench_data_path
    # todo: measure runtime
    # todo: ensure that only one job runs per GPU, so that the time measurements are accurate
    # todo: make sure to install the version with kermac
    paths = Paths.from_env_variables()
    task_infos = TaskCollection.from_name('meta-test-class', paths).load_infos(paths)
    task_infos.extend(TaskCollection.from_name('meta-test-reg', paths).load_infos(paths))
    task_infos = [ti for ti in task_infos if 70_000 <= ti.n_samples]
    class_task_infos = [ti for ti in task_infos if ti.task_type == TaskType.CLASSIFICATION]
    reg_task_infos = [ti for ti in task_infos if ti.task_type == TaskType.REGRESSION]
    TaskCollection('meta-test-large-class', [info.task_desc for info in class_task_infos]).save(paths)
    TaskCollection('meta-test-large-reg', [info.task_desc for info in reg_task_infos]).save(paths)
    for name, infos in [('class', class_task_infos), ('reg', reg_task_infos)]:
        print(f'{name} task infos:')
        for info in infos:
            print(f'{info.task_desc}: n_samples={info.n_samples}')
        print()

    config = RunConfig(n_tt_splits=1, n_cv=1, n_refit=0, save_y_pred=False)

    job_mgr = TabBenchJobManager(paths)
    scheduler = SimpleJobScheduler(RayJobManager())
    for step_idx in range(n_hpo_steps):
        job_mgr.add_jobs(task_infos, config,
                         f'xRFM-HPO-{hpo_space_name}_new_step-{step_idx}',
                         RandomParamsxRFMInterfaceWrapper(model_idx=step_idx, hpo_space_name=hpo_space_name,
                                                          M_batch_size=8192, max_leaf_size=40_000),
                         tags=[f'xrfm_hpo_{hpo_space_name}_new_steps'], rerun=rerun)

    job_mgr.run_jobs(scheduler)

    alg_names = [f'xRFM-HPO-{hpo_space_name}_new_step-{i}' for i in range(n_hpo_steps)]

    run_alg_selection(paths, config, class_task_infos,
                      f'xRFM-HPO-{hpo_space_name}_new', alg_names, val_metric_name='class_error',
                      tags=[f'xrfm_hpo_{hpo_space_name}', 'xrfm_hpo', 'default'], rerun=True)
    run_alg_selection(paths, config, reg_task_infos,
                      f'xRFM-HPO-{hpo_space_name}_new', alg_names, val_metric_name='rmse',
                      tags=[f'xrfm_hpo_{hpo_space_name}', 'xrfm_hpo', 'default'], rerun=True)


def run_xrfm_large_ablations_old(hpo_space_name: str = 'paper-large-pca', n_hpo_steps: int = 30, rerun: bool = False):
    # todo: install xrfm directly from the repo
    # todo: set env variable for the tab_bench_data_path
    # todo: measure runtime
    # todo: ensure that only one job runs per GPU, so that the time measurements are accurate
    # todo: make sure to install the version with kermac
    paths = Paths.from_env_variables()
    task_infos = TaskCollection.from_name('meta-test-class', paths).load_infos(paths)
    task_infos.extend(TaskCollection.from_name('meta-test-reg', paths).load_infos(paths))
    task_infos = [ti for ti in task_infos if 70_000 <= ti.n_samples <= 200_000]
    class_task_infos = [ti for ti in task_infos if ti.task_type == TaskType.CLASSIFICATION]
    reg_task_infos = [ti for ti in task_infos if ti.task_type == TaskType.REGRESSION]
    TaskCollection('meta-test-medlarge-class', [info.task_desc for info in class_task_infos]).save(paths)
    TaskCollection('meta-test-medlarge-reg', [info.task_desc for info in reg_task_infos]).save(paths)
    for name, infos in [('class', class_task_infos), ('reg', reg_task_infos)]:
        print(f'{name} task infos:')
        for info in infos:
            print(f'{info.task_desc}: n_samples={info.n_samples}')
        print()

    config = RunConfig(n_tt_splits=1, n_cv=1, n_refit=0, save_y_pred=False)

    job_mgr = TabBenchJobManager(paths)
    scheduler = SimpleJobScheduler(RayJobManager())
    for step_idx in range(n_hpo_steps):
        job_mgr.add_jobs(task_infos, config,
                         f'xRFM-HPO-{hpo_space_name}_Mbs-8192_step-{step_idx}',
                         RandomParamsxRFMInterfaceWrapper(model_idx=step_idx, hpo_space_name=hpo_space_name,
                                                          M_batch_size=8192),
                         tags=[f'xrfm_hpo_{hpo_space_name}_steps'], rerun=rerun)

    job_mgr.run_jobs(scheduler)

    alg_names = [f'xRFM-HPO-{hpo_space_name}_Mbs-8192_step-{i}' for i in range(n_hpo_steps)]

    run_alg_selection(paths, config, class_task_infos,
                      f'xRFM-HPO-{hpo_space_name}_Mbs-8192', alg_names, val_metric_name='class_error',
                      tags=[f'xrfm_hpo_{hpo_space_name}', 'xrfm_hpo', 'default'], rerun=True)
    run_alg_selection(paths, config, reg_task_infos,
                      f'xRFM-HPO-{hpo_space_name}_Mbs-8192', alg_names, val_metric_name='rmse',
                      tags=[f'xrfm_hpo_{hpo_space_name}', 'xrfm_hpo', 'default'], rerun=True)


def run_xrfm_small_test_ablations(n_hpo_steps: int = 50, rerun: bool = False):
    # todo: install xrfm directly from the repo
    # todo: set env variable
    # todo: measure runtime
    # todo: ensure that only one job runs per GPU, so that the time measurements are accurate
    paths = Paths.from_env_variables()
    task_infos = TaskCollection.from_name('meta-test-class', paths).load_infos(paths)
    task_infos.extend(TaskCollection.from_name('meta-test-reg', paths).load_infos(paths))
    task_infos = [ti for ti in task_infos if 100 <= ti.n_samples <= 2000]
    class_task_infos = [ti for ti in task_infos if ti.task_type == TaskType.CLASSIFICATION]
    reg_task_infos = [ti for ti in task_infos if ti.task_type == TaskType.REGRESSION]
    TaskCollection('meta-test-small-class', [info.task_desc for info in class_task_infos]).save(paths)
    TaskCollection('meta-test-small-reg', [info.task_desc for info in reg_task_infos]).save(paths)
    for name, infos in [('class', class_task_infos), ('reg', reg_task_infos)]:
        print(f'{name} task infos:')
        for info in infos:
            print(f'{info.task_desc}: n_samples={info.n_samples}')
        print()

    config = RunConfig(n_tt_splits=10, n_cv=1, n_refit=0, save_y_pred=False)

    job_mgr = TabBenchJobManager(paths)
    scheduler = SimpleJobScheduler(RayJobManager())
    hpo_space_name = 'paper-large-pca'
    for step_idx in range(n_hpo_steps):
        job_mgr.add_jobs(task_infos, config,
                         f'xRFM-HPO-{hpo_space_name}_small_step-{step_idx}',
                         RandomParamsxRFMInterfaceWrapper(model_idx=step_idx, hpo_space_name=hpo_space_name,
                                                          max_leaf_size=200),
                         tags=[f'xrfm_hpo_{hpo_space_name}_steps'], rerun=rerun)

    job_mgr.run_jobs(scheduler)

    alg_names = [f'xRFM-HPO-{hpo_space_name}_small_step-{i}' for i in range(n_hpo_steps)]

    run_alg_selection(paths, config, class_task_infos,
                      f'xRFM-HPO-{hpo_space_name}_small', alg_names, val_metric_name='class_error',
                      tags=[f'xrfm_hpo_{hpo_space_name}', 'xrfm_hpo', 'default'], rerun=True)
    run_alg_selection(paths, config, reg_task_infos,
                      f'xRFM-HPO-{hpo_space_name}_small', alg_names, val_metric_name='rmse',
                      tags=[f'xrfm_hpo_{hpo_space_name}', 'xrfm_hpo', 'default'], rerun=True)


if __name__ == '__main__':
    # run_xrfm_small_test_ablations()
    fire.Fire(run_xrfm_large_ablations)
    # run_xrfm_large_ablations(hpo_space_name='paper-large-pca')
    # run_xrfm_large_ablations(hpo_space_name='paper-large')
