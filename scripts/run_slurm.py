import functools

import fire

from run_experiments import run_trees_custom, run_gbdts_hpo, run_gbdt_rs_configs
from pytabkit.bench.data.paths import Paths


if __name__ == '__main__':
    # paths = Paths.from_env_variables()
    # run_configs(paths)
    # run_trees_custom(paths, n_estimators=10, tag='gbdts_nest-10')
    # run_gbdts_hpo(paths, n_estimators=1000, tag='paper', early_stopping_rounds=300)
    fire.Fire(run_gbdt_rs_configs)
