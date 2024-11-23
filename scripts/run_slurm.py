import functools

import fire

from run_experiments import run_gbdt_rs_configs
from pytabkit.bench.data.paths import Paths


if __name__ == '__main__':
    # paths = Paths.from_env_variables()
    # run_configs(paths)
    fire.Fire(run_gbdt_rs_configs)
