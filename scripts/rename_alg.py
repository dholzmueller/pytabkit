import os
import shutil
from pathlib import Path

import fire

from pytabkit.bench.data.paths import Paths
from pytabkit.models import utils


def rename_alg(old_name: str, new_name: str, copy: bool = False):
    # what to rename:
    # results folder
    # result_summaries folder
    # alg_name in algs/alg_name/extended_config.yaml and in the path
    # cannot realistically change the code in src/
    # maybe change alg_name in algs/alg_name/wrapper.pkl (if it can be loaded)
    paths = Paths.from_env_variables()

    if utils.existsDir(paths.algs() / new_name):
        raise ValueError(f'Directory for new name {new_name} already exists')

    def rename_or_copy(src: Path, dst: Path):
        if copy:
            shutil.copytree(src, dst)
        else:
            os.rename(src, dst)

    rename_or_copy(paths.algs() / old_name, paths.algs() / new_name)
    if utils.existsDir(paths.results() / old_name):
        rename_or_copy(paths.results() / old_name, paths.results() / new_name)
    if utils.existsDir(paths.result_summaries() / old_name):
        rename_or_copy(paths.result_summaries() / old_name, paths.result_summaries() / new_name)

    # change alg_name in extended_config.yaml
    extended_config_path = paths.algs() / new_name / 'extended_config.yaml'
    extended_config = utils.deserialize(extended_config_path, use_yaml=True)
    extended_config['alg_name'] = new_name
    utils.serialize(extended_config_path, extended_config, use_yaml=True)

    # try to change alg_name in wrapper.pkl
    try:
        alg_wrapper_path = paths.algs() / new_name / 'wrapper.pkl'
        alg_wrapper = utils.deserialize(alg_wrapper_path)
        alg_wrapper.config['alg_name'] = new_name
        utils.serialize(alg_wrapper_path, alg_wrapper)
    except Exception as e:
        print(f'Could not modify alg_wrapper.pkl, got an exception: {e}')


if __name__ == '__main__':
    fire.Fire(rename_alg)
