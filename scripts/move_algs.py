import shutil

import fire

from pytabkit.bench.data.paths import Paths
from pytabkit.models import utils


def move_algs(base_path_1: str, base_path_2: str, *alg_names):
    paths_1 = Paths(base_folder=base_path_1)
    paths_2 = Paths(base_folder=base_path_2)
    for alg_name in alg_names:
        print(f'Moving alg {alg_name}')

        assert isinstance(alg_name, str)
        assert utils.existsDir(base_path_1)
        assert utils.existsDir(base_path_2)
        assert not utils.existsDir(paths_2.algs() / alg_name)
        assert not utils.existsDir(paths_2.results() / alg_name)
        assert not utils.existsDir(paths_2.result_summaries() / alg_name)

        if utils.existsDir(paths_1.algs() / alg_name):
            shutil.move(paths_1.algs() / alg_name, paths_2.algs() / alg_name)
        if utils.existsDir(paths_1.results() / alg_name):
            shutil.move(paths_1.results() / alg_name, paths_2.results() / alg_name)
        if utils.existsDir(paths_1.result_summaries() / alg_name):
            shutil.move(paths_1.result_summaries() / alg_name, paths_2.result_summaries() / alg_name)


def move_specific_algs(base_path_1: str, base_path_2: str):
    paths_1 = Paths(base_folder=base_path_1)
    alg_names = []
    for path in paths_1.algs().iterdir():
        name = path.name
        if name.startswith('MLP-cumul-abl-') and not name.startswith('MLP-cumul-abl-new'):
            alg_names.append(name)
    # print(alg_names)
    move_algs(base_path_1, base_path_2, *alg_names)


if __name__ == '__main__':
    fire.Fire(move_algs)
    # fire.Fire(move_specific_algs)



