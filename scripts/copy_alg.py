import shutil
from typing import List

from pytabkit.bench.data.paths import Paths


def copy_algs(paths_1: Paths, paths_2: Paths, alg_names: List[str]):
    for alg_name in alg_names:
        print(f'Copying alg {alg_name}')
        shutil.copytree(paths_1.algs() / alg_name, paths_2.algs() / alg_name)
        shutil.copytree(paths_1.results() / alg_name, paths_2.results() / alg_name)
        shutil.copytree(paths_1.result_summaries() / alg_name, paths_2.result_summaries() / alg_name)


if __name__ == '__main__':
    paths_1 = Paths('first_path')
    paths_2 = Paths('second_path')

    alg_names = [f'{method}-{version}'
                 for method in ['XGB', 'LGBM', 'CatBoost']
                 for version in ['D', 'TD-class', 'TD-reg', 'HPO']]
    alg_names.extend([an + suffix for an in ['MLP-RTDL-D', 'ResNet-RTDL-D', 'TabR-S-D'] for suffix in ['-class', '-reg']])
    alg_names.extend(['MLP-HPO', 'MLP-RTDL-HPO', 'RF-SKL-D', 'XGB-PBB-D'])

    copy_algs(paths_1, paths_2, alg_names)



