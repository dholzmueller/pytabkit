import fire

from pytabkit.bench.data.paths import Paths
from pytabkit.bench.eval.analysis import ResultsTables
from pytabkit.bench.eval.evaluation import DefaultEvalModeSelector


def print_complete_results(coll_name: str, n_splits: int = 10):
    """
    Only show alg_names for which results for all splits exist.
    :param coll_name:
    :param n_splits:
    :return:
    """
    paths = Paths.from_env_variables()
    tables = ResultsTables(paths)
    table = tables.get(coll_name)
    test_table = table.get_test_results_table(DefaultEvalModeSelector())
    test_table = test_table.filter_n_splits(n_splits)
    alg_names = test_table.alg_names
    alg_names.sort(key=lambda x: x.lower())
    print(f'Algorithms with {n_splits} splits available on all datasets of {coll_name}:')
    for alg_name in alg_names:
        print(alg_name)


if __name__ == '__main__':
    fire.Fire(print_complete_results)
