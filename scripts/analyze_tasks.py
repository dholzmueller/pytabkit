from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskCollection
from pytabkit.models import utils


def print_task_analysis(coll_name: str, paths: Paths):
    coll = TaskCollection.from_name(coll_name, paths)
    # coll.save(paths)
    task_infos = coll.load_infos(paths)
    print(f'Data sets in task collection {coll_name}:')
    str_table = [['Data set: ', 'n ', 'k ', 'd ', 'd_one_hot ', 'd_one_hot_leq_10 ', 'd_one_hot_target ', 'largest_cat']]
    for task_info in task_infos:
        name = task_info.task_desc.task_name
        n = task_info.n_samples
        # k = number of classes
        k = task_info.tensor_infos['y'].get_cat_sizes()[0].item()
        n_cont = task_info.tensor_infos['x_cont'].get_n_features()
        cat_sizes = task_info.tensor_infos['x_cat'].get_cat_sizes().numpy()
        d = n_cont + len(cat_sizes)
        # ignore 'missing' categories
        d_one_hot = n_cont + sum([1 if cs==3 else cs-1 for cs in cat_sizes])
        d_one_hot_leq_10 = n_cont + sum([(1 if cs==3 else cs-1) if cs <= 11 else 1 for cs in cat_sizes])
        n_target = 1 if k <= 2 else k
        d_one_hot_target = n_cont + sum([(1 if cs==3 else min(n_target, cs-1)) for cs in cat_sizes])
        largest_cat = 0
        if cat_sizes is not None and len(cat_sizes) > 0:
            largest_cat = int(np.max(task_info.tensor_infos['x_cat'].get_cat_sizes().numpy()))

        str_table.append([name + ' ', str(n) + ' ', str(k) + ' ', str(d.item()) + ' ', str(d_one_hot.item()) + ' ',
                          str(d_one_hot_leq_10.item()) + ' ', str(d_one_hot_target.item()) + ' ',
                          str(largest_cat) + ' '])
    print(utils.pretty_table_str(str_table))
    print()
    print(f'Number of tasks with more than 1000 samples: {len([ti for ti in task_infos if ti.n_samples >= 1000])}')
    print()
    print()


def plot_tasks(coll_name: str, paths: Paths):
    coll = TaskCollection.from_name(coll_name, paths)
    task_infos = coll.load_infos(paths)
    plt.figure(figsize=(5, 4))

    for task_info in task_infos:
        n_cont = task_info.tensor_infos['x_cont'].get_n_features()
        cat_sizes = task_info.tensor_infos['x_cat'].get_cat_sizes().numpy()
        d = n_cont + len(cat_sizes)
        n = task_info.n_samples
        plt.loglog(n, d, 'k.')

    plt.xlabel('Number of samples')
    plt.ylabel('Number of features')
    plt.tight_layout()
    filename = Path('../plots') / f'{coll_name}.pdf'
    utils.ensureDir(filename)
    plt.savefig(filename, bbox_inches='tight')


def plot_tasks_multi(coll_names: List[str], paths: Paths):
    plt.figure(figsize=(7, 5))

    for coll_name in coll_names:
        coll = TaskCollection.from_name(coll_name, paths)
        task_infos = coll.load_infos(paths)

        ds = []
        ns = []

        for task_info in task_infos:
            n_cont = task_info.tensor_infos['x_cont'].get_n_features()
            cat_sizes = task_info.tensor_infos['x_cat'].get_cat_sizes().numpy()
            d = n_cont + len(cat_sizes)
            n = task_info.n_samples
            ds.append(d)
            ns.append(n)

        plt.loglog(ns, ds, '.', label=coll_name)

    plt.legend()

    plt.xlabel('Number of samples')
    plt.ylabel('Number of features')
    plt.tight_layout()
    filename = Path('../plots') / f'data_set_characteristics.pdf'
    utils.ensureDir(filename)
    plt.savefig(filename, bbox_inches='tight')


if __name__ == '__main__':
    paths = Paths.from_env_variables()

    coll_names = ['meta-train-class', 'meta-train-reg', 'meta-test-class', 'meta-test-reg',
                  # 'grinsztajn-cat-class', 'grinsztajn-num-class', 'grinsztajn-cat-reg', 'grinsztajn-num-reg',
                  # 'grinsztajn-cat-class-15k', 'grinsztajn-num-class-15k', 'grinsztajn-cat-reg-15k',
                  # 'grinsztajn-num-reg-15k'
                  ]

    for coll_name in coll_names:
        print_task_analysis(coll_name, paths)
        plot_tasks(coll_name, paths)

    plot_tasks_multi(coll_names, paths)

    # print_task_analysis('cc18-bin-class', paths)
    # print_task_analysis('cc18-multi-class', paths)

