import copy
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.pyplot import arrow

from pytabkit.bench.eval.analysis import get_opt_groups, get_simplified_name, ResultsTables, \
    get_benchmark_results, get_display_name
from pytabkit.bench.eval.colors import more_percep_uniform_hue

matplotlib.use('agg')
# matplotlib.use('pdf')
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size': 10.95,
    'text.usetex': True,
    'pgf.rcfonts': False,
    # 'legend.framealpha': 0.5,
    'text.latex.preamble': r'\usepackage{times} \usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb} \usepackage{xcolor}'
})
from tueplots import bundles, fonts, fontsizes, figsizes

matplotlib.rcParams.update(bundles.icml2022())
matplotlib.rcParams.update(fonts.icml2022_tex())
matplotlib.rcParams.update(fontsizes.icml2022())

matplotlib.rcParams['text.latex.preamble'] = matplotlib.rcParams['text.latex.preamble'] + r'\usepackage{xcolor}'

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import patches as mpatches

import seaborn as sns
from adjustText import adjust_text
import matplotlib.patheffects as PathEffects
from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskCollection
from pytabkit.bench.eval.evaluation import MultiResultsTable, DefaultEvalModeSelector, FunctionAlgFilter, TaskWeighting
from pytabkit.bench.eval.runtimes import get_avg_train_times
from pytabkit.models import utils
from pytabkit.models.training.scheduling import get_schedule


# import distinctipy
# class CustomPalette:
#     default = distinctipy.get_colors(n_colors=14,
#                                      exclude_colors=[(a, b, c) for a in [1, 0.8] for b in [1, 0.8] for c in [1, 0.8]],
#                                      pastel_factor=0.5, rng=0)


def get_plot_color_idx(alg_name: str):
    parts = ['BestModel', 'Ensemble', 'MLP-RTDL', 'MLP-PLR', 'RealMLP', 'ResNet', 'FTT',
             ['TabR', 'RealTabR'],
             # 'SAINT',
             'XGB', 'LGBM', 'CatBoost',
             # 'GBT',
             'RF']

    # don't use prefixes and reverse to get better colors for BestModel_FTT-D_prep etc.
    for i, part_or_list in reversed(list(enumerate(parts))):
        lst = part_or_list if isinstance(part_or_list, list) else [part_or_list]
        for part in lst:
            if part in alg_name:
                return i
    raise ValueError(f'Unknown method: {alg_name}')


def gg_color_hue(n: int, saturation: float = 1.0, value: float = 0.65):
    # hues = np.linspace(13, 375, num=n + 1)[:-1]  # exclude the last element to avoid a duplicate of the first color
    # return [tuple(matplotlib.colors.hsv_to_rgb((h / 360.0, saturation, value)).tolist()) for h in hues]
    hues = np.linspace(0.0, 1.0, n + 1)[:-1]
    hues = [more_percep_uniform_hue(hue) for hue in hues]
    return [tuple(matplotlib.colors.hsv_to_rgb((h, saturation, value)).tolist()) for h in hues]


def get_plot_color(alg_name: str):
    idx = get_plot_color_idx(alg_name)
    special = ('rssc' in alg_name or 'TPE' in alg_name or 'no-ls' in alg_name)
    half_special = '_prep' in alg_name
    colors = gg_color_hue(12, saturation=0.6 if special else (0.8 if half_special else 1.0),
                          value=0.9 if special else (0.775 if half_special else 0.65))
    return colors[idx]


def coll_name_to_title(coll_name: str) -> str:
    if coll_name == 'meta-train-class':
        title = r'Meta-train classification benchmark'
    elif coll_name == 'meta-train-reg':
        title = r'Meta-train regression benchmark'
    elif coll_name == 'meta-test-class':
        title = r'Meta-test classification benchmark'
    elif coll_name == 'meta-test-reg':
        title = r'Meta-test regression benchmark'
    elif coll_name == 'meta-test-class-no-missing':
        title = r'$\mathcal{B}^{\mathrm{test}}_{\mathrm{class}}$ without missing value datasets'
    elif coll_name == 'meta-test-reg-no-missing':
        title = r'$\mathcal{B}^{\mathrm{test}}_{\mathrm{reg}}$ without missing value datasets'
    elif coll_name == 'grinsztajn-class-filtered':
        title = r'Grinsztajn et al.\ (2022) classification benchmark'
    elif coll_name == 'grinsztajn-reg':
        title = r'Grinsztajn et al.\ (2022) regression benchmark'
    else:
        title = coll_name
    title = r'\textbf{' + title + r'}'
    return title


def plot_schedule(paths: Paths, filename: str, sched_name: str) -> None:
    with plt.rc_context(figsizes.icml2022_half()):
        plt.figure()
        ts = np.linspace(0.0, 1.0, 400)
        sched = get_schedule(sched_name)
        sched_values = [sched.call_time_(t) for t in ts]
        plt.plot(ts, sched_values, 'tab:blue')
        plt.xlabel('$t$')
        plt.ylabel('$f(t)$')
        # plt.tight_layout()
        plot_name = paths.plots() / filename
        utils.ensureDir(plot_name)
        plt.savefig(plot_name)
        plt.close()


def plot_schedules(paths: Paths, filename: str, sched_names: List[str], sched_labels: List[str]) -> None:
    with plt.rc_context(figsizes.icml2022_half(height_to_width_ratio=0.4)):
        plt.figure()
        ts = np.linspace(0.0, 1.0, 400)
        for sched_name, sched_label in zip(sched_names, sched_labels):
            sched = get_schedule(sched_name)
            sched_values = [sched.call_time_(t) for t in ts]
            plt.plot(ts, sched_values, label=sched_label)
        plt.legend(loc='best')
        plt.xlabel('$t$')
        plt.ylabel('$f(t)$')
        # plt.tight_layout()
        plot_name = paths.plots() / filename
        utils.ensureDir(plot_name)
        plt.savefig(plot_name)
        plt.close()


def _create_benchmark_result_plot(file_path: Path, benchmark_results: Dict[str, Dict[str, float]],
                                  alg_names: List[str], colors: List):
    # generated mostly using ChatGPT
    df = pd.DataFrame(benchmark_results)

    # Reorder DataFrame based on alg_names
    df = df.reindex(alg_names)

    # Plotting
    # todo: use ICML compatible size
    fig, axs = plt.subplots(nrows=1, ncols=len(df.columns), figsize=(10, 7), sharey=True)

    for i, col in enumerate(df.columns):
        ax = axs[i]
        values = df[col].values
        bar_height = 1.0
        bar_positions = np.arange(len(df), dtype=np.float64)[::-1] * bar_height

        # Handle empty strings in alg_names to create gaps between bars
        mask = df.index != ''
        non_empty_indices = np.where(mask)[0]

        ax.xaxis.grid(True)

        # Plot only if the method name is not an empty string
        non_empty_values = values[mask]
        non_empty_bar_positions = bar_positions[non_empty_indices]
        # ax.barh(non_empty_bar_positions, non_empty_values, align='edge', color=colors[:len(non_empty_bar_positions)], alpha=0.8, height=bar_height)
        ax.barh(non_empty_bar_positions, non_empty_values, align='edge', color=[colors[j] for j in non_empty_indices],
                alpha=0.8, height=bar_height)

        # Add method names on the y-axis
        # ax.invert_yaxis()  # Invert y-axis to have Method A on top
        ax.tick_params(left=False)
        ax.set_yticks(bar_positions + 0.5 * bar_height)
        ax.set_yticklabels(df.index)

        ax.set_xlabel(r'Error increase in \% vs best')
        ax.set_title(col)

        # Remove frame around plot
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Set x-axis ticks and gridlines
        ax.xaxis.set_ticks_position('bottom')

        # Highlight x=0 tick and corresponding gridline
        ax.axvline(x=0, color='black', linewidth=1.5)

    # Set common labels and adjust layout
    # fig.text(0.5, 0.04, 'Performance', ha='center')
    # fig.suptitle('Method Performance Comparison', y=1.05)
    plt.tight_layout()
    utils.ensureDir(file_path)
    plt.savefig(file_path)
    plt.close(fig)


def _create_benchmark_result_plot_with_intervals(file_path: Path, benchmark_results: Dict[str, Dict[str, float]],
                                                 benchmark_intervals: Dict[str, Dict[str, Tuple[float, float]]],
                                                 alg_names: List[str], colors: List):
    n_benchmarks = len(benchmark_results)

    with plt.rc_context(figsizes.icml2022_full(height_to_width_ratio=1.3)):
        # Plotting
        fig, axs = plt.subplots(nrows=1, ncols=n_benchmarks, sharey=True)

        # for i, col in enumerate(df.columns):
        for i, (col, results) in enumerate(benchmark_results.items()):
            ax = axs[i]
            # values = df[col].values
            bar_height = 1.0
            bar_positions = np.arange(len(alg_names), dtype=np.float64)[::-1] * bar_height

            # Handle empty strings in alg_names to create gaps between bars
            # mask = df.index != ''
            mask = [alg_name != '' for alg_name in alg_names]
            non_empty_indices = np.where(mask)[0]

            non_empty_alg_names = [alg_name for alg_name in alg_names if alg_name != '']
            values = [results[alg_name] if alg_name in results else 0.0 for alg_name in non_empty_alg_names]

            ax.xaxis.grid(True)

            # Plot only if the method name is not an empty string
            non_empty_values = values
            non_empty_bar_positions = bar_positions[non_empty_indices]
            intervals = np.array([benchmark_intervals[col][alg_name]
                                  if alg_name in results else (0.0, 0.0)
                                  for alg_name in non_empty_alg_names]).transpose()
            rel_intervals = intervals - non_empty_values
            errors = np.array([-rel_intervals[0], rel_intervals[1]])  # turn them into (absolute) errors

            # ax.barh(non_empty_bar_positions, non_empty_values, align='edge', color=colors[:len(non_empty_bar_positions)], alpha=0.8, height=bar_height)
            ax.barh(non_empty_bar_positions, non_empty_values, align='edge',
                    color=[colors[j] for j in non_empty_indices],
                    alpha=0.8, height=bar_height)
            ax.errorbar(non_empty_values, non_empty_bar_positions + 0.5 * bar_height,
                        xerr=errors, fmt='none', color='black')

            # Add method names on the y-axis
            # ax.invert_yaxis()  # Invert y-axis to have Method A on top
            ax.tick_params(left=False)
            ax.set_yticks(bar_positions + 0.5 * bar_height)
            ax.set_yticklabels(alg_names)

            # ax.set_xlabel(r'Error increase in \% vs best ($\downarrow$)')
            ax.set_title(col)

            # Remove frame around plot
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            # Set x-axis ticks and gridlines
            ax.xaxis.set_ticks_position('bottom')

            # Highlight x=0 tick and corresponding gridline
            ax.axvline(x=0, color='black', linewidth=1.5)

        # Set common labels and adjust layout
        fig.text(0.6, -0.02, r'Error increase in \% vs best ($\downarrow$)', ha='center')
        # fig.suptitle('Method Performance Comparison', y=1.05)
        plt.tight_layout()
        utils.ensureDir(file_path)
        plt.savefig(file_path)
        plt.close(fig)


def get_equidistant_colors(n: int):
    cmap = plt.get_cmap('viridis')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=n - 1)
    colors = [cmap(norm(i)) for i in range(n)]
    return colors


def plot_benchmark_bars(paths: Paths, tables: ResultsTables, filename: str = None,
                        coll_names: Optional[List[str]] = None,
                        val_metric_name: Optional[str] = None, test_metric_name: Optional[str] = None,
                        alg_names: Optional[List[str]] = None,
                        simplify_name_fn: Optional[Callable[[str], str]] = None,
                        use_geometric_mean: bool = True, shift_eps: float = 1e-2):
    benchmark_results = {}
    benchmark_intervals = {}

    if coll_names is None:
        coll_names = ['meta-train-class', 'meta-test-class', 'meta-train-reg', 'meta-test-reg']

    for coll_name in coll_names:
        table = tables.get(coll_name)
        rel_means_dict, rel_intervals_dict = get_benchmark_results(paths, table=table, coll_name=coll_name,
                                                                   val_metric_name=val_metric_name,
                                                                   test_metric_name=test_metric_name,
                                                                   use_geometric_mean=use_geometric_mean,
                                                                   shift_eps=shift_eps,
                                                                   simplify_name_fn=simplify_name_fn)

        benchmark_results[coll_name] = rel_means_dict
        benchmark_intervals[coll_name] = rel_intervals_dict

    # ens_group_names = ['GBDTs-TD_MLP-TD', 'MLP-TD_MLP-TD-S', 'GBDTs-HPO', 'GBDTs-TD']
    ens_group_names = ['-HPO', '-TD']

    ens_alg_names = sum([[f'Ensemble{gn}', f'BestModel{gn}', ''] for gn in ens_group_names], [])

    # ens_alg_names = ['BestModel_GBDTs-HPO_MLP-HPO', ''] + ens_alg_names  # todo
    # ens_alg_names = ['HPO', ''] + ens_alg_names  # todo

    # single_alg_names = [
    #     # 'MLP-TD', 'MLP-TD-S', 'MLP-SKLD', '',
    #     'BestModel_MLP-HPO+TD', 'MLP-HPO', 'MLP-TD', 'MLP-TD-S', '',
    #     'BestModel_CatBoost-HPO+TD', 'CatBoost-HPO', 'CatBoost-TD', 'CatBoost-D', '',
    #     'BestModel_LGBM-HPO+TD', 'LGBM-HPO', 'LGBM-TD', 'LGBM-D', '',
    #     'BestModel_XGB-HPO+TD', 'XGB-HPO', 'XGB-TD', 'XGB-D', '',
    #     'RF-SKLD',
    # ]
    single_alg_names = [
        # 'MLP-TD', 'MLP-TD-S', 'MLP-SKLD', '',
        'MLP-HPO', 'MLP-TD', 'MLP-TD-S', '',
        'CatBoost-HPO', 'CatBoost-TD', 'CatBoost-D', '',
        'LGBM-HPO', 'LGBM-TD', 'LGBM-D', '',
        'XGB-HPO', 'XGB-TD', 'XGB-D', 'XGB-PBB-D', '',
        'RF-SKL-D',
    ]

    if alg_names is None:
        alg_names = ens_alg_names + single_alg_names

    mean_name = f'geometric_eps-{shift_eps:g}' if use_geometric_mean else 'arithmetic'
    if filename is None:
        filename = f'benchmarks_bars_{mean_name}.pdf'
    file_path = paths.plots() / filename

    # todo
    # colors = ['b'] * len(alg_names)
    colors = get_equidistant_colors(len(alg_names))

    _create_benchmark_result_plot_with_intervals(file_path=file_path, benchmark_results=benchmark_results,
                                                 benchmark_intervals=benchmark_intervals, alg_names=alg_names,
                                                 colors=colors)
    # _create_benchmark_result_plot(file_path=file_path, benchmark_results=benchmark_results, alg_names=alg_names,
    #                               colors=colors)


def plot_scatter_ax(paths: Paths, tables: ResultsTables, ax: matplotlib.axes.Axes, coll_name: str, alg_name_1: str,
                    alg_name_2: str,
                    test_metric_name: Optional[str] = None, val_metric_name: Optional[str] = None,
                    use_validation_errors: bool = False):
    task_collection = TaskCollection.from_name(coll_name, paths)
    task_infos = task_collection.load_infos(paths)
    task_type_name = 'class' if task_infos[0].tensor_infos['y'].is_cat() else 'reg'
    table = tables.get(coll_name=coll_name, n_cv=1, tag='paper')
    opt_groups = get_opt_groups(task_type_name)
    alg_group_dict = {'BestModel': (lambda an, tags, config: True), **{
        f'BestModel{group_name}': (lambda an, tags, config, ans=alg_names: an in ans)
        for group_name, alg_names in opt_groups.items()
    }}
    val_test_groups = {'HPO-on-BestModel-TD': {f'{family}-TD-{task_type_name}': f'{family}-HPO'
                                               for family in ['XGB', 'LGBM', 'CatBoost', 'MLP']}}
    test_table = table.get_test_results_table(DefaultEvalModeSelector(), alg_group_dict=alg_group_dict,
                                              test_metric_name=test_metric_name,
                                              val_metric_name=val_metric_name,
                                              val_test_groups=val_test_groups,
                                              use_validation_errors=use_validation_errors)
    test_table = test_table.filter_n_splits(n_splits=10)
    test_table.alg_names = [get_simplified_name(alg_name) for alg_name in test_table.alg_names]
    test_arr = test_table.to_array()
    mean_results = np.mean(test_arr, axis=-1)
    alg_1_results = mean_results[test_table.alg_names.index(alg_name_1)]
    alg_2_results = mean_results[test_table.alg_names.index(alg_name_2)]

    with plt.rc_context(figsizes.icml2022_half(height_to_width_ratio=1)):
        max_err = max(np.max(alg_1_results), np.max(alg_2_results))
        lim_err = max_err * 1.02
        ax.set_xlim(0.0, lim_err)
        ax.set_ylim(0.0, lim_err)
        # ax.set_xscale('symlog')
        # ax.set_yscale('symlog')
        ax.plot([0.0, lim_err], [0.0, lim_err], 'k-')
        ax.scatter(alg_1_results, alg_2_results, color='tab:blue', s=8.0, zorder=3)

        display_name_1 = get_display_name(alg_name_1)
        display_name_2 = get_display_name(alg_name_2)

        if test_metric_name is not None:
            raise NotImplementedError(f'Correct label for custom test metric name is not implemented')
        metric = 'Classification error' if task_type_name == 'class' else 'nRMSE'
        ax.set_xlabel(f'{metric} for {display_name_1}' + r' ($\downarrow$)')
        ax.set_ylabel(f'{metric} for {display_name_2}' + r' ($\downarrow$)')
        ax.set_title(coll_name_to_title(coll_name))

        # diagonal text version
        # eps = 0.3
        # # upper left text
        # ax.text(eps*lim_err, (1-eps)*lim_err, f'{alg_name_1} better',
        #         ha="center", va="center", rotation=45, size=11, zorder=-2)
        # # bottom right text
        # ax.text((1-eps) * lim_err, eps * lim_err, f'{alg_name_2} better',
        #         ha="center", va="center", rotation=45, size=11, zorder=-2)

        eps = 0.05
        # upper left text
        ax.text(eps * lim_err, (1 - eps) * lim_err, f'{display_name_1} better',
                ha="left", va="top", rotation=0, size=11, zorder=-2)
        # bottom right text
        ax.text((1 - eps) * lim_err, eps * lim_err, f'{display_name_2} better',
                ha="right", va="bottom", rotation=0, size=11, zorder=-2)


def plot_scatter(paths: Paths, filename: str, tables: ResultsTables, coll_names: List[str], alg_name_1: str,
                 alg_name_2: str,
                 test_metric_name: Optional[str] = None, val_metric_name: Optional[str] = None,
                 use_validation_errors: bool = False):
    print(f'Creating scatterplot: {filename}')
    context_mgr = plt.rc_context(figsizes.icml2022_half(height_to_width_ratio=1)) if len(coll_names) == 1 \
        else plt.rc_context(figsizes.icml2022_full(
        height_to_width_ratio=3 if len(coll_names) == 6 else (2 if len(coll_names) == 4 else 0.5)))
    with context_mgr:
        if len(coll_names) == 1:
            fig, ax = plt.subplots(1, 1)
            axs_list = [ax]
        elif len(coll_names) == 2:
            fig, axs = plt.subplots(1, 2)
            axs_list = [axs[0], axs[1]]
        elif len(coll_names) == 4:
            fig, axs = plt.subplots(2, 2)
            axs_list = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
        elif len(coll_names) == 6:
            fig, axs = plt.subplots(3, 2)
            axs_list = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1], axs[2, 0], axs[2, 1]]
        else:
            raise ValueError(f'{len(coll_names)=} is not in [1, 2, 4, 6]')

        for coll_name, ax in zip(coll_names, axs_list):
            plot_scatter_ax(ax=ax, paths=paths, tables=tables, coll_name=coll_name,
                            alg_name_1=alg_name_1, alg_name_2=alg_name_2,
                            val_metric_name=val_metric_name, test_metric_name=test_metric_name,
                            use_validation_errors=use_validation_errors)

        file_path = paths.plots() / filename
        utils.ensureDir(file_path)
        plt.savefig(file_path)
        plt.close(fig)


def _plot_scatter_with_labels(x_dict: Dict[str, float], y_dict: Dict[str, float],
                              y_intervals: Optional[Dict[str, Tuple[float, float]]],
                              ax: matplotlib.axes.Axes,
                              xlabel: str, ylabel: str, title: Optional[str] = None,
                              name_tfm_func: Optional[Callable[[str], str]] = None,
                              plot_pareto_frontier: bool = True,
                              arrow_alg_names: Optional[List[Tuple[str, str]]] = None,
                              pareto_frontier_width: float = 2.,
                              alg_names_to_hide: Optional[List[str]] = None):
    if alg_names_to_hide is None:
        alg_names_to_hide = []

    # First, convert dictionaries to a format suitable for seaborn
    # take shared models
    models = list(set(x_dict.keys()).intersection(set(y_dict.keys())))
    models.sort()
    print(f'{models=}')
    # show models not in both
    # print("Models not in both x and y dicts")
    # print(set(x_dict.keys()).symmetric_difference(set(y_dict.keys())))
    x_vals = [x_dict[model] for model in models]
    y_vals = [y_dict[model] for model in models]

    # Now, create a DataFrame from the dictionaries for easy plotting
    import pandas as pd
    df = pd.DataFrame({'model': models, 'x_value': x_vals, 'y_value': y_vals})

    # split model into model_name and model_type
    # replace underscores with -
    # df['model'] = df['model'].str.replace('_', '-')
    # df['model_name'] = df['model'].str.split('-', expand=True)[0]

    def get_model_type(alg_name: str) -> str:
        if '-HPO' in alg_name:
            return 'HPO'
        elif '-TD' in alg_name:
            return 'TD'
        # elif '-PBB-D' in alg_name:
        #     return 'PBB-D'
        elif '-D' in alg_name:
            return 'D'
        else:
            return 'unknown'

    # df['model_type'] = df['model'].str.split('-', expand=True)[1].str.split('(', expand=True)[0]
    df['model_type'] = [get_model_type(alg_name) for alg_name in df['model']]
    df['color'] = [get_plot_color(alg_name) for alg_name in models]
    df['alpha'] = [1.0 if alg_name not in alg_names_to_hide else 0.0 for alg_name in models]

    # Set up the figure size and style
    # fig = plt.figure(figsize=(10, 10))
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # sns.set_theme(style="whitegrid", font_scale=2)
    print(f'{df=}')

    color_mapping = {color: color for color in df['color'].unique()}

    # Create the scatter plot
    ax = sns.scatterplot(
        x="x_value",
        y="y_value",
        hue="color",
        style="model_type",
        data=df,
        s=400,  # size of the points
        palette=color_mapping,
        markers={'D': 'o', 'TD': 's', 'HPO': 'X', 'PBB-D': 'P'},
        # palette='tab10',  # palette can be changed as needed
        legend=False,  # No need to draw legend at this point
        ax=ax,
        alpha=df['alpha'],
    )

    ax.set_xscale('log')
    # ax.set_yscale('log')

    # Get the color of each point to set the color of the text
    point_colors = ax.collections[0].get_facecolor()

    if y_intervals is not None:
        y_intervals_arr = np.array([y_intervals[model] for model in models])
        y_errors_arr = np.stack([np.array(y_vals) - y_intervals_arr[:, 0],
                                 y_intervals_arr[:, 1] - np.array(y_vals)], axis=1)
        for x, y, errors, color in zip(x_vals, y_vals, y_errors_arr, point_colors):
            ax.errorbar(x, y, elinewidth=4, yerr=errors[:, None], fmt='none', color=color)

    # Prepare to annotate the points
    texts = []
    for i, point in enumerate(ax.collections[0].get_offsets()):
        model_name = df.iloc[i]['model']
        if model_name in alg_names_to_hide:
            continue
        x, y = point
        text_color = point_colors[i]
        # Annotate the model names
        display_name = model_name
        if name_tfm_func is not None:
            display_name = name_tfm_func(display_name)
        # bold if it's an arrow end
        is_arrow_end = False if arrow_alg_names is None else any(
            model_name == end_name for _, end_name in arrow_alg_names)
        if is_arrow_end:
            display_name = rf'\textbf{{{display_name}}}'
        text = ax.text(x, y, display_name, color=text_color, fontsize=20, ha='center', va='center')
        text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
        texts.append(text)

    # Use adjust_text to repel the labels
    # Use adjust_text to repel the labels from each other and the points
    adjust_text(texts,
                x=df['x_value'].values,
                y=df['y_value'].values,
                avoid_self=False,
                expand=(1.15, 1.3),
                ax=ax,
                )

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    eps = 0.12
    text_x = x_min ** (1 - eps) * x_max ** eps
    text_y = y_min + eps * (y_max - y_min)

    ax.set_axisbelow(True)

    # scatter.annotate('lower is better', xy=(text_x, text_y), rotation=)
    # ax.text(text_x, text_y, "lower is better",
    #         ha="center", va="center", rotation=-45, size=30)
    ax.text(text_x, text_y, "better",
            ha="center", va="center", rotation=45, size=30,
            bbox=dict(boxstyle="larrow,pad=0.5",
                      fc="lightgreen", ec="forestgreen", lw=4), zorder=50)

    # Set arrow coordinates based on the plot limits
    # arrow_x = x_min ** 0.1 * x_max ** 0.9  # Adjust 0.1 as needed
    # arrow_y = y_min + 0.1 * (y_max - y_min)  # Adjust 0.1 as needed
    #
    # # Set the corrected arrow properties
    # arrow_props = dict(facecolor='red', edgecolor='red', shrink=0.05, width=2, headwidth=10)
    #
    # # Add the arrow to the plot
    # ax.annotate('', xy=(arrow_x, arrow_y), xytext=(x_min, y_min),
    #                  arrowprops=arrow_props, annotation_clip=False)

    if plot_pareto_frontier:
        xs = np.array(x_vals)
        ys = np.array(y_vals)
        perm = np.argsort(xs)
        xs = xs[perm]
        ys = ys[perm]

        xs_pareto = [xs[0], xs[0]]
        ys_pareto = [ax.get_ylim()[1], ys[0]]
        for i in range(1, len(xs)):
            if ys[i] < ys_pareto[-1]:
                xs_pareto.append(xs[i])
                ys_pareto.append(ys_pareto[-1])
                xs_pareto.append(xs[i])
                ys_pareto.append(ys[i])
        xs_pareto.append(ax.get_xlim()[1])
        ys_pareto.append(ys_pareto[-1])

        ax.plot(xs_pareto, ys_pareto, '--', color='k', linewidth=pareto_frontier_width, zorder=0.8)

    if arrow_alg_names is not None:
        # arrow_head_length =
        for first, second in arrow_alg_names:
            if first in alg_names_to_hide or second in alg_names_to_hide:
                continue
            x1 = x_dict[first]
            y1 = y_dict[first]
            x2 = x_dict[second]
            y2 = y_dict[second]
            # plt.arrow(x1, y1, x2-x1, y2-y1, length_includes_head=True,
            #           head_width=0.08, head_length=0.00002)
            color = get_plot_color(second)
            color = tuple(list(color) + [0.5])  # add alpha channel
            # color = tuple(0.5 + 0.5*v for v in color)

            ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=5,
                        # arrowprops=dict(arrowstyle="->"),
                        arrowprops=dict(  # facecolor='#444444',
                            facecolor=color,
                            # width=3.0, headwidth=10.0, headlength=8.0,
                            shrink=0.01, edgecolor='none'))

    # Set the axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    # sns.reset_orig()


def extend_runtimes(times: Dict[str, float], task_type_name: str, keep_gpu: bool = True) -> Dict[str, float]:
    times = copy.copy(times)
    opt_groups = get_opt_groups(task_type_name)
    # for device in ['CPU', 'GPU']:
    for device in ['CPU']:
        # compute HPO times
        for method_name in ['RealMLP', 'MLP-RTDL', 'MLP-PLR', 'ResNet-RTDL', 'XGB', 'LGBM', 'CatBoost', 'TabR',
                            'RF', 'FTT']:
            if f'{method_name}-HPO-2_{device}' in times:
                times[f'{method_name}-HPO_{device}'] = (50. / 2.) * times[f'{method_name}-HPO-2_{device}']
            elif f'{method_name}-HPO-1_{device}' in times:
                times[f'{method_name}-HPO_{device}'] = 50. * times[f'{method_name}-HPO-1_{device}']
            elif f'{method_name}-TD_{device}' in times:
                # simple surrogate time
                print(f'Warning: Guessing HPO time for {method_name} on device {device} from TD time')
                times[f'{method_name}-HPO_{device}'] = 50 * times[f'{method_name}-TD_{device}']
            elif f'{method_name}-S-D_{device}' in times:
                # simple surrogate time
                print(f'Warning: Guessing HPO time for {method_name} on device {device} from S-D time')
                times[f'{method_name}-HPO_{device}'] = 50 * times[f'{method_name}-S-D_{device}']
            elif f'{method_name}-D_{device}' in times:
                # simple surrogate time
                print(f'Warning: Guessing HPO time for {method_name} on device {device} from D time')
                times[f'{method_name}-HPO_{device}'] = 50 * times[f'{method_name}-D_{device}']

            if f'{method_name}-HPO_{device}' in times:
                times[f'{method_name}-HPO_best-1-auc-ovr_{device}'] = times[f'{method_name}-HPO_{device}']

        # print(f'Warning: Guessing no-ls time for RealMLP on device {device} from ls time')
        # times[f'RealMLP-TD_no-ls_{device}'] = times[f'RealMLP-TD_{device}']
        # times[f'RealMLP-TD-S_no-ls_{device}'] = times[f'RealMLP-TD-S_{device}']

        for model in ['XGB', 'LGBM', 'CatBoost']:
            # simple surrogate times
            if f'{model}-HPO_{device}' not in times and f'{model}-TD_{device}' in times:
                print(f'Warning: Guessing HPO time for {model} on device {device} from TD time')
                times[f'{model}-HPO_{device}'] = 50 * times[f'{model}-TD_{device}']

        # raw_names = list(set('_'.join(name.split('_')[:-1]) for name in times))
        # print(f'Warning: Guessing additional times')
        # for name in raw_names:
        #     for new_suffix in ['_no-ls', '_val-ce', '_val-ce_no-ls', '_rssc']:
        #         old_name = f'{name}_CPU'
        #         new_name = f'{name}{new_suffix}_CPU'
        #         if new_name not in times and old_name in times:
        #             times[new_name] = times[old_name]

        for group_name, alg_names in opt_groups.items():
            if group_name not in ['-D', '-TD', '-HPO', '-D_val-ce', '-TD_val-ce'] and not group_name.endswith('_prep'):
                continue  # exclude the other ones for now
            alg_names = [
                alg_name.replace('-class', '').replace('-reg', '')
                for alg_name in alg_names]
            alg_device_names = [f'{alg_name}_{device}' for alg_name in alg_names]
            if all(alg_device_name in times for alg_device_name in alg_device_names):
                sum_time = sum([times[alg_device_name] for alg_device_name in alg_device_names])
                times[f'BestModel{group_name}_{device}'] = sum_time
                times[f'Ensemble{group_name}_{device}'] = sum_time

    if not keep_gpu:
        times = {key: value for key, value in times.items() if not 'GPU' in key}

    times = {get_simplified_name(key): value for key, value in times.items()}

    return times


def plot_pareto_ax(ax: matplotlib.axes.Axes, paths: Paths, tables: ResultsTables, coll_name: str,
                   alg_names: List[str],
                   val_metric_name: Optional[str] = None, test_metric_name: Optional[str] = None,
                   use_ranks: bool = False, use_normalized_errors: bool = False, tag: Optional[str] = None,
                   use_geometric_mean: bool = True, use_grinnorm_errors: bool = False,
                   shift_eps: float = 1e-2, use_validation_errors: bool = False,
                   arrow_alg_names: Optional[List[Tuple[str, str]]] = None, plot_pareto_frontier: bool = True,
                   alg_names_to_hide: Optional[List[str]] = None,
                   pareto_frontier_width: float = 2.):
    print(f'Creating plot for {coll_name}')
    is_reg = TaskCollection.from_name(coll_name, paths).load_infos(paths)[0].tensor_infos[
                 'y'].get_cat_size_product() == 0
    default_metric_name = ('1-r2' if use_grinnorm_errors else 'nrmse') if is_reg else 'class_error'
    if val_metric_name is None:
        val_metric_name = default_metric_name
    if test_metric_name is None:
        test_metric_name = default_metric_name
    table = tables.get(coll_name, n_cv=1, tag=tag or 'paper')
    rel_means_dict, rel_intervals_dict = get_benchmark_results(paths, table=table, coll_name=coll_name,
                                                               use_relative_score=False,
                                                               return_percentages=False,
                                                               val_metric_name=val_metric_name,
                                                               test_metric_name=test_metric_name,
                                                               use_ranks=use_ranks,
                                                               use_normalized_errors=use_normalized_errors,
                                                               use_grinnorm_errors=use_grinnorm_errors,
                                                               filter_alg_names_list=alg_names,
                                                               use_geometric_mean=use_geometric_mean,
                                                               shift_eps=shift_eps,
                                                               use_validation_errors=use_validation_errors)

    task_infos = TaskCollection.from_name(coll_name, paths).load_infos(paths)
    task_type_name = 'class' if task_infos[0].tensor_infos['y'].is_cat() else 'reg'
    time_coll_name = f'meta-train-{task_type_name}'

    # get runtimes
    avg_train_times = get_avg_train_times(paths, time_coll_name, per_1k_samples=True)
    # print(f'{avg_train_times=}')
    avg_train_times = extend_runtimes(avg_train_times, task_type_name=task_type_name, keep_gpu=False)

    # print(f'After extending: {avg_train_times=}')

    # def tfm_key(key: str) -> str:
    #     return key.replace('_CPU', ' (CPU)').replace('_GPU', ' (GPU)')
    def tfm_key(key: str) -> str:
        return key.replace('_CPU', '')

    avg_train_times = {tfm_key(key): value for key, value in avg_train_times.items()}

    # remove sklearn MLP
    if 'MLP-SKL-D' in avg_train_times:
        del avg_train_times['MLP-SKL-D']
    # if 'ResNet-RTDL-D' in avg_train_times:
    #     del avg_train_times['ResNet-RTDL-D']

    # convert MLP-HPO runtime
    # get simplified associated names (without the CPU/GPU thing)
    # generate ensemble/BestModel runtimes?
    # add CPU/GPU to rel_means_dict keys

    extended_means_dict = rel_means_dict
    extended_intervals_dict = rel_intervals_dict

    print(f'{list(avg_train_times.keys())=}')
    print(f'{list(extended_means_dict.keys())=}')

    common_keys = set(avg_train_times.keys()).intersection(set(extended_means_dict.keys()))

    # avg_train_times = {tfm_alg_name(key): value for key, value in avg_train_times.items() if key in common_keys}
    # extended_means_dict = {tfm_alg_name(key): value for key, value in extended_means_dict.items() if key in common_keys}
    # extended_intervals_dict = {tfm_alg_name(key): value for key, value in extended_intervals_dict.items() if
    #                            key in common_keys}

    # extended_means_dict = utils.join_dicts(
    #     *[{f'{key} ({device})': value for key, value in rel_means_dict.items()} for device in ['CPU', 'GPU']]
    # )
    # extended_intervals_dict = utils.join_dicts(
    #     *[{f'{key} ({device})': value for key, value in rel_intervals_dict.items()} for device in ['CPU', 'GPU']]
    # )

    # print('times keys:', sorted(list(avg_train_times.keys())))
    # print('means keys:', sorted(list(extended_means_dict.keys())))
    #
    # print(f'x_dict = {avg_train_times}')
    # print(f'y_dict = {extended_means_dict}')

    title = coll_name_to_title(coll_name)
    # coll_name_latex = coll_name
    # for split_name in ['train', 'test']:
    #     if coll_name == f'meta-{split_name}-{task_type_name}':
    #         coll_name_latex = r'$\mathcal{B}^{\mathrm{' + split_name + r'}}_{\mathrm{' + task_type_name + r'}}$'
    ylabel = ('Shifted geometric mean' if use_geometric_mean else 'Arithmetic mean') + ' of '
    if use_ranks:
        ylabel = ylabel + r'\textbf{ranks}'
    else:
        if use_normalized_errors:
            ylabel = ylabel + r'\textbf{normalized} '
        elif use_grinnorm_errors:
            ylabel = ylabel + r'\textbf{custom-normalized} '
        if task_type_name == 'class':
            if test_metric_name is None or test_metric_name == 'class_error':
                ylabel = ylabel + r'\textbf{classification errors}'
            elif test_metric_name == '1-auc_ovr':
                ylabel = ylabel + r'\textbf{1-AUC(one-vs-rest)}'
            elif test_metric_name == 'cross_entropy':
                ylabel = ylabel + r'\textbf{cross-entropies}'
            else:
                raise ValueError(f'Test metric {test_metric_name} not implemented')
        else:
            if test_metric_name is None or test_metric_name == 'rmse':
                ylabel = ylabel + r'\textbf{RMSEs}'
            elif test_metric_name == 'nrmse':
                ylabel = ylabel + r'\textbf{nRMSEs}'
            elif test_metric_name == '1-r2':
                ylabel = ylabel + r'$1-R^2$'
            else:
                raise ValueError(f'Test metric {test_metric_name} not implemented')
    _plot_scatter_with_labels(avg_train_times, extended_means_dict,
                              y_intervals=extended_intervals_dict,
                              xlabel=r'Average training \textbf{time (CPU)} per 1K samples [s]',
                              # + r' ($\downarrow$)',
                              ylabel=ylabel,
                              ax=ax,
                              title=title,
                              name_tfm_func=get_display_name,
                              arrow_alg_names=arrow_alg_names,
                              plot_pareto_frontier=plot_pareto_frontier,
                              alg_names_to_hide=alg_names_to_hide,
                              pareto_frontier_width=pareto_frontier_width,
                              # ylabel=r'Benchmark score relative to best model',
                              # ylabel=r'Error increase in \% vs best ($\downarrow$)',
                              # title=f'Benchmark scores on {coll_name_latex} vs train time',
                              )


def shorten_coll_names(coll_names: List[str]) -> List[str]:
    coll_name_dict = {'meta-train-class': 'mtrc', 'meta-train-reg': 'mtrr',
                      'meta-test-class': 'mtec', 'meta-test-reg': 'mter',
                      'grinsztajn-class-filtered': 'gcf', 'grinsztajn-reg': 'gr'}
    short_coll_names = [coll_name if coll_name not in coll_name_dict else coll_name_dict[coll_name] for coll_name in
                        coll_names]
    return short_coll_names


def plot_pareto(paths: Paths, tables: ResultsTables, coll_names: List[str], alg_names: List[str],
                val_metric_name: Optional[str] = None, test_metric_name: Optional[str] = None,
                use_ranks: bool = False, use_normalized_errors: bool = False, filename: Optional[str] = None,
                filename_suffix: Optional[str] = None,
                tag: Optional[str] = None, use_grinnorm_errors: bool = False,
                use_geometric_mean: bool = True, shift_eps: float = 1e-2, use_validation_errors: bool = False,
                arrow_alg_names: Optional[List[Tuple[str, str]]] = None, plot_pareto_frontier: bool = True,
                alg_names_to_hide: Optional[List[str]] = None,
                subfolder: Optional[str] = None,
                pareto_frontier_width: float = 2., use_2x3: bool = False):
    print(f'Plotting pareto plot for {coll_names}')
    sns.set_theme(style="whitegrid", font_scale=2)
    if len(coll_names) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        axs_list = [ax]
    elif len(coll_names) == 2:
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        axs_list = [axs[0], axs[1]]
    elif len(coll_names) == 3:
        fig, axs = plt.subplots(1, 3, figsize=(30, 10))
        axs_list = [axs[0], axs[1], axs[2]]
    elif len(coll_names) == 4:
        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        axs_list = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
    elif len(coll_names) == 6:
        if use_2x3:
            fig, axs = plt.subplots(2, 3, figsize=(30, 20))
            axs_list = [axs[0, 0], axs[1, 0], axs[0, 1], axs[1, 1], axs[0, 2], axs[1, 2]]
        else:
            fig, axs = plt.subplots(3, 2, figsize=(20, 30))
            axs_list = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1], axs[2, 0], axs[2, 1]]
    else:
        raise ValueError(f'{len(coll_names)=} is not in [1, 2, 3, 4, 6]')

    for coll_name, ax in zip(coll_names, axs_list):
        # print(f'{val_metric_name=}, {test_metric_name=}, {coll_name=}')
        plot_pareto_ax(ax=ax, paths=paths, tables=tables, coll_name=coll_name, alg_names=alg_names,
                       val_metric_name=val_metric_name, test_metric_name=test_metric_name,
                       use_ranks=use_ranks, use_normalized_errors=use_normalized_errors, tag=tag,
                       use_grinnorm_errors=use_grinnorm_errors,
                       use_geometric_mean=use_geometric_mean, shift_eps=shift_eps,
                       use_validation_errors=use_validation_errors,
                       arrow_alg_names=arrow_alg_names, plot_pareto_frontier=plot_pareto_frontier,
                       alg_names_to_hide=alg_names_to_hide,
                       pareto_frontier_width=pareto_frontier_width)

    mean_name = f'geometric_eps-{shift_eps:g}' if use_geometric_mean else 'arithmetic'
    if use_ranks:
        mean_name = 'ranks_' + mean_name
    elif use_normalized_errors:
        mean_name = 'normerrors_' + mean_name
    elif use_grinnorm_errors:
        mean_name = 'grinnormerrors_' + mean_name

    name_parts = shorten_coll_names(coll_names) + [mean_name]
    if use_validation_errors:
        name_parts = ['validation'] + name_parts

    if use_2x3:
        name_parts = ['2x3'] + name_parts

    plots_path = paths.plots()
    if subfolder is not None:
        plots_path = plots_path / subfolder

    if filename is None:
        file_path = plots_path / f'pareto_{"_".join(name_parts)}.pdf'
    else:
        file_path = plots_path / filename
    if filename_suffix is not None:
        file_path = file_path.with_stem(f'{file_path.stem}{filename_suffix}')

    if len(coll_names) in [4, 6]:
        labels = ['D = defaults {} {} {} {} {} TD = tuned defaults {} {} {} {} {} HPO = hyperparameter optimization',
                  'Best/Ensemble: out of XGB, LGBM, CatBoost, (Real)MLP']
        r = matplotlib.patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='none',
                                         visible=False)
        fig.legend(handles=[r] * len(labels), labels=labels, fontsize=30,
                   handlelength=0, handletextpad=0, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=1)

        # plt.tight_layout(rect=[0, 0.09, 1.0, 1.0])

    utils.ensureDir(file_path)
    plt.savefig(file_path, bbox_inches='tight')
    plt.close(fig)
    sns.reset_orig()
    print(f'Created plot {file_path}')


def plot_winrates(paths: Paths, tables: ResultsTables, coll_name: str, alg_names: List[str],
                  val_metric_name: Optional[str] = None, test_metric_name: Optional[str] = None):
    print(f'Plotting winrate matrix plot for {coll_name}')
    table = tables.get(coll_name)
    task_collection = TaskCollection.from_name(coll_name, paths)
    task_infos = task_collection.load_infos(paths)
    task_type_name = 'class' if task_infos[0].tensor_infos['y'].is_cat() else 'reg'
    opt_groups = get_opt_groups(task_type_name)
    alg_group_dict = {'BestModel': (lambda an, tags, config: not an.startswith('Ensemble')), **{
        f'BestModel{group_name}': (lambda an, tags, config, ans=alg_names: an in ans)
        for group_name, alg_names in opt_groups.items()
    }}
    test_table = table.get_test_results_table(DefaultEvalModeSelector(), alg_group_dict=alg_group_dict,
                                              test_metric_name=test_metric_name,
                                              val_metric_name=val_metric_name)
    simplify_name_fn = get_simplified_name
    test_table = test_table.rename_algs(simplify_name_fn)
    test_table = test_table.filter_algs(alg_names)

    use_task_weighting = coll_name.startswith('meta-train') or coll_name.startswith('uci')
    if use_task_weighting:
        separate_task_names = ['facebook_comment_volume', 'facebook_live_sellers_thailand_shares']
        task_weights = TaskWeighting(test_table.task_infos, separate_task_names).get_task_weights()
    else:
        n_tasks = len(test_table.task_infos)
        task_weights = np.ones(n_tasks) / n_tasks

    n_splits = 10
    test_table = test_table.filter_n_splits(n_splits)
    # shape: [n_algs, n_tasks, n_splits]
    errors = test_table.to_array()

    # do it once with < and once with <= to make sure that ties count as half a win
    wins_tensor = 0.5 * ((errors[:, None] <= errors[None, :]).astype(np.float32)
                         + (errors[:, None] < errors[None, :]).astype(np.float32))

    avg_wins_per_task = np.mean(wins_tensor, axis=-1)  # average over splits
    # average wins by task weights
    winrate_matrix = np.einsum('ijt,t->ij', avg_wins_per_task, task_weights)
    win_percentage_matrix = 100.0 * winrate_matrix

    perm = np.argsort(np.mean(win_percentage_matrix, axis=-1))  # sort by average winrate
    win_percentage_matrix = win_percentage_matrix[perm, :][:, perm]
    alg_names = [test_table.alg_names[i] for i in perm]
    alg_names = [alg_name.replace('_', r'\_') for alg_name in alg_names]

    # with matplotlib.rc_context():
    # Create a heatmap using seaborn
    fig = plt.figure(figsize=(10, 8))
    sns.set_theme(style="white", font_scale=0.6)
    mask = np.eye(win_percentage_matrix.shape[0], dtype=bool)
    heatmap = sns.heatmap(win_percentage_matrix, annot=True, fmt=".1f", cmap="YlGnBu",
                          vmin=0, vmax=100, linewidths=0.5, mask=mask,
                          square=True, cbar_kws={"shrink": 0.8})

    display_alg_names = [get_display_name(an) for an in alg_names]

    # Set labels for rows and columns
    heatmap.set_xticklabels(display_alg_names, rotation=90, fontsize=8)  # Adjust font size
    heatmap.set_yticklabels(display_alg_names, rotation=0, fontsize=8)  # Adjust font size

    # Remove x and y labels
    heatmap.set_xlabel('')
    heatmap.set_ylabel('')

    # Add a label to the color scale
    cbar = heatmap.collections[0].colorbar
    # cbar.set_label("Percentage of row wins", fontsize=10)

    heatmap.set_title(coll_name_to_title(coll_name) + ', percentage of row wins', fontsize=15)

    file_path = paths.plots() / f'winrate_matrix_{coll_name}.pdf'
    utils.ensureDir(file_path)
    plt.savefig(file_path)
    plt.close(fig)
    sns.reset_orig()


def plot_stopping_ax(ax: plt.Axes, paths: Paths, tables: ResultsTables, method: str, classification: bool):
    esr_list = [10, 20, 50, 100, 300, 1000]

    ax.set_xscale('log')

    ax.plot([10, 1000], [0.0, 0.0], 'k--')

    if classification:
        combinations = [('meta-train-class', 'stopped on classification error', '', 'tab:blue'),
                        ('meta-train-class', 'stopped on Brier loss', '_val-brier', 'tab:orange'),
                        ('meta-train-class', 'stopped on cross-entropy loss', '_val-ce',
                         'tab:green')]
    else:
        combinations = [('meta-train-reg', 'stopped on RMSE', '', 'tab:blue')]

    for coll_name, label, suffix, color in combinations:
        table = tables.get(coll_name, n_cv=1, tag='paper_early_stopping')
        # print(f'{table.test_table.alg_names=}')
        rel_alg_name = method + '_esr-1000'  # stopped on standard metric
        rel_results, rel_intervals = get_benchmark_results(paths, table=table, coll_name=coll_name,
                                                           rel_alg_name=rel_alg_name)
        alg_names = [method + suffix + f'_esr-{esr}' for esr in esr_list]
        results_list = [rel_results[alg_name] for alg_name in alg_names]
        lower_list = [rel_intervals[alg_name][0] for alg_name in alg_names]
        upper_list = [rel_intervals[alg_name][1] for alg_name in alg_names]

        ax.plot(esr_list, results_list, '.-', color=color, label=label)
        ax.fill_between(esr_list, lower_list, upper_list, color=color, alpha=0.3)
        ax.set_xlabel('Stopping patience')
        ax.set_xticks(esr_list, labels=[str(esr) for esr in esr_list])
        ax.grid(True)


def plot_stopping(paths: Paths, tables: ResultsTables, classification: bool):
    print(f'Generating stopping plot')

    with plt.rc_context(figsizes.icml2022_full(height_to_width_ratio=0.9)):
        fig, axs = plt.subplots(1, 3, sharey='all')
        for i, method in enumerate(['XGB-TD', 'LGBM-TD', 'CatBoost-TD']):
            ax = axs[i]
            ax.set_title(method)
            plot_stopping_ax(ax, paths, tables, method=method, classification=classification)

        # axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

        axs[0].set_ylabel(r'Error increase in \%')

        fig.legend(*axs[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, 0.15), ncol=3)

        task_type_name = 'class' if classification else 'reg'

        file_path = paths.plots() / f'stopping_{task_type_name}.pdf'
        plt.tight_layout(rect=[0, 0.15, 1.0, 1.0])

        if classification:
            y_min, y_max = axs[0].get_ylim()
            y_max = min(y_max, 15)
            axs[0].set_ylim(y_min, y_max)

        utils.ensureDir(file_path)
        plt.savefig(file_path)
        plt.close(fig)


def get_equidistant_blue_colors(n: int):
    # cmap = plt.get_cmap('viridis')
    cmap = sns.color_palette("ch:s=.25,rot=-.25", n)
    # cmap = sns.color_palette("viridis", n)
    # norm = matplotlib.colors.Normalize(vmin=0, vmax=n - 1)
    # colors = [cmap(norm(i)) for i in range(n)]
    colors = [cmap[i] for i in range(n)]
    return colors


def _create_cumul_abl_plot(file_path: Path, benchmark_results: Dict[str, Dict[str, float]],
                           benchmark_intervals: Dict[str, Dict[str, Tuple[float, float]]],
                           alg_names: List[str], colors: List, contribs: List[str],
                           improv_groups: List[str]):
    n_benchmarks = len(benchmark_results)

    n_improvements = len(list(benchmark_results.values())[0])
    start_color = mcolors.to_rgb('tab:blue')  # Color for vanilla MLP
    end_color = mcolors.to_rgb('tab:green')  # Color for final MLP
    gradient_colors = [mcolors.to_hex(c) for c in np.linspace(start_color, end_color, n_improvements)]

    start_alpha = 0.3
    end_alpha = 0.6
    alpha_cumulative_list = np.linspace(start_alpha, end_alpha, n_improvements)
    start_alpha_improvement = 0.65
    end_alpha_improvement = 1.
    alpha_improvement_list = np.linspace(start_alpha_improvement, end_alpha_improvement, n_improvements)

    with plt.rc_context(figsizes.icml2022_half(height_to_width_ratio=1.5)):
        # Plotting
        fig, axs = plt.subplots(nrows=1, ncols=n_benchmarks, sharey=True)

        # for i, col in enumerate(df.columns):
        for i, (col, results) in enumerate(benchmark_results.items()):
            ax = axs[i]
            # values = df[col].values
            # bar_height = 1.0
            # bar_positions = np.arange(len(alg_names), dtype=np.float64)[::-1] * bar_height
            bar_height = 1.0
            bar_positions = np.arange(len(alg_names), dtype=np.float64)[::-1] * (bar_height + 0.1)

            # Handle empty strings in alg_names to create gaps between bars
            # mask = df.index != ''
            mask = [alg_name != '' for alg_name in alg_names]
            non_empty_indices = np.where(mask)[0]

            non_empty_alg_names = [alg_name for alg_name in alg_names if alg_name != '']
            values = [results[alg_name] if alg_name in results else 0.0 for alg_name in non_empty_alg_names]

            ax.xaxis.grid(True)

            # Plot only if the method name is not an empty string
            non_empty_values = values
            non_empty_bar_positions = bar_positions[non_empty_indices]
            print(non_empty_bar_positions)
            intervals = np.array([benchmark_intervals[col][alg_name]
                                  if alg_name in results else (0.0, 0.0)
                                  for alg_name in non_empty_alg_names]).transpose()
            rel_intervals = intervals - non_empty_values
            errors = np.array([-rel_intervals[0], rel_intervals[1]])  # turn them into (absolute) errors

            for j in range(len(non_empty_values)):
                value = non_empty_values[j]
                last_value = value if j == 0 else non_empty_values[j - 1]
                ax.barh(non_empty_bar_positions[j], min(value, last_value), align='edge', color=gradient_colors[j],
                        alpha=alpha_cumulative_list[j],
                        height=bar_height)
                if value > last_value:
                    ax.barh(non_empty_bar_positions[j], value - last_value, left=last_value, align='edge',
                            color=gradient_colors[j],
                            alpha=alpha_improvement_list[j], height=bar_height)
                elif value < last_value:
                    ax.barh(non_empty_bar_positions[j], last_value - value, left=value, align='edge',
                            color=gradient_colors[j],
                            # color='white', edgecolor='red', hatch='/', linewidth=2,
                            # color='red', fill=False,
                            # color='white',
                            edgecolor='tab:green', hatch='//' * 3,
                            facecolor='none',
                            linewidth=0,
                            alpha=alpha_improvement_list[j], height=bar_height)
            ax.errorbar(non_empty_values, non_empty_bar_positions + 0.5 * bar_height,
                        xerr=errors, fmt='none', color='gray', linewidth=0.8)

            # Add method names on the y-axis
            ax.tick_params(left=False)
            ax.set_yticks(bar_positions + 0.5 * bar_height)
            # Get the default font size for y-tick labels
            default_fontsize = plt.rcParams['ytick.labelsize']
            font_properties = {'family': 'sans-serif', 'size': default_fontsize + 1}
            ax.set_yticklabels(alg_names, fontdict=font_properties)

            # ax.set_xlabel(r'Error increase in \% vs best ($\downarrow$)')
            ax.set_title(col)

            # Remove frame around plot
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            # Set x-axis ticks and gridlines
            ax.xaxis.set_ticks_position('bottom')

            # Highlight x=0 tick and corresponding gridline
            # ax.axvline(x=-0.09, color='black', linewidth=1.5) #FIXME it'd be better but right now it's a bit off compared to the grid lines

            color_map = {'New': '#ff7f0e', 'Unusual': '#2ca02c',
                         'default': (0.35, 0.35, 0.35)}
            colors_contrib = [color_map.get(key, 'black') for key in contribs]

            for label, color in zip(ax.get_yticklabels(), colors_contrib):
                label.set_color(color)

        max_value = max(max(results.values()) for results in benchmark_results.values())
        for ax in axs:
            ax.set_xlim(0, max_value * 1.1)  # Add some padding

        # Identify the unique categories and their y-coordinates
        # unique values with the right order
        unique_groups = list(dict.fromkeys(improv_groups))
        group_indices = {group: [] for group in unique_groups}

        # Loop over improvements and store the indices for each group
        for i, group in enumerate(improv_groups):
            if group in group_indices:
                group_indices[group].append(i)

        # Calculate the bracket positions
        bracket_positions = {}
        bracket_widths = {}
        for group, indices_ in group_indices.items():
            # add 1 to the indices to take into account the first bar
            indices = [i + 1 for i in indices_]
            start_pos = non_empty_bar_positions[min(indices)] - 0.0
            end_pos = non_empty_bar_positions[max(indices)]
            bracket_positions[group] = (start_pos + end_pos) / 2 + bar_height / 2 + 0.0
            bracket_widths[group] = (start_pos - end_pos) * 0.9 + bar_height - 0.4

        # Call the draw_bracket function for each unique group
        text_offset = 0.3  # Offset for the text annotation from the bracket

        # find the very left of the figure in ax[0] coordinates
        left = -31.5  # TODO

        for group in unique_groups:
            y = bracket_positions[group]
            width = bracket_widths[group]
            # show text on the left side of the figure
            axs[0].annotate(group, xy=(left, y), xytext=(left - text_offset, y),
                            ha='right', va='center', color='black', fontsize='small',
                            rotation=90,
                            arrowprops=dict(arrowstyle=f'-[, widthB={width}, lengthB=0.5', lw=1., color='black'),
                            annotation_clip=False,
                            font_properties={'family': 'sans-serif', 'size': default_fontsize - 1})

        # make a legend
        legend_elements = [mpatches.Patch(facecolor=color_map[key], edgecolor='black', label=key) for key in color_map \
                           if key != "default"]
        font_properties = {'family': 'sans-serif', 'size': default_fontsize}
        fig.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.09, 0.00),
                   prop=font_properties)

        # # Set common labels and adjust layout
        fig.text(0.65, -0.02, r'Benchmark score improvement (\%) vs. vanilla', ha='center')
        # fig.suptitle('Method Performance Comparison', y=1.05)
        # plt.show()
        # plt.tight_layout() # break the annotations
        # utils.ensureDir(file_path)
        plt.savefig(file_path)
        plt.close(fig)


def plot_cumulative_ablations(paths: Paths, tables: ResultsTables, filename: str = None,
                              val_metric_name: Optional[str] = None, test_metric_name: Optional[str] = None,
                              use_geometric_mean: bool = True, shift_eps: float = 1e-2):
    print(f'Creating cumulative ablations plot')

    improvements = {
        'vanilla': (r'\textbf{Vanilla MLP}', 'default'),
        'robust-scale-smooth-clip': ('Robust scale + smooth clip', 'New', "Preprocessing"),
        'one-hot-small-cat': ('One-hot for small cat.', 'default', "Preprocessing"),
        'no-early-stop': ('No early stopping', 'default', "Hyperparameters"),
        'last-best-epoch': ('Last best epoch', 'Unusual', "Hyperparameters"),
        'lr-multi-cycle': (r'$\mathrm{coslog}_4$ lr sched', 'Unusual', "Hyperparameters"),
        'beta2-0.95': (r'Adam $\beta_2 = 0.95$', 'Unusual', "Hyperparameters"),
        'label-smoothing': (r'Label smoothing (class.)', 'Unusual', "Hyperparameters"),
        'output-clipping': (r'Output clipping (reg.)', 'Unusual', "Hyperparameters"),
        'ntp': (r'NT parametrization', 'Unusual', "Architecture"),
        'different-act': (r'Act. fn. SELU / Mish', 'default', "Architecture"),
        'param-act': (r'Parametric act. fn.', 'Unusual', "Architecture"),
        'front-scale': (r'Scaling layer', 'New', "Architecture"),
        'num-emb-pl': (r'Num. embeddings: PL', 'default', "Architecture"),
        'num-emb-pbld': (r'PL emb.\ $\to$ PBLD emb.', 'New', "Architecture"),
        'alt-pdrop-0.15': (r'Dropout $p=0.15$', 'default', "Regularization"),
        'alt-pdrop-flat-cos': (r'Dropout sched: $\mathrm{flat\_cos}$', 'New', "Regularization"),
        'alt-wd-0.02': (r'Weight decay wd $= 0.02$', 'default', "Regularization"),
        'alt-wd-flat-cos': (r'wd sched: $\mathrm{flat\_cos}$', 'New', "Regularization"),
        'alt-bias-init-he+5': (r'Bias init: he+5', 'Unusual', "Initialization"),
        'alt-weight-init-std': (r'Weight init: data-driven', 'New', "Initialization"),
        'final': (r'\textbf{= RealMLP}', "default")
    }

    group_labels = {key: value[0] for key, value in improvements.items()}
    contribs = [value[1] for key, value in improvements.items()]
    improv_groups = [value[2] for key, value in improvements.items() if len(value) > 2]

    coll_names = ['meta-train-class', 'meta-train-reg']
    benchmark_results = {}
    benchmark_intervals = {}

    for coll_name in coll_names:
        table = tables.get(coll_name, tag='paper_cumulative_ablations_new')
        rel_means_dict, rel_intervals_dict = get_benchmark_results(paths, table=table, coll_name=coll_name,
                                                                   val_metric_name=val_metric_name,
                                                                   test_metric_name=test_metric_name,
                                                                   use_relative_score=False, return_percentages=False,
                                                                   use_geometric_mean=use_geometric_mean,
                                                                   shift_eps=shift_eps)

        alg_names = list(rel_means_dict.keys())
        vanilla_alg_names = [alg_name for alg_name in alg_names if 'vanilla' in alg_name]
        vanilla_alg_results = [rel_means_dict[alg_name] for alg_name in vanilla_alg_names]
        best_vanilla_alg_name = vanilla_alg_names[np.argmin(vanilla_alg_results)]

        # get the results again, but now relative to the best vanilla alg, in percent
        rel_means_dict, rel_intervals_dict = get_benchmark_results(paths, table=table, coll_name=coll_name,
                                                                   val_metric_name=val_metric_name,
                                                                   test_metric_name=test_metric_name,
                                                                   rel_alg_name=best_vanilla_alg_name,
                                                                   use_geometric_mean=use_geometric_mean,
                                                                   shift_eps=shift_eps)

        # group different lr values together
        alg_group_names = [alg_name.split('_')[-2] if len(alg_name.split('_')) >= 2 else '' for alg_name in alg_names]
        # alg_group_names_unique = list(set(alg_group_names))

        rel_means_dict_group = dict()
        rel_intervals_dict_group = dict()
        for alg_group_name, display_name in group_labels.items():
            # alg names in this group
            group_alg_names = [an for an, agn in zip(alg_names, alg_group_names) if agn == alg_group_name]
            if len(group_alg_names) == 0:
                print(f'No algs for group {alg_group_name}')
                continue
            best_alg_name = group_alg_names[np.argmin([rel_means_dict[an] for an in group_alg_names])]
            print(f'best lr: {best_alg_name.split("_")[-1]} for {alg_group_name}')
            rel_means_dict_group[alg_group_name] = -rel_means_dict[best_alg_name]
            low, high = rel_intervals_dict[best_alg_name]
            rel_intervals_dict_group[alg_group_name] = -high, -low

        benchmark_results[coll_name] = rel_means_dict_group
        benchmark_intervals[coll_name] = rel_intervals_dict_group

    for coll_name in coll_names:
        for mydict in [benchmark_results, benchmark_intervals]:
            # copy the last result because we need it twice but we can't have the same dictionary key twice
            print(f'{list(mydict[coll_name].keys())=}')
            mydict[coll_name]['final'] = mydict[coll_name]['alt-weight-init-std']

    # change keys to descriptions
    def map_keys(f: Dict, to_be_mapped: Dict):
        return {f[key]: value for key, value in to_be_mapped.items()}

    for coll_name in benchmark_results:
        benchmark_results[coll_name] = map_keys(group_labels, benchmark_results[coll_name])
        benchmark_intervals[coll_name] = map_keys(group_labels, benchmark_intervals[coll_name])

    alg_names = list(benchmark_results['meta-train-class'].keys())

    # colors = ['b'] * len(alg_names)
    colors = get_equidistant_blue_colors(len(list(group_labels.keys())))
    # colors = ['tab:blue'] * len(list(group_labels.keys()))

    if filename is None:
        filename = f'cumulative_ablations.pdf'
    file_path = paths.plots() / filename
    _create_cumul_abl_plot(file_path=file_path, benchmark_results=benchmark_results,
                           benchmark_intervals=benchmark_intervals, alg_names=alg_names,
                           colors=colors, contribs=contribs, improv_groups=improv_groups)


def plot_cdd_ax(ax: matplotlib.axes.Axes, paths: Paths, tables: ResultsTables, coll_name: str,
                alg_names: List[str],
                val_metric_name: Optional[str] = None, test_metric_name: Optional[str] = None,
                tag: Optional[str] = None, use_validation_errors: bool = False):
    print(f'Creating plot for {coll_name}')
    table = tables.get(coll_name, n_cv=1, tag=tag or 'paper')
    simplify_name_fn = get_simplified_name
    n_splits = 10

    task_collection = TaskCollection.from_name(coll_name, paths)
    task_infos = task_collection.load_infos(paths)
    task_type_name = 'class' if task_infos[0].tensor_infos['y'].is_cat() else 'reg'
    opt_groups = get_opt_groups(task_type_name)
    alg_group_dict = {'BestModel': (lambda an, tags, config: not an.startswith('Ensemble')), **{
        f'BestModel{group_name}': (lambda an, tags, config, ans=alg_names: an in ans)
        for group_name, alg_names in opt_groups.items()
    }}
    test_table = table.get_test_results_table(DefaultEvalModeSelector(), alg_group_dict=alg_group_dict,
                                              test_metric_name=test_metric_name,
                                              val_metric_name=val_metric_name,
                                              use_validation_errors=use_validation_errors)
    test_table = test_table.rename_algs(simplify_name_fn)
    # print(f'{test_table.alg_names=}')
    # print(f'{filter_alg_names_list=}')
    test_table = test_table.filter_algs(alg_names)

    # new code
    test_table = test_table.filter_n_splits(n_splits)
    # shape: [n_algs, n_tasks, n_splits]
    errors = test_table.to_array()
    errors = np.mean(errors, axis=2)  # average over splits

    # adapted from https://sherbold.github.io/autorank/
    data = pd.DataFrame()
    for i, alg_name in enumerate(test_table.alg_names):
        data[get_display_name(alg_name)] = errors[i]
    from autorank import autorank, plot_stats, create_report, latex_table
    result = autorank(data, alpha=0.05, verbose=False, order='ascending', force_mode='nonparametric')
    plot_stats(result, ax=ax, allow_insignificant=True)
    print(create_report(result))
    ax.set_title('grinsztajn-class' if coll_name == 'grinsztajn-class-filtered' else coll_name)


def plot_cdd(paths: Paths, tables: ResultsTables, coll_names: List[str], alg_names: List[str],
             val_metric_name: Optional[str] = None, test_metric_name: Optional[str] = None,
             filename: Optional[str] = None,
             tag: Optional[str] = None,
             use_validation_errors: bool = False):
    print(f'Plotting pareto plot for {coll_names}')
    old_value = plt.rcParams['text.usetex']
    plt.rcParams['text.usetex'] = False  # apparently doesn't work with the cdd plot package (autorank)
    assert len(coll_names) in [1, 2, 4, 6]
    if len(coll_names) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        axs_list = [ax]
    elif len(coll_names) == 2:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs_list = [axs[0], axs[1]]
    elif len(coll_names) == 4:
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs_list = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
    else:
        fig, axs = plt.subplots(3, 2, figsize=(12, 12))
        axs_list = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1], axs[2, 0], axs[2, 1]]

    for coll_name, ax in zip(coll_names, axs_list):
        plot_cdd_ax(ax=ax, paths=paths, tables=tables, coll_name=coll_name, alg_names=alg_names,
                    val_metric_name=val_metric_name, test_metric_name=test_metric_name, tag=tag,
                    use_validation_errors=use_validation_errors)

    name_parts = shorten_coll_names(coll_names)
    if use_validation_errors:
        name_parts = ['validation'] + name_parts
    if filename is None:
        file_path = paths.plots() / f'cdd_{"_".join(name_parts)}.pdf'
    else:
        file_path = paths.plots() / filename

    utils.ensureDir(file_path)
    plt.savefig(file_path, bbox_inches='tight')
    plt.close(fig)
    plt.rcParams['text.usetex'] = old_value
