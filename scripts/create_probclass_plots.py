from typing import Optional, List

import numpy as np
import pandas as pd
import torch
from adjustText import adjust_text

from tueplots import bundles, fonts, fontsizes, figsizes
import matplotlib

matplotlib.rcParams.update(bundles.icml2024())
matplotlib.rcParams.update(fonts.icml2024_tex())
matplotlib.rcParams.update(fontsizes.icml2024())

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects

import seaborn as sns

from pytabkit.bench.data.common import SplitType
from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskCollection
from pytabkit.bench.eval.analysis import ResultsTables, get_benchmark_results
from pytabkit.bench.run.results import ResultManager
from pytabkit.models import utils


def load_stopping_times(paths: Paths, alg_name: str, n_cv: int, n_tt_splits: int, val_metric_name: str,
                        coll_name: str = 'talent-class-small') -> np.ndarray:
    results = []
    coll = TaskCollection.from_name(coll_name, paths)
    for task_desc in coll.task_descs:
        for split_id in range(n_tt_splits):
            results_path = paths.results_alg_task_split(task_desc, alg_name, n_cv=n_cv, split_type=SplitType.RANDOM,
                                                        split_id=split_id)
            rm = ResultManager.load(results_path, load_other=True, load_preds=False)
            fit_params = rm.other_dict['cv']['fit_params']

            while True:
                if 'sub_fit_params' in fit_params:
                    fit_params = fit_params['sub_fit_params']
                elif isinstance(fit_params, list):
                    assert len(fit_params) == 1
                    fit_params = fit_params[0]
                else:
                    break

            result = None
            if 'stop_epoch' in fit_params:
                result = fit_params['stop_epoch']
            elif 'n_estimators' in fit_params:
                result = fit_params['n_estimators']
            else:
                print(f'No stopping epoch found in {fit_params=}')

            if isinstance(result, dict):
                result = result[val_metric_name]
            results.append(result)

    return np.asarray(results)


def plot_barscatter_ax(ax: plt.Axes, df: pd.DataFrame, xlabel: Optional[str], ylabel: str,
                       threshold: Optional[float] = None, use_symlog: bool = False):
    # hues = list(cal_methods.values())
    hues = df['hue'].unique().tolist()

    # adapted from https://cduvallet.github.io/posts/2018/03/boxplots-in-python

    sns.set_style('white')

    # colors = ['#B25116', '#FB84D1']
    # colors = ['tab:blue', 'tab:orange']
    colors = [(0.6, 0.8, 1.0), (1.0, 0.8, 0.6), (0.6, 1.0, 0.8)]

    if len(hues) == 1:
        if 'XGB' in hues[0]:
            colors = colors[2:3]
        elif hues[0].startswith('MLP'):
            colors = colors[1:2]
    pal = {key: value for key, value in zip(hues, colors[:len(hues)])}

    # Set up another palette for the boxplots, with slightly lighter shades
    # light_colors = ['#E5B699', '#FFC9EC']
    light_colors = colors
    face_pal = {key: value for key, value in zip(hues, light_colors[:len(hues)])}

    hue_order = hues

    # Make sure to remove the 'facecolor': 'w' property here, otherwise
    # the palette gets overrided
    boxprops = {'edgecolor': 'k', 'linewidth': 1}
    lineprops = {'color': 'k', 'linewidth': 1}

    boxplot_kwargs = {'boxprops': boxprops, 'medianprops': lineprops,
                      'whiskerprops': lineprops, 'capprops': lineprops,
                      'width': 0.75, 'palette': face_pal,
                      'whis': (10, 90),  # use 10% and 90% quantiles for whiskers
                      'hue_order': hue_order}

    stripplot_kwargs = {'linewidth': 0.4, 'size': 2.5, 'alpha': 0.6,
                        'palette': pal, 'hue_order': hue_order}

    ax.axhline(y=0, color='#888888', linestyle='--')
    ax.grid(True, which='both')

    sns.boxplot(x='label', y='value', hue='hue', data=df, ax=ax,
                fliersize=0, **boxplot_kwargs)
    sns.stripplot(x='label', y='value', hue='hue', data=df, ax=ax,
                  dodge=True, jitter=0.18, **stripplot_kwargs)

    if threshold is not None:
        ax.set_ylim(-threshold, threshold)

    if use_symlog:
        ax.set_yscale('symlog', linthresh=1)
        ax.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.ticklabel_format(style='plain', axis='y')

    print(f'{len(hues)=}')

    # Fix the legend, keep only the first len(hues) legend elements
    # (there would be twice as many because there are also the ones for the scatter plot
    if len(hues) > 1:
        handles, hues_ax = ax.get_legend_handles_labels()

        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
        #           fancybox=True, shadow=True, ncol=5)
        lgd = ax.legend(handles[:len(hues)], hues_ax[:len(hues)],
                        loc='upper center',
                        bbox_to_anchor=(0.5, -0.25 if xlabel is not None else -0.15),
                        ncol=len(hues),
                        # fancybox=True, shadow=True
                        # fontsize='large',
                        # handletextpad=0.5,
                        )
        # lgd.legend_handles[0]._sizes = [40]
        # lgd.legend_handles[1]._sizes = [40]
    else:
        ax.get_legend().remove()

    ax.set_ylabel(ylabel, fontsize='small')
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize='small')
    else:
        ax.set_xlabel('', fontsize='small')


def plot_results(paths: Paths, tables: ResultsTables, base_names: List[str], n_hpo_steps: int,
                 n_tt_splits: int, coll_name: str = 'talent-class-small',
                 metric_name: str = 'n_cross_entropy', use_mean_results: bool = False,
                 use_percentages: bool = False,
                 plot_stopping_times: bool = False,
                 n_cv: int = 1, threshold: Optional[float] = 0.02, use_validation_errors: bool = False,
                 use_small_plot: bool = False, use_medium_plot: bool = False, title: Optional[str] = None):
    val_metrics = {'cross_entropy': 'Logloss', '1-auroc-ovr': 'AUROC', 'brier': 'Brier', 'ref-ll-ts': 'TS-Ref.',
                   'ref-br-ts': 'Brier-ref.', 'class_error': 'Accuracy'}

    cal_methods = {'': 'No post-hoc cal.', '_ts-mix': 'Temp. scaling'}

    metric_display_name_dict = {'n_cross_entropy': 'normalized Logloss', 'cross_entropy': 'Logloss',
                                'n_brier': 'normalized Brier loss', 'brier': 'Brier loss', 'class_error': 'Class. err.',
                                '1-auroc-ovr': '1-AUROC'}

    metric_display_name = metric_display_name_dict.get(metric_name, metric_name)

    # assert use_small_plot or all(len(bn) == 1 for bn in base_names)
    assert use_small_plot or use_medium_plot or len(base_names) == 1

    with (plt.rc_context(figsizes.icml2024_half() if use_small_plot else figsizes.icml2024_full())):
        # fig, axs = plt.subplots(1, len(base_names))
        fig, ax = plt.subplots()

        dfs = []

        for base_name in base_names:

            cv_suffix = '' if n_cv == 1 else f'-cv{n_cv}'
            bag_suffix = f' [bag-{n_cv}]'
            hpo_steps_suffix = f'-{n_hpo_steps}' if 'HPO' in base_name else ''

            means = None

            if not plot_stopping_times:
                if use_mean_results:
                    means_dicts = []
                    for tag in [f'paper_hpo_{base_name}{cv_suffix}', f'paper_hpo-calib_{base_name}{cv_suffix}']:
                        table = tables.get(coll_name, tag=tag, n_cv=n_cv)

                        means, intervals = get_benchmark_results(paths, table, coll_name=coll_name,
                                                                 use_relative_score=False,
                                                                 test_metric_name=metric_name,
                                                                 val_metric_name=metric_name,
                                                                 n_splits=n_tt_splits,
                                                                 # don't replace '-class' because it occurs in val-class_error
                                                                 # also don't replace ' [bag-1]' for the cv case
                                                                 simplify_name_fn=lambda s: s,
                                                                 return_percentages=False, use_task_mean=False,
                                                                 use_validation_errors=use_validation_errors,
                                                                 use_geometric_mean=False)
                        means_dicts.append(means)

                    all_means = utils.join_dicts(*means_dicts)
                    means = dict()

                    print(f'Available alg names before aggregating:')
                    for alg_name in all_means:
                        print(alg_name)
                    print()

                    for val_metric_key, val_metric_label in val_metrics.items():
                        for cal_method_key, cal_method_name in cal_methods.items():
                            alg_name = f'{base_name}{cv_suffix}-{n_hpo_steps}_val-{val_metric_key}{cal_method_key}{bag_suffix}'
                            alg_names_source = [
                                f'{base_name}{cv_suffix}_step-{i}_val-{val_metric_key}{cal_method_key}{bag_suffix}' for
                                i in
                                range(n_hpo_steps)]
                            means[alg_name] = np.mean(np.stack([all_means[an] for an in alg_names_source], axis=1),
                                                      axis=1)

                else:
                    table = tables.get(coll_name,
                                       tag=f'paper_{base_name}{cv_suffix}' if n_cv == 1 else f'paper{cv_suffix}',
                                       n_cv=n_cv)

                    means, intervals = get_benchmark_results(paths, table, coll_name=coll_name,
                                                             use_relative_score=False,
                                                             test_metric_name=metric_name, val_metric_name=metric_name,
                                                             n_splits=n_tt_splits,
                                                             # don't replace '-class' because it occurs in val-class_error
                                                             # also don't replace ' [bag-1]' for the cv case
                                                             use_validation_errors=use_validation_errors,
                                                             simplify_name_fn=lambda s: s,
                                                             return_percentages=False, use_task_mean=False,
                                                             use_geometric_mean=False)

            # df should contain columns 'value', 'val_metric', 'cal_method'
            alg_dfs = []

            if means is not None:
                print(f'Available alg names:')
                for alg_name in means:
                    print(alg_name)
                print()

            rel_alg = f'{base_name}{cv_suffix}{hpo_steps_suffix}_val-cross_entropy_ts-mix{bag_suffix}'

            if use_small_plot:
                if plot_stopping_times:
                    combinations = [
                        ('cross_entropy', '', 'Logloss'),
                        ('brier', '', 'Brier'),
                        ('1-auroc-ovr', '', 'AUROC'),
                        ('ref-ll-ts', '', 'TS-Ref.'),
                        ('ref-br-ts', '', 'Brier-Ref.'),
                        ('class_error', '', 'Accuracy'),
                    ]
                else:
                    combinations = [
                        ('cross_entropy', '', 'Logloss'),
                        ('cross_entropy', '_ts-mix', 'Logloss+TS'),
                        ('ref-ll-ts', '_ts-mix', 'TS-Ref.+TS'),
                        ('class_error', '_ts-mix', 'Accuracy+TS'),
                    ]
            elif use_medium_plot:
                combinations = [
                    ('cross_entropy', '', 'Logloss'),
                    ('cross_entropy', '_ts-mix', 'Logloss+TS'),
                    ('brier', '_ts-mix', 'Brier+TS'),
                    ('1-auroc-ovr', '_ts-mix', 'AUROC+TS'),
                    ('ref-ll-ts', '_ts-mix', 'TS-Ref.+TS'),
                    ('ref-br-ts', '_ts-mix', 'Brier-Ref.+TS'),
                    ('class_error', '_ts-mix', 'Accuracy+TS'),
                ]
                if not any('-HPO' in base_name for base_name in base_names):
                    combinations.insert(5, ('ref-ll-ts-cv5', '_ts-mix', 'TS-Ref.-5CV+TS'))
            else:
                combinations = [(val_metric_key, cal_method_key, val_metric_label) for val_metric_key, val_metric_label
                                in val_metrics.items() for cal_method_key in cal_methods]

            for val_metric_key, cal_method_key, label in combinations:
                alg_name = f'{base_name}{cv_suffix}{hpo_steps_suffix}_val-{val_metric_key}{cal_method_key}'
                print(f'Adding results for {alg_name}')
                if plot_stopping_times:
                    assert not use_mean_results
                    values = load_stopping_times(paths, alg_name=alg_name, n_cv=n_cv, n_tt_splits=n_tt_splits,
                                                 val_metric_name=val_metric_key,
                                                 coll_name=coll_name)
                else:
                    if use_percentages:
                        values = 100 * (means[alg_name + bag_suffix] / means[rel_alg] - 1)
                    else:
                        values = means[alg_name + bag_suffix] - means[rel_alg]

                    if threshold is not None:
                        values = np.clip(values, -threshold, threshold)

                if use_small_plot or use_medium_plot:
                    hue = base_name.split('-')[0]
                    if hue == 'XGB':
                        hue = 'XGBoost'
                else:
                    hue = cal_methods[cal_method_key]

                alg_dfs.append(pd.DataFrame(dict(
                    value=values.tolist(),
                    label=[label] * len(values),
                    hue=[hue] * len(values),
                )))

            df = pd.concat(alg_dfs, axis='index', ignore_index=True)
            dfs.append(df)

        df = pd.concat(dfs, axis='index', ignore_index=True)

        ylabel = ('Stopping iteration' if 'XGB' in base_name else f'Stopping epoch') \
            if plot_stopping_times else f'{metric_display_name} diff.\\ to baseline'
        if use_percentages:
            ylabel = ylabel + r' [\%]'
        plot_barscatter_ax(ax=ax, df=df, xlabel=None,  # 'Validation and optimization metric',
                           ylabel=ylabel, use_symlog=use_percentages,
                           threshold=threshold if plot_stopping_times else None)

        if title:
            ax.set_title(title)

        suffix = '_mean' if use_mean_results else ''
        suffix = suffix + ('_rel' if use_percentages else '')
        suffix = suffix + ('_stoptime' if plot_stopping_times else '')
        suffix = suffix + ('_valid' if use_validation_errors else '')
        suffix = suffix + ('' if coll_name == 'talent-class-small' else '_' + coll_name)
        suffix = suffix + ('_small' if use_small_plot else ('_medium' if use_medium_plot else ''))

        threshold_str = f'None' if threshold is None else f'{threshold:g}'
        file_path = paths.plots() / f'boxplot_{"-".join(base_names)}{cv_suffix}_{metric_name}_{threshold_str}{suffix}.pdf'

        plt.tight_layout()
        utils.ensureDir(file_path)
        plt.savefig(file_path)
        plt.close()


def plot_calib_benchmark(paths: Paths, tables: ResultsTables, metric_name: str = 'cross_entropy',
                         n_tt_splits: int = 5, use_validation_errors: bool = False, use_extra_methods: bool = False):
    times_df = pd.read_csv(paths.base() / 'calib_times' / 'times.csv')
    methods = list(times_df['calib_name'].unique())

    coll_name = 'talent-class-small'
    table = tables.get(coll_name, tag=f'paper_calib-bench', n_cv=1)

    means, _ = get_benchmark_results(paths, table, coll_name=coll_name, use_relative_score=False,
                                     test_metric_name=metric_name, val_metric_name=metric_name,
                                     n_splits=n_tt_splits,
                                     # don't replace '-class' because it occurs in val-class_error
                                     # also don't replace ' [bag-1]' for the cv case
                                     simplify_name_fn=lambda s: s,
                                     return_percentages=False, use_task_mean=True,
                                     use_validation_errors=use_validation_errors,
                                     use_geometric_mean=False)

    # ----- get reference score without post-hoc calibration

    means_nocalib, _ = get_benchmark_results(paths, tables.get(coll_name, tag=f'paper_XGB-D', n_cv=1),
                                             coll_name=coll_name, use_relative_score=False,
                                             test_metric_name=metric_name, val_metric_name=metric_name,
                                             n_splits=n_tt_splits,
                                             # don't replace '-class' because it occurs in val-class_error
                                             # also don't replace ' [bag-1]' for the cv case
                                             simplify_name_fn=lambda s: s,
                                             return_percentages=False, use_task_mean=True,
                                             use_validation_errors=use_validation_errors,
                                             use_geometric_mean=False)

    orig_score = means_nocalib['XGB-D_val-class_error [bag-1]']

    avg_times = dict()

    min_n_val = 10_000
    df = times_df.loc[times_df['n_val'] >= min_n_val]
    for method in methods:
        where = df['calib_name'] == method
        # * 1000 for per 1K, *1000 for milliseconds
        avg_times[method] = np.mean(df.loc[where, 'time'] / df.loc[where, 'n_val']) * 1_000_000

    print(repr(means))
    print(repr(avg_times))

    val_metrics = {'cross_entropy': 'Logloss', '1-auroc-ovr': 'AUROC', 'brier': 'Brier', 'ref-ll-ts': 'TS-Ref.',
                   'n_cross_entropy': 'norm. Logloss', 'n_brier': 'norm. Brier',
                   'ref-br-ts': 'Brier-ref.', 'class_error': 'Accuracy'}

    methods_with_labels = {'ts': r'TS (ours)',
                           # 'ts-mix': r'Bisection + smoothing (ours)',
                           'ag-ts': r'TS (AutoGluon)',
                           # 'ag-inv-ts': r'AutoGluon + inv. temp.',
                           'torchunc-ts': 'TS (TorchUncertainty)',
                           'guo-ts': 'TS (Guo et al., 2017)',
                           }

    if use_extra_methods:
        methods_with_labels = utils.join_dicts(methods_with_labels, {
            'ir-mix': 'Isotonic (sklearn) + LS',
            'ts-mix': 'TS+LS (ours)'
        })

    labels_list = list(methods_with_labels.values())

    with plt.rc_context(figsizes.icml2024_half(height_to_width_ratio=0.5 if use_extra_methods else 0.4)):
        fig, ax = plt.subplots()

        # sns.set_theme(style="whitegrid", font_scale=2)

        plt.ylabel(f'Mean {val_metrics[metric_name]}')
        plt.xlabel(f'Mean runtime (ms) per 1K samples')

        colors = ['tab:green', 'tab:blue', 'tab:orange', 'tab:red', 'tab:purple', 'tab:cyan']

        lines = []

        lines.append(
            ax.axhline(y=means['XGB-D_val-class_error_calib-bench_ts [bag-1]'], color=colors[0], linestyle='--',
                       linewidth=1.0,
                       zorder=-50))

        times_list = [avg_times[method] for method in methods_with_labels.keys()]
        metrics_list = [means[f'XGB-D_val-class_error_calib-bench_{method} [bag-1]'] for method in
                        methods_with_labels.keys()]

        plt.scatter(times_list, metrics_list, c=colors[:len(times_list)], s=10)

        # Prepare to annotate the points
        texts = []
        for i, point in enumerate(ax.collections[0].get_offsets()):
            model_name = labels_list[i]
            x, y = point
            if x < np.mean(times_list):
                # x = 0.7 * x + 0.3 * np.max(times_list)
                x += 0.15 * (np.max(times_list) - np.min(times_list))
            else:
                # x = 0.7 * x + 0.3 * np.min(times_list)
                x -= 0.15 * (np.max(times_list) - np.min(times_list))
            y = 0.8 * y + 0.2 * np.mean(metrics_list)
            text_color = colors[i]
            # Annotate the model names
            display_name = model_name
            # with plt.rc_context({'font.family': 'sans-serif', "font.sans-serif": "DejaVu Sans"}):
            # from matplotlib import font_manager
            # font_path = font_manager.findfont("DejaVu Sans")
            # print(f'{font_path=}')
            with plt.rc_context({'font.family': 'sans-serif', "text.usetex": False}):
                text = ax.text(x, y, display_name, color=text_color, fontsize=8, ha='center', va='center', font='Arial')
            # text.set_path_effects([matplotlib.patheffects.withStroke(linewidth=1.2, foreground='white')])
            texts.append(text)

        # import matplotlib.font_manager as fm
        # print([f.name for f in fm.fontManager.ttflist])

        lines.append(ax.axhline(y=orig_score, color='tab:gray', linestyle='--', linewidth=1.0,
                                zorder=-50))
        # with plt.rc_context({'font.family': 'sans-serif', "font.sans-serif": "DejaVu Sans"}):
        with plt.rc_context({'font.family': 'sans-serif', "text.usetex": False}):
            text = ax.text(np.mean(times_list), orig_score - 0.1 * (np.max(metrics_list) - np.min(metrics_list)),
                           'No post-hoc cal.', color='tab:gray', fontsize=8, ha='center', va='center', font='Arial')
        texts.append(text)

        plt.xlim(left=0)
        # plt.grid(True, which='both', zorder=-100)
        ax.set_axisbelow(True)

        print(ax.collections)

        # line = ax.axhline(y=means['XGB-D_val-class_error_calib-bench_ts [bag-1]']-0.01, color='white', linestyle='--',
        #                   linewidth=1.5,
        #                   zorder=-50)

        # Use adjust_text to repel the labels from each other and the points
        adjust_text(texts,
                    # force_text=(0.01, 0.02),
                    # objects=lines,
                    x=times_list,
                    y=metrics_list,
                    # force_pull=(0.1, 0.1),
                    # force_explode=(0.1, 0.2),
                    avoid_self=False,
                    expand=(1.15, 1.3),
                    ax=ax,
                    )

        if use_extra_methods:
            ymin, ymax = ax.get_ylim()
            ymin = ymin - 0.15 * (ymax - ymin)
            plt.ylim(ymin, ymax)

        suffix = '_extra' if use_extra_methods else ''
        filename = f'calib_benchmark_{coll_name}_{metric_name}{suffix}'
        if use_validation_errors:
            filename = filename + '_valid'
        filename = filename + '.pdf'

        file_path = paths.plots() / filename
        utils.ensureDir(file_path)

        plt.tight_layout()
        plt.savefig(file_path)

        plt.close(fig)


def plot_gap_vs_ds_size(paths: Paths, tables: ResultsTables, base_name: str, metric_name: str, n_hpo_steps: int,
                        use_smallest_class: bool = False, use_2nd_largest_class: bool = False,
                        use_entropy: bool = False, use_percentages: bool = False, color_by_total_loss: bool = False):
    table = tables.get('talent-class-small', tag=f'paper_{base_name}',
                       n_cv=1)
    coll_name = 'talent-class-small'
    task_infos = TaskCollection.from_name(coll_name, paths).load_infos(paths)

    means, intervals = get_benchmark_results(paths, table, coll_name='talent-class-small', use_relative_score=False,
                                             test_metric_name=metric_name, val_metric_name=metric_name,
                                             n_splits=5,
                                             # don't replace '-class' because it occurs in val-class_error
                                             # also don't replace ' [bag-1]' for the cv case
                                             use_validation_errors=False,
                                             simplify_name_fn=lambda s: s,
                                             return_percentages=False, use_task_mean=False,
                                             use_geometric_mean=False)

    print(f'Available alg names:')
    for alg_name in means:
        print(alg_name)

    extended_base_name = f'{base_name}-{n_hpo_steps}' if 'HPO' in base_name else base_name

    alg_name_1 = f'{extended_base_name}_val-cross_entropy_ts-mix [bag-1]'
    alg_name_2 = f'{extended_base_name}_val-ref-ll-ts_ts-mix [bag-1]'

    if use_percentages:
        diffs = 100 * (means[alg_name_2] / means[alg_name_1] - 1)
    else:
        diffs = means[alg_name_2] - means[alg_name_1]

    suffix = '_rel' if use_percentages else ''

    if use_smallest_class:
        suffix = suffix + '_smallest-class'
        x = []
        for task_info in task_infos:
            class_frequencies = torch.bincount(task_info.load_task(paths).ds.tensors['y'].squeeze(-1)).numpy()
            x.append(np.min(class_frequencies))
    elif use_2nd_largest_class:
        suffix = suffix + '_2nd-largest-class'
        x = []
        for task_info in task_infos:
            class_frequencies = torch.bincount(task_info.load_task(paths).ds.tensors['y'].squeeze(-1)).numpy()
            x.append(np.sort(class_frequencies)[-2])
    elif use_entropy:
        suffix = suffix + '_entropy'
        x = []
        for task_info in task_infos:
            class_frequencies = torch.bincount(task_info.load_task(paths).ds.tensors['y'].squeeze(-1)).numpy()
            class_probs = class_frequencies.astype(np.float32) / task_info.n_samples
            x.append(-task_info.n_samples * np.dot(class_probs, np.log2(class_probs + 1e-30)))
    else:
        x = [ti.n_samples for ti in task_infos]

    if color_by_total_loss:
        cbar_label = 'Sum of losses of both versions'
        suffix = suffix + '_col-loss'
        colors = means[alg_name_1] + means[alg_name_2]
    else:
        cbar_label = 'Total Entropy of Y'
        colors = []
        for task_info in task_infos:
            class_frequencies = torch.bincount(task_info.load_task(paths).ds.tensors['y'].squeeze(-1)).numpy().astype(
                np.float32)
            p = class_frequencies / np.sum(class_frequencies)
            entropy = -np.dot(p, np.log(p))
            colors.append(entropy)

    metric_display_names = {'cross_entropy': 'Logloss', '1-auroc-ovr': 'AUROC', 'brier': 'Brier',
                            'ref-ll-ts': 'TS-Ref.',
                            'n_cross_entropy': 'norm. Logloss', 'n_brier': 'norm. Brier',
                            'ref-br-ts': 'Brier-Ref.', 'class_error': 'Accuracy'}

    with (plt.rc_context(figsizes.icml2024_half(height_to_width_ratio=0.8))):
        with plt.rc_context(fontsizes.icml2024(default_smaller=0)):
            fig, ax = plt.subplots()

            norm = matplotlib.colors.LogNorm(vmin=np.min(colors), vmax=np.max(colors))
            cmap = plt.cm.plasma_r  # You can use other colormaps like 'plasma', 'coolwarm', etc.
            colors = cmap(norm(colors))

            # Plot with color based on z
            for i in range(len(x)):
                ax.plot(x[i], diffs[i], '.', color=colors[i])

            if use_percentages:
                ax.set_yscale('symlog', linthresh=1)

            method_display_name = base_name.replace('-HPO', ' (tuned)')
            method_display_name = method_display_name.replace('-TD', ' (default)')
            method_display_name = method_display_name.replace('-D', ' (default)')

            ax.set_title(r'\textbf{' + method_display_name + r'}')
            ax.set_xscale('log')

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])  # Set an empty array as required

            # Add the colorbar
            plt.colorbar(sm, label=cbar_label, ax=ax)
            # plt.semilogx(x, diffs, '.', color='tab:blue')
            plt.xlabel('Number of samples')
            plt.ylabel(f'Relative difference in {metric_display_names[metric_name]} [\\%]')
            plt.axhline(y=0, color='k', linestyle='--', zorder=-1)
            # plt.tight_layout()
            file_path = paths.plots() / f'gap_vs_ds_size_{base_name}_{metric_name}{suffix}.pdf'
            utils.ensureDir(file_path)
            plt.savefig(file_path)
            plt.close()


if __name__ == '__main__':
    paths = Paths.from_env_variables()
    tables = ResultsTables(paths)

    # calibration methods  (label positions can be improved post-hoc using Inkscape)
    for metric_name in ['cross_entropy', 'n_cross_entropy', 'brier', 'n_brier']:
        for use_extra_methods in [True, False]:
            plot_calib_benchmark(paths, tables, metric_name=metric_name, use_extra_methods=use_extra_methods)

    # results for individual datasets
    for base_name in ['MLP-HPO', 'XGB-HPO', 'RealMLP-HPO', 'MLP-D', 'XGB-D', 'RealMLP-TD']:
        plot_gap_vs_ds_size(paths, tables, base_name=base_name, metric_name='cross_entropy', n_hpo_steps=30,
                            use_smallest_class=False,
                            use_2nd_largest_class=False,
                            use_entropy=False, use_percentages=True,
                            color_by_total_loss=True)

    # plot main benchmark results
    for use_small_plot in [False, True]:
        for base_names in [['RealMLP-HPO', 'MLP-HPO', 'XGB-HPO'], ['RealMLP-TD', 'MLP-D', 'XGB-D']]:
            for coll_name in ['talent-class-small-above10k', 'talent-class-small']:
                for metric_name in ['cross_entropy', 'class_error', '1-auroc-ovr']:
                    plot_results(paths, tables, base_names, n_hpo_steps=30, n_tt_splits=5,
                                 use_percentages=True,
                                 metric_name=metric_name, coll_name=coll_name, use_validation_errors=False,
                                 use_small_plot=use_small_plot, use_medium_plot=not use_small_plot,
                                 use_mean_results=False, threshold=100, n_cv=1,
                                 title=r'\textbf{Tabular data, tuned hyperparameters}' if 'RealMLP-HPO' in base_names
                                 else r'\textbf{Tabular data, default hyperparameters}')

    # plot stopping times
    for base_names in [  # ['RealMLP-HPO'], ['MLP-HPO'], ['XGB-HPO'],
        ['RealMLP-TD'], ['MLP-D'], ['XGB-D']]:
        for coll_name in ['talent-class-small-above10k', 'talent-class-small']:
            plot_results(paths, tables, base_names, n_hpo_steps=30, n_tt_splits=5,
                         use_percentages=False,
                         metric_name='cross_entropy', coll_name=coll_name, use_validation_errors=False,
                         use_small_plot=True, use_medium_plot=False, use_mean_results=False, n_cv=1,
                         plot_stopping_times=True, threshold=None,
                         title=r'\textbf{Tabular data, tuned hyperparameters}' if any(
                             '-HPO' in base_name for base_name in base_names)
                         else r'\textbf{Tabular data, default hyperparameters}')
