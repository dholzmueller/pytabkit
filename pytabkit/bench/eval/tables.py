from typing import List, Optional

import numpy as np

from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskCollection
from pytabkit.bench.eval.analysis import ResultsTables, get_benchmark_results, get_opt_groups, get_simplified_name, \
    get_display_name
from pytabkit.bench.eval.evaluation import TaskWeighting, FunctionAlgFilter, MultiResultsTable, DefaultEvalModeSelector
from pytabkit.models import utils
from pytabkit.models.data.data import TaskType
from pytabkit.models.data.nested_dict import NestedDict


def _get_table_str(table_head: List[List[str]], table_body: List[List[str]]):
    head_row_strs = [' & '.join(row) + r' \\' for row in table_head]
    body_row_strs = [' & '.join(row) + r' \\' for row in table_body]
    n_cols = max(len(row) for row in table_head + table_body)
    begin_table_str = r'\begin{tabular}{' + ('c' * n_cols) + r'}' + '\n' + r'\toprule'
    end_table_str = r'\bottomrule' + '\n' + r'\end{tabular}'
    all_row_strs = [begin_table_str] + head_row_strs + [r'\midrule'] + body_row_strs + [end_table_str]
    complete_str = '\n'.join(all_row_strs)
    return complete_str


def generate_ds_table(paths: Paths, task_collection: TaskCollection, include_openml_ids: bool = False):
    print(f'Generating dataset table for {task_collection.coll_name}')
    task_infos = task_collection.load_infos(paths)
    task_infos.sort(key=lambda ti: ti.task_desc.task_name)
    file_path = paths.plots() / f'datasets_{task_collection.coll_name}.tex'

    is_classification = any(ti.task_type == TaskType.CLASSIFICATION for ti in task_infos)

    # columns to include: name, n_samples, n_numerical, n_categorical, largest_category, openml id,
    # (link), (subsampled), (n_classes), (citation), (weight)
    table_rows = [['Name', r'\#samples', r'\#num.\ features', r'\#cat.\ features', r'largest \#categories']]
    if is_classification:
        table_rows[0].append(r'\#classes')
    if include_openml_ids:
        table_rows[0].append('OpenML task ID')
    for task_info in task_infos:
        row = []
        row.append(task_info.task_desc.task_name.replace('_', r'\_'))
        row.append(str(task_info.n_samples))
        row.append(str(task_info.tensor_infos['x_cont'].get_n_features()))
        n_cat = task_info.tensor_infos['x_cat'].get_n_features()
        row.append(str(n_cat))
        # subtract 1 for the missing class
        row.append(str(task_info.tensor_infos['x_cat'].get_cat_sizes().max().item() - 1) if n_cat > 0 else '')
        if is_classification:
            row.append(str(task_info.tensor_infos['y'].get_cat_size_product()))
        if include_openml_ids:
            row.append(str(task_info.more_info_dict.get('openml_task_id', '')))
        table_rows.append(row)

    begin_table_str = r'\begin{tabular}{' + ('c' * len(table_rows[0])) + r'}' + '\n' + r'\toprule'
    row_strs = [' & '.join(row) + r' \\' for row in table_rows]
    end_table_str = r'\bottomrule' + '\n' + r'\end{tabular}'
    all_row_strs = [begin_table_str, row_strs[0], r'\midrule'] + row_strs[1:] + [end_table_str]
    complete_str = '\n'.join(all_row_strs)
    utils.writeToFile(file_path, complete_str)


def generate_collections_table(paths: Paths):
    print(f'Creating collections table')
    coll_display_names = {'meta-train-class': r'$\mathcal{B}^{\operatorname{train}}_{\mathrm{class}}$',
                          'meta-test-class': r'$\mathcal{B}^{\operatorname{test}}_{\mathrm{class}}$',
                          'grinsztajn-class-filtered': r'$\mathcal{B}^{\operatorname{Grinsztajn}}_{\mathrm{class}}$',
                          'meta-train-reg': r'$\mathcal{B}^{\operatorname{train}}_{\mathrm{reg}}$',
                          'meta-test-reg': r'$\mathcal{B}^{\operatorname{test}}_{\mathrm{reg}}$',
                          'grinsztajn-reg': r'$\mathcal{B}^{\operatorname{Grinsztajn}}_{\mathrm{reg}}$'}
    coll_names = list(coll_display_names.keys())

    # todo: number of distinct data sets
    rows = [r'\#datasets', r'\#dataset groups', r'min \#samples', r'max \#samples', r'max \#classes', r'max \#features',
            r'max \#categories']

    table_columns = {'': rows}

    for coll_name in coll_names:
        task_collection = TaskCollection.from_name(coll_name, paths)
        task_infos = task_collection.load_infos(paths)
        task_infos.sort(key=lambda ti: ti.task_desc.task_name)

        is_classification = any(ti.task_type == TaskType.CLASSIFICATION for ti in task_infos)

        n_samples_list = []
        n_features_list = []
        max_cat_size_list = []
        n_classes_list = []

        for task_info in task_infos:
            n_samples_list.append(task_info.n_samples)
            n_features_list.append(task_info.tensor_infos['x_cont'].get_n_features()
                                   + task_info.tensor_infos['x_cat'].get_n_features())
            n_cat = task_info.tensor_infos['x_cat'].get_n_features()
            # subtract 1 for the missing class
            max_cat_size_list.append(
                task_info.tensor_infos['x_cat'].get_cat_sizes().max().item() - 1 if n_cat > 0 else 0)
            if is_classification:
                n_classes_list.append(task_info.tensor_infos['y'].get_cat_size_product())
            else:
                n_classes_list.append(0)

        separate_task_names = ['facebook_comment_volume', 'facebook_live_sellers_thailand_shares']
        if coll_name.startswith('meta-train'):
            n_dataset_groups = TaskWeighting(task_infos, separate_task_names).get_n_groups()
        else:
            n_dataset_groups = len(task_infos)

        table_columns[coll_display_names[coll_name]] = \
            [str(len(task_infos)), str(n_dataset_groups), str(min(n_samples_list)), str(max(n_samples_list)),
             str(max(n_classes_list)), str(max(n_features_list)), str(max(max_cat_size_list))]

    keys = list(table_columns.keys())
    n_info_rows = len(table_columns[keys[0]])
    table_rows = [keys] + [[table_columns[key][i] for key in keys] for i in range(n_info_rows)]

    begin_table_str = r'\begin{tabular}{' + ('c' * len(table_rows[0])) + r'}' + '\n' + r'\toprule'
    row_strs = [' & '.join(row) + r' \\' for row in table_rows]
    end_table_str = r'\bottomrule' + '\n' + r'\end{tabular}'
    all_row_strs = [begin_table_str, row_strs[0], r'\midrule'] + row_strs[1:] + [end_table_str]
    complete_str = '\n'.join(all_row_strs)
    file_path = paths.plots() / f'collections_summary.tex'
    utils.writeToFile(file_path, complete_str)


def generate_individual_results_table(paths: Paths, tables: ResultsTables, filename: str, coll_name: str,
                                      alg_names: List[str],
                                      test_metric_name: Optional[str] = None,
                                      val_metric_name: Optional[str] = None):

    table = tables.get(coll_name)

    means, intervals = get_benchmark_results(paths, table, coll_name=coll_name, use_relative_score=False,
                                             test_metric_name=test_metric_name, val_metric_name=val_metric_name,
                                             return_percentages=False, use_task_mean=False, use_geometric_mean=False)

    alg_names = [an for an in alg_names if an in means]

    table_head = [['Dataset'] + [get_display_name(an) for an in alg_names]]
    table_body = []

    enumerated_task_infos = list(enumerate(table.test_table.task_infos))
    enumerated_task_infos.sort(key=lambda tup: tup[1].task_desc.task_name.lower())

    print(f'{coll_name=}')
    print(f'{list(means.keys())=}')

    for task_idx, task_info in enumerated_task_infos:
        row_scores = [means[alg_name][task_idx] for alg_name in alg_names]
        row_errs = [means[alg_name][task_idx] - intervals[alg_name][0][task_idx] for alg_name in alg_names]
        min_row_score = np.min(row_scores)
        is_best_list = [score == min_row_score for score in row_scores]
        is_significant_list = [score <= min_row_score + stderr for score, stderr in zip(row_scores, row_errs)]
        row_strs = []
        for is_best, is_significant, row_score, row_err in zip(is_best_list, is_significant_list, row_scores, row_errs):
            cur_str = f'{row_score:4.3f}'
            if is_best:
                cur_str = r'\textbf{' + cur_str + r'}'
            elif is_significant:
                cur_str = r'\underline{' + cur_str + r'}'

            cur_str = cur_str + r'$\pm$' + f'{row_err:4.3f}'
            row_strs.append(cur_str)

        table_body.append([task_info.task_desc.task_name] + row_strs)

    # escape underscores for latex
    table_head = [[val.replace('_', r'\_') for val in row] for row in table_head]
    table_body = [[val.replace('_', r'\_') for val in row] for row in table_body]

    table_str = _get_table_str(table_head, table_body)
    file_path = paths.plots() / filename
    utils.writeToFile(file_path, table_str)


def generate_ablations_table(paths: Paths, tables: ResultsTables):
    print(f'Generating ablations table')
    # load results from the right tag (maybe with MLP-best-ablation)
    # problem: relative model should be the best one of the defaults (with best lr)
    # group by and optimize lrfactor
    coll_names = ['meta-train-class', 'meta-train-reg']
    # all_group_names = dict()
    # all_best_lrfactors = dict()

    abl_names = [
        (r'MLP-TD (without ablation)', 'default'),
        # (r'MLP-TD (fixed lr factor = 1.0)', 'default_lrfactor-1.0'),
        ('', ''),
        (r'Num.\ embeddings: PL', 'num-embeddings-pl'),
        (r'Num.\ embeddings: PLR', 'num-embeddings-plr'),
        (r'Num.\ embeddings: None', 'num-embeddings-none'),
        ('', ''),
        (r'Adam $\beta_2=0.999$ instead of $\beta_2=0.95$', 'beta2-0.999'),
        ('', ''),
        ('Learning rate schedule = cosine decay', 'lr-cos-decay'),
        ('Learning rate schedule = constant', 'lr-constant'),
        ('', ''),
        ('No label smoothing', 'no-label-smoothing'),
        ('', ''),
        (r'No learnable scaling', 'no-front-scale'),
        ('', ''),
        ('Non-parametric activation', 'non-parametric-act'),
        ('', ''),
        (r'Activation=Mish', 'act-mish'),
        (r'Activation=ReLU', 'act-relu'),
        (r'Activation=SELU', 'act-selu'),
        ('', ''),
        ('No dropout', 'pdrop-0.0'),
        ('Dropout prob.\ $0.15$ (constant)', 'pdrop-0.15'),
        ('', ''),
        ('No weight decay', 'wd-0.0'),
        # ('Weight decay = 0.02 ($\operatorname{flat\_cos}$)', 'wd-0.02-flatcos'),
        ('Weight decay = 0.02 (constant)', 'wd-0.02'),
        ('', ''),
        (r'Standard param + no weight decay', 'standard-param_no-wd'),
        ('', ''),
        ('No data-dependent init', 'normal-init'),
        ('', ''),
        ('First best epoch instead of last best', 'first-best-epoch'),
        ('', ''),
        ('Only one-hot encoding', 'no-cat-embs'),
        # ('First best epoch (fixed lr factor = 0.5)', 'first-best-epoch_lrfactor-0.5'),
    ]

    results_dict = NestedDict()  # index by [short_group_name][coll_name][property]
    # possible properties: 'score', 'lower', 'upper', 'best_lr_factor',

    for coll_name in coll_names:
        table = tables.get(coll_name, n_cv=1, tag='paper_mlp_ablations')
        results, _ = get_benchmark_results(paths, table=table, coll_name=coll_name, use_relative_score=False,
                                           return_percentages=False,
                                           simplify_name_fn=lambda x: x.replace(' [bag-1]', ''))
        default_keys = [key for key in results if 'default' in key]
        # print(f'{default_keys=}')
        default_scores = [results[key] for key in default_keys]
        best_key = default_keys[np.argmin(default_scores)]
        rel_results, rel_intervals = get_benchmark_results(paths, table=table, coll_name=coll_name,
                                                           rel_alg_name=best_key,
                                                           simplify_name_fn=lambda x: x.replace(' [bag-1]', ''))
        keys = list(key for key in rel_results.keys() if key.startswith('RealMLP-TD-'))
        # keys = list(rel_results.keys())
        group_names = list(set([key.split('lrfactor-')[0] for key in keys]))
        # all_group_names[coll_name] = group_names
        for group_name in group_names:
            # remove the 'MLP-TD-reg-ablation_' and last '_'
            short_group_name = r'_'.join(group_name.split('_')[1:-1])
            group_keys = [key for key in keys if key.startswith(group_name)]
            group_results = [rel_results[key] for key in group_keys]
            best_key = group_keys[np.argmin(group_results)]
            # print(f'{best_key=}')
            results_dict[short_group_name, coll_name, 'best_lr_factor'] = best_key.split('lrfactor-')[1]
            results_dict[short_group_name, coll_name, 'score'] = rel_results[best_key]
            best_interval = rel_intervals[best_key]
            results_dict[short_group_name, coll_name, 'lower'] = best_interval[0]
            results_dict[short_group_name, coll_name, 'upper'] = best_interval[1]

        for key in keys:
            # also add non-optimized versions to the table
            # add default with default lr
            # short_group_name = 'default_lrfactor-1.0'
            # key = [key for key in rel_results.keys() if key.endswith('default_lrfactor-1.0')][0]
            short_group_name = '_'.join(key.split('_')[1:])
            results_dict[short_group_name, coll_name, 'best_lr_factor'] = ''
            results_dict[short_group_name, coll_name, 'score'] = rel_results[key]
            best_interval = rel_intervals[key]
            results_dict[short_group_name, coll_name, 'lower'] = best_interval[0]
            results_dict[short_group_name, coll_name, 'upper'] = best_interval[1]

        # all_best_lrfactors[coll_name] = best_lrfactors

    table_head = [[''] + [r'\multicolumn{2}{c}{' + coll_name + r'}' for coll_name in coll_names],
                  ['Ablation'] + [r'Error increase in \%', 'best lr factor'] * len(coll_names)]

    # all_group_names = sorted(list(results_dict.get_dict().keys()))
    table_body = []
    # for group_name in all_group_names:
    for label, short_group_name in abl_names:
        # short_group_name = r'\_'.join(group_name.split('_')[1:-1])
        row = [label]
        for coll_name in coll_names:
            if (short_group_name, coll_name, 'best_lr_factor') in results_dict:
                results = results_dict[short_group_name, coll_name]
                score = results['score']
                lower = results['lower']
                upper = results['upper']
                row.append(f'{score:2.1f} [{lower:2.1f}, {upper:2.1f}]')
                row.append(results['best_lr_factor'])
            else:
                row.append('')
                row.append('')
        table_body.append(row)

    table_str = _get_table_str(table_head, table_body)
    file_path = paths.plots() / 'ablations.tex'
    utils.writeToFile(file_path, table_str)


def generate_refit_table(paths: Paths, tables: ResultsTables, alg_family: str):
    print(f'Generating refit table for {alg_family}')
    coll_names = ['meta-train-class', 'meta-test-class', 'meta-train-reg', 'meta-test-reg']

    table_head = [['', r'\multicolumn{4}{c}{Error \textbf{reduction} relative to 1 fold in \%}'],
                  ['Method'] + coll_names]

    methods_labels_names = [
        (f' (bagging, 1 model, indiv. stopping)', f'_mean-cv-False_mean-refit-False [bag-1]'),
        (f' (bagging, 1 model, joint stopping)', f'_mean-cv-True_mean-refit-True [bag-1]'),
        (f' (bagging, 5 models, indiv. stopping)', f'_mean-cv-False_mean-refit-False [bag-5]'),
        (f' (bagging, 5 models, joint stopping)', f'_mean-cv-True_mean-refit-True [bag-5]'),
        (f' (refitting, 1 model, indiv. stopping)', f'_mean-cv-False_mean-refit-False [ens-1]'),
        (f' (refitting, 1 model, joint stopping)', f'_mean-cv-True_mean-refit-True [ens-1]'),
        (f' (refitting, 5 models, indiv. stopping)', f'_mean-cv-False_mean-refit-False [ens-5]'),
        (f' (refitting, 5 models, joint stopping)', f'_mean-cv-True_mean-refit-True [ens-5]')
    ]

    labels = [f'{alg_family}-TD{label_suffix}' for label_suffix, _ in methods_labels_names]
    table_body_columns = [labels[0:]]

    for coll_name in coll_names:
        column = []
        table = tables.get(coll_name, n_cv=5, tag='paper')
        # print(f'{table.test_table.alg_names=}')
        task_type_name = 'class' if 'class' in coll_name else 'reg'
        rel_alg_name = f'{alg_family}-TD-{task_type_name}_mean-cv-False_mean-refit-False [bag-1]'
        rel_results, rel_intervals = get_benchmark_results(paths, table=table, coll_name=coll_name,
                                                           rel_alg_name=rel_alg_name,
                                                           simplify_name_fn=lambda x: x)
        alg_names = [f'{alg_family}-TD-{task_type_name}{suffix}' for _, suffix in methods_labels_names]
        results_list = [rel_results[alg_name] for alg_name in alg_names]
        for alg_name in alg_names[0:]:
            result = rel_results[alg_name]
            lower, upper = rel_intervals[alg_name]
            is_best = (result == np.min(results_list))
            not_significantly_worse = (np.min(results_list) >= lower)
            result_str = f'{-result:2.1f}'
            if is_best:
                result_str = r'\textbf{' + result_str + r'}'
            elif not_significantly_worse:
                result_str = r'\underline{' + result_str + r'}'
            column.append(result_str + f' [{-upper:2.1f}, {-lower:2.1f}]')
        table_body_columns.append(column)

    table_body = utils.shift_dim_nested(table_body_columns, 0, 1)

    table_str = _get_table_str(table_head, table_body)
    file_path = paths.plots() / f'refit_table_{alg_family}.tex'
    utils.writeToFile(file_path, table_str)


def generate_preprocessing_table(paths: Paths, tables: ResultsTables):
    print(f'Generating preprocessing table')
    coll_names = ['meta-train-class', 'meta-train-reg']

    table_head = [['', r'\multicolumn{2}{c}{Error \textbf{increase} relative to robust scale + smooth clip in \%}'],
                  ['Method'] + coll_names]

    methods_labels_names = [
        (r'Robust scale + smooth clip', f'RealMLP-TD-S_tfms-mc-rs-sc-oh'),
        (r'Robust scale', f'RealMLP-TD-S_tfms-mc-rs-oh'),
        (r'Standardize + smooth clip', f'RealMLP-TD-S_tfms-std-sc-oh'),
        (r'Standardize', f'RealMLP-TD-S_tfms-std-oh'),
        (r'Quantile transform (output dist.\ = normal)', f'RealMLP-TD-S_tfms-quantile-oh'),
        (r'Quantile transform (RTDL version)', f'RealMLP-TD-S_tfms-quantiletabr-oh'),
        (r'KDI transform ($\alpha = 1$, output dist.\ = normal)', f'RealMLP-TD-S_tfms-kdi1-oh'),
    ]

    labels = [label for label, _ in methods_labels_names]
    table_body_columns = [labels]

    for coll_name in coll_names:
        column = []
        table = tables.get(coll_name, n_cv=1, tag='paper_preprocessing')
        # print(f'{table.test_table.alg_names=}')
        rel_alg_name = f'RealMLP-TD-S_tfms-mc-rs-sc-oh'
        rel_results, rel_intervals = get_benchmark_results(paths, table=table, coll_name=coll_name,
                                                           rel_alg_name=rel_alg_name)
        alg_names = [alg_name for _, alg_name in methods_labels_names]
        results_list = [rel_results[alg_name] for alg_name in alg_names]
        for alg_name in alg_names:
            result = rel_results[alg_name]
            lower, upper = rel_intervals[alg_name]
            is_best = (result == np.min(results_list))
            not_significantly_worse = (np.min(results_list) >= lower)
            result_str = f'{result:2.1f}'
            if is_best:
                result_str = r'\textbf{' + result_str + r'}'
            elif not_significantly_worse:
                result_str = r'\underline{' + result_str + r'}'
            column.append(result_str + f' [{lower:2.1f}, {upper:2.1f}]')
        table_body_columns.append(column)

    table_body = utils.shift_dim_nested(table_body_columns, 0, 1)

    table_str = _get_table_str(table_head, table_body)
    file_path = paths.plots() / f'preprocessing_ablation.tex'
    utils.writeToFile(file_path, table_str)


def generate_stopping_table(paths: Paths, tables: ResultsTables):
    print(f'Generating stopping table')
    coll_names = ['meta-train-class', 'meta-train-reg']

    table_head = [['', r'\multicolumn{2}{c}{Error \textbf{increase} relative to no early stopping in \%}'],
                  ['Method'] + coll_names]

    table_body = []

    for i, method in enumerate(['XGB-TD', 'LGBM-TD', 'CatBoost-TD']):
        esr_list = [1000, 300, 100, 50, 20, 10]
        labels = [method + f' (patience = {esr})' for esr in esr_list]
        table_body_columns = [labels]

        for coll_name in coll_names:
            column = []
            table = tables.get(coll_name, n_cv=1, tag='paper_early_stopping')
            # print(f'{table.test_table.alg_names=}')
            rel_alg_name = method + '_esr-1000'
            rel_results, rel_intervals = get_benchmark_results(paths, table=table, coll_name=coll_name,
                                                               rel_alg_name=rel_alg_name)
            alg_names = [method + f'_esr-{esr}' for esr in esr_list]
            results_list = [rel_results[alg_name] for alg_name in alg_names]
            for alg_name in alg_names:
                result = rel_results[alg_name]
                lower, upper = rel_intervals[alg_name]
                is_best = (result == np.min(results_list))
                result_str = f'{result:2.1f}'
                if is_best:
                    result_str = r'\textbf{' + result_str + r'}'
                column.append(result_str + f' [{lower:2.1f}, {upper:2.1f}]')
            table_body_columns.append(column)

        new_rows = utils.shift_dim_nested(table_body_columns, 0, 1)

        if i > 0:
            new_rows[0][0] = r'\midrule' + '\n' + new_rows[0][0]

        table_body.extend(new_rows)

    table_str = _get_table_str(table_head, table_body)
    file_path = paths.plots() / f'early_stopping_table.tex'
    utils.writeToFile(file_path, table_str)


def generate_architecture_table(paths: Paths, tables: ResultsTables):
    print(f'Generating architecture table')
    coll_names = ['meta-train-class', 'meta-train-reg', 'meta-test-class', 'meta-test-reg']

    table_head = [['', r'\multicolumn{4}{c}{Error \textbf{reduction} relative to MLP-D in \%}'],
                  ['Method'] + coll_names]

    methods_labels_names = [
        (r'MLP-D', f'MLP-RTDL-D'),
        (r'MLP-D (RS+SC)', f'MLP-RTDL-D_rssc'),
        (r'MLP-D (RS+SC, no wd, meta-tuned lr)', f'MLP-RTDL-reprod'),
        (r'MLP-D (RS+SC, no wd, meta-tuned lr, PL embeddings)', f'MLP-RTDL-reprod-pl'),
        (r'MLP-D (RS+SC, no wd, meta-tuned lr, RealMLP architecture)', f'MLP-RTDL-reprod-RealMLP-arch'),
        (r'RealMLP-TD-S', f'RealMLP-TD-S'),
        (r'RealMLP-TD', f'RealMLP-TD'),
        (r'TabR-S-D', f'TabR-S-D'),
        (r'TabR-S-D (RS+SC)', f'TabR-S-D_rssc'),
        (r'ResNet-D', f'ResNet-RTDL-D'),
        (r'ResNet-D (RS+SC)', f'ResNet-RTDL-D_rssc'),
    ]

    labels = [label for label, _ in methods_labels_names]
    table_body_columns = [labels]

    for coll_name in coll_names:
        column = []
        table = tables.get(coll_name, n_cv=1, tag='paper')
        # print(f'{table.test_table.alg_names=}')
        rel_alg_name = f'MLP-RTDL-D'
        rel_results, rel_intervals = get_benchmark_results(paths, table=table, coll_name=coll_name,
                                                           rel_alg_name=rel_alg_name)
        alg_names = [alg_name for _, alg_name in methods_labels_names]
        results_list = [rel_results[alg_name] for alg_name in alg_names]
        for alg_name in alg_names:
            result = rel_results[alg_name]
            lower, upper = rel_intervals[alg_name]
            is_best = (result == np.min(results_list))
            not_significantly_worse = (np.min(results_list) >= lower)

            # flip sign
            result = -result
            lower, upper = -upper, -lower
            result_str = f'{result:2.1f}'
            if is_best:
                result_str = r'\textbf{' + result_str + r'}'
            elif not_significantly_worse:
                result_str = r'\underline{' + result_str + r'}'
            column.append(result_str + f' [{lower:2.1f}, {upper:2.1f}]')
        table_body_columns.append(column)

    table_body = utils.shift_dim_nested(table_body_columns, 0, 1)

    table_str = _get_table_str(table_head, table_body)
    table_str = table_str.replace('ccccc', 'lcccc')  # make first column left-aligned
    file_path = paths.plots() / f'arch_and_preprocessing.tex'
    utils.writeToFile(file_path, table_str)
