from pytabkit.bench.data.paths import Paths
from pytabkit.bench.data.tasks import TaskCollection
from pytabkit.bench.eval.analysis import ResultsTables
from pytabkit.bench.eval.plotting import plot_schedule, plot_schedules, plot_benchmark_bars, plot_scatter, \
    plot_pareto, plot_winrates, plot_stopping, plot_cumulative_ablations, plot_cdd
from pytabkit.bench.eval.tables import generate_ds_table, generate_collections_table, generate_individual_results_table, \
    generate_ablations_table, generate_refit_table, generate_preprocessing_table, generate_stopping_table, \
    generate_architecture_table

if __name__ == '__main__':
    paths = Paths.from_env_variables()
    coll_names = ['meta-train-class', 'meta-train-reg', 'meta-test-class', 'meta-test-reg', 'grinsztajn-class-filtered',
                  'grinsztajn-reg']

    tables = ResultsTables(paths)

    arrow_alg_names = [('MLP-PLR-D', 'RealMLP-TD'), ('TabR-S-D', 'RealTabR-D'), ('XGB-D', 'XGB-TD'),
                       ('LGBM-D', 'LGBM-TD'), ('CatBoost-D', 'CatBoost-TD'), ('MLP-PLR-HPO', 'RealMLP-HPO')]

    alg_names = [f'{method}-{version}'
                 for method in ['XGB', 'LGBM', 'CatBoost', 'BestModel', 'Ensemble']
                 for version in ['D', 'TD', 'HPO']]
    alg_names.extend(['RealMLP-TD', 'RealMLP-TD-S', 'RealMLP-HPO', 'MLP-RTDL-D', 'MLP-RTDL-HPO',
                      'MLP-PLR-D', 'MLP-PLR-HPO', 'RealTabR-D', 'FTT-D', 'FTT-HPO',
                      'ResNet-RTDL-D', 'ResNet-RTDL-HPO', 'RF-SKL-D', 'RF-HPO', 'XGB-PBB-D', 'TabR-S-D', 'TabR-HPO'])

    alg_names_short = [f'{method}-{version}'
                       for method in ['XGB', 'LGBM', 'CatBoost']
                       for version in ['D', 'TD', 'HPO']]
    alg_names_short.extend(['RealMLP-TD', 'RealMLP-TD-S', 'RealMLP-HPO', 'MLP-RTDL-D', 'MLP-RTDL-HPO',
                            'MLP-PLR-D', 'MLP-PLR-HPO', 'FTT-D', 'FTT-HPO',
                            'ResNet-RTDL-D', 'ResNet-RTDL-HPO', 'RF-SKL-D', 'RF-HPO', 'XGB-PBB-D', 'TabR-S-D',
                            'RealTabR-D',
                            'TabR-HPO'])

    alg_names_hpo_vs_tpe = [f'{method}-{version}'
                            for method in ['XGB', 'LGBM', 'CatBoost']
                            for version in ['D', 'TD', 'HPO', 'HPO-TPE']]
    alg_names_hpo_vs_tpe.extend(['RealMLP-TD', 'RealMLP-HPO'])

    # extra plot for the README.md
    plot_pareto(paths, tables, coll_names=['meta-test-class', 'meta-test-reg'], alg_names=alg_names,
                use_ranks=False, use_normalized_errors=False,
                use_grinnorm_errors=False,
                use_geometric_mean=True, use_validation_errors=False, arrow_alg_names=arrow_alg_names)

    for use_ranks, use_normalized_errors, use_geometric_mean, use_grinnorm_errors in [[False, False, False, False],
                                                                                      [False, False, True, False],
                                                                                      [True, False, False, False],
                                                                                      [False, True, False, False],
                                                                                      [False, False, False, True]]:
        plot_pareto(paths, tables, coll_names=['grinsztajn-class-filtered', 'grinsztajn-reg'], alg_names=alg_names,
                    use_ranks=use_ranks, use_normalized_errors=use_normalized_errors,
                    use_grinnorm_errors=use_grinnorm_errors,
                    use_geometric_mean=use_geometric_mean, arrow_alg_names=arrow_alg_names)
        plot_pareto(paths, tables, coll_names=coll_names, alg_names=alg_names,
                    use_ranks=use_ranks, use_normalized_errors=use_normalized_errors,
                    use_grinnorm_errors=use_grinnorm_errors,
                    use_geometric_mean=use_geometric_mean, arrow_alg_names=arrow_alg_names)
        plot_pareto(paths, tables, coll_names=coll_names, alg_names=alg_names,
                    use_ranks=use_ranks, use_normalized_errors=use_normalized_errors,
                    use_grinnorm_errors=use_grinnorm_errors,
                    use_geometric_mean=use_geometric_mean, use_validation_errors=True, arrow_alg_names=arrow_alg_names)

    # alg_names_rssc = alg_names + ['MLP-RTDL-D_rssc', 'ResNet-RTDL-D_rssc', 'TabR-S-D_rssc']
    without_rssc = ['MLP-RTDL-D', 'ResNet-RTDL-D', 'TabR-S-D', 'FTT-D', 'MLP-PLR-D']
    alg_names_rssc = without_rssc + [an + '_rssc' for an in without_rssc] + ['BestModel_' + an + '_prep' for an in
                                                                             without_rssc]
    alg_names_rssc = alg_names_rssc + ['RealMLP-TD', 'RealTabR-D']
    # alg_names_rssc = alg_names_rssc + ['MLP-RTDL-HPO', 'ResNet-RTDL-HPO', 'FTT-D-HPO', 'MLP-PLR-HPO', 'TabR-HPO']

    plot_pareto(paths, tables, coll_names=coll_names, alg_names=alg_names_rssc,
                filename='pareto_rssc.pdf')
    # plot_pareto(paths, tables, coll_names=['meta-train-class', 'meta-train-reg'], alg_names=alg_names_rssc,
    #             filename='pareto_rssc_meta-train.pdf')
    # plot_pareto(paths, tables, coll_names=['meta-test-class', 'meta-test-reg'], alg_names=alg_names_rssc,
    #             filename='pareto_rssc_meta-test.pdf')

    plot_pareto(paths, tables, coll_names=['meta-train-class', 'meta-train-reg', 'meta-test-class', 'meta-test-reg'],
                alg_names=alg_names_hpo_vs_tpe, plot_pareto_frontier=False,
                use_ranks=False, use_normalized_errors=False,
                use_geometric_mean=True, filename='pareto_hpo-rs-vs-tpe.pdf')

    plot_pareto(paths, tables, coll_names=['meta-test-class-no-missing', 'meta-test-reg-no-missing'],
                alg_names=alg_names, arrow_alg_names=arrow_alg_names,
                filename='pareto_no-missing_geometric.pdf')

    alg_names_auc = [f'{method}-{version}'
                     for method in ['XGB', 'LGBM', 'CatBoost', 'BestModel']
                     for version in ['D', 'TD', 'HPO_best-1-auc-ovr']]
    alg_names_auc.extend(['RealMLP-TD', 'RealMLP-TD-S', 'RealMLP-HPO_best-1-auc-ovr',
                          'RealMLP-TD_no-ls', 'RealMLP-TD-S_no-ls',
                          'MLP-RTDL-D', 'MLP-RTDL-HPO_best-1-auc-ovr',
                          'MLP-PLR-D', 'MLP-PLR-HPO_best-1-auc-ovr',
                          'ResNet-RTDL-D', 'ResNet-RTDL-HPO_best-1-auc-ovr',
                          'RF-SKL-D', 'RF-HPO_best-1-auc-ovr', 'XGB-PBB-D',
                          'TabR-S-D', 'RealTabR-D', 'RealTabR-D_no-ls', 'TabR-HPO_best-1-auc-ovr', 'BestModel-HPO'])

    arrow_alg_names_auc = [('MLP-PLR-D', 'RealMLP-TD_no-ls'), ('TabR-S-D', 'RealTabR-D_no-ls'), ('XGB-D', 'XGB-TD'),
                           ('LGBM-D', 'LGBM-TD'), ('CatBoost-D', 'CatBoost-TD'),
                           ('MLP-PLR-HPO_best-1-auc-ovr', 'RealMLP-HPO_best-1-auc-ovr')]

    plot_pareto(paths, tables, coll_names=['meta-train-class', 'meta-test-class'], alg_names=alg_names_auc,
                arrow_alg_names=arrow_alg_names_auc,
                val_metric_name='1-auc_ovr', test_metric_name='1-auc_ovr',
                filename='pareto_mtrc_mtec_auc-ovr_val-acc.pdf')
    plot_pareto(paths, tables, coll_names=['meta-test-class', 'grinsztajn-class-filtered'], alg_names=alg_names_auc,
                arrow_alg_names=arrow_alg_names_auc,
                val_metric_name='1-auc_ovr', test_metric_name='1-auc_ovr',
                filename='pareto_mtec_gcf_auc-ovr_val-acc.pdf')
    plot_pareto(paths, tables, coll_names=['meta-train-class', 'meta-test-class', 'grinsztajn-class-filtered'],
                alg_names=alg_names_auc,
                arrow_alg_names=arrow_alg_names_auc,
                val_metric_name='1-auc_ovr', test_metric_name='1-auc_ovr',
                filename='pareto_mtrc_mtec_gcf_auc-ovr_val-acc.pdf')

    alg_names_ext = [an + '_val-ce' for an in alg_names] + ['RealMLP-TD_val-ce_no-ls', 'RealMLP-TD-S_val-ce_no-ls',
                                                            'RealTabR-D_val-ce_no-ls',
                                                            'BestModel-TD_val-ce', 'BestModel-D_val-ce']
    arrow_alg_names_valce = [('MLP-PLR-D_val-ce', 'RealMLP-TD_val-ce_no-ls'),
                             ('TabR-S-D_val-ce', 'RealTabR-D_val-ce_no-ls'), ('XGB-D_val-ce', 'XGB-TD_val-ce'),
                             ('LGBM-D_val-ce', 'LGBM-TD_val-ce'), ('CatBoost-D_val-ce', 'CatBoost-TD_val-ce')]
    plot_pareto(paths, tables, coll_names=['meta-train-class', 'meta-test-class'], alg_names=alg_names_ext,
                val_metric_name='1-auc_ovr', test_metric_name='1-auc_ovr', tag='paper_val_ce',
                arrow_alg_names=arrow_alg_names_valce,
                filename='pareto_mtrc_mtec_auc-ovr_val-cross-entropy.pdf')
    plot_pareto(paths, tables, coll_names=['meta-test-class', 'grinsztajn-class-filtered'], alg_names=alg_names_ext,
                val_metric_name='1-auc_ovr', test_metric_name='1-auc_ovr', tag='paper_val_ce',
                arrow_alg_names=arrow_alg_names_valce,
                filename='pareto_mtec_gcf_auc-ovr_val-cross-entropy.pdf')
    plot_pareto(paths, tables, coll_names=['meta-train-class', 'meta-test-class', 'grinsztajn-class-filtered'],
                alg_names=alg_names_ext,
                val_metric_name='1-auc_ovr', test_metric_name='1-auc_ovr', tag='paper_val_ce',
                arrow_alg_names=arrow_alg_names_valce,
                filename='pareto_mtrc_mtec_gcf_auc-ovr_val-cross-entropy.pdf')

    # ----- other plots -----

    plot_cumulative_ablations(paths, tables)

    plot_cdd(paths, tables, coll_names=coll_names, alg_names=alg_names_short)
    plot_cdd(paths, tables, coll_names=coll_names[0:2], alg_names=alg_names_short)
    plot_cdd(paths, tables, coll_names=coll_names[2:4], alg_names=alg_names_short)

    generate_architecture_table(paths, tables)

    plot_stopping(paths, tables, classification=True)
    plot_stopping(paths, tables, classification=False)

    generate_preprocessing_table(paths, tables)

    generate_refit_table(paths, tables, 'RealMLP')
    generate_refit_table(paths, tables, 'LGBM')

    generate_ablations_table(paths, tables)

    generate_collections_table(paths)

    for coll_name in coll_names:
        plot_winrates(paths=paths, tables=tables, coll_name=coll_name, alg_names=alg_names)

    for coll_name in coll_names:
        for algs_name, new_alg_names in [
            ('defaults',
             ['RealMLP-TD', 'RealTabR-D', 'TabR-S-D', 'MLP-PLR-D', 'MLP-RTDL-D', 'CatBoost-TD', 'LGBM-TD', 'XGB-TD',
              'RF-SKL-D']),
            ('hpo',
             ['RealMLP-HPO', 'TabR-HPO', 'MLP-PLR-HPO', 'FTT-HPO', 'ResNet-RTDL-HPO', 'MLP-RTDL-HPO', 'CatBoost-HPO',
              'LGBM-HPO',
              'XGB-HPO', 'RF-HPO'])]:
            generate_individual_results_table(paths, tables, f'individual_results_{coll_name}_{algs_name}.tex',
                                              coll_name=coll_name,
                                              alg_names=new_alg_names)

    generate_ds_table(paths, TaskCollection.from_name('meta-train-class', paths), include_openml_ids=False)
    generate_ds_table(paths, TaskCollection.from_name('meta-train-reg', paths), include_openml_ids=False)
    generate_ds_table(paths, TaskCollection.from_name('meta-test-class', paths), include_openml_ids=True)
    generate_ds_table(paths, TaskCollection.from_name('meta-test-reg', paths), include_openml_ids=True)
    generate_ds_table(paths, TaskCollection.from_name('grinsztajn-class-filtered', paths), include_openml_ids=True)
    generate_ds_table(paths, TaskCollection.from_name('grinsztajn-reg', paths), include_openml_ids=True)
    plot_schedule(paths, filename='coslog4.pdf', sched_name='coslog4')
    plot_schedules(paths, filename='coslog4_and_flatcos.pdf', sched_names=['coslog4', 'flat_cos'],
                   sched_labels=[r'$\mathrm{coslog}_4$', r'$\mathrm{flat\_cos}$'])

    for coll_name in ['meta-test-class', 'meta-test-reg']:
        plot_scatter(paths, tables=tables, filename=f'scatter_{coll_name}_BestModel-TD_CatBoost-HPO.pdf',
                     coll_names=[coll_name],
                     alg_name_1='BestModel-TD', alg_name_2='CatBoost-HPO')
        # plot_scatter(paths, tables=tables, filename=f'scatter_{coll_name}_HPO-on-BestModel-TD_MLP-TD-HPO.pdf',
        #              coll_names=[coll_name],
        #              alg_name_2='RealMLP-HPO', alg_name_1='HPO-on-BestModel-TD')
        # plot_scatter(paths, tables=tables, filename=f'scatter_{coll_name}_HPO-on-BestModel-TD_BestModel-HPO.pdf',
        #              coll_names=[coll_name],
        #              alg_name_2='BestModel-HPO', alg_name_1='HPO-on-BestModel-TD')
        # plot_scatter(paths, tables=tables, filename=f'scatter_{coll_name}_HPO-on-BestModel-TD_BestModel-TD.pdf',
        #              coll_names=[coll_name],
        #              alg_name_2='BestModel-TD', alg_name_1='HPO-on-BestModel-TD')
    for coll_name in coll_names:
        for alg_name_1, alg_name_2 in [('RealMLP-TD', 'CatBoost-TD'), ('RealMLP-TD', 'RealMLP-HPO'),
                                       ('RealMLP-HPO', 'CatBoost-HPO'),
                                       ('CatBoost-TD', 'CatBoost-HPO'), ('BestModel-TD', 'BestModel-HPO'),
                                       ('Ensemble-TD', 'BestModel-TD'), ('BestModel-TD', 'CatBoost-HPO'),
                                       ('RealMLP-TD', 'MLP-RTDL-D'), ('CatBoost-TD', 'LGBM-TD'),
                                       ('BestModel-TD', 'BestModel-D')]:
            plot_scatter(paths, tables=tables, filename=f'scatter_3x2_{alg_name_1}_{alg_name_2}.pdf',
                         coll_names=coll_names,
                         alg_name_1=alg_name_1, alg_name_2=alg_name_2)
    plot_scatter(paths, tables=tables, filename=f'scatter_3x2_CatBoost-TD_CatBoost-HPO_valid-errors.pdf',
                 coll_names=coll_names,
                 alg_name_1='CatBoost-TD', alg_name_2='CatBoost-HPO', use_validation_errors=True)
