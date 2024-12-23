from typing import List

from pytabkit.bench.eval.plotting import plot_pareto
from pytabkit.bench.data.paths import Paths
from pytabkit.bench.eval.analysis import ResultsTables
from pathlib import Path

def plot_animations(coll_names: List[str]):
    paths = Paths.from_env_variables()

    tables = ResultsTables(paths)

    arrow_alg_names = [('MLP-PLR-D', 'RealMLP-TD'), ('TabR-S-D', 'RealTabR-D'), ('XGB-D', 'XGB-TD'),
                        ('LGBM-D', 'LGBM-TD'), ('CatBoost-D', 'CatBoost-TD'), ('MLP-PLR-HPO', 'RealMLP-HPO')]

    alg_names = [f'{method}-{version}'
                    for method in ['XGB', 'LGBM', 'CatBoost', 'BestModel', 'Ensemble']
                    for version in ['D', 'TD', 'HPO']]
    alg_names.extend(['RealMLP-TD', 'RealMLP-TD-S', 'RealMLP-HPO', 'MLP-RTDL-D', 'MLP-RTDL-HPO',
                        'MLP-PLR-D', 'MLP-PLR-HPO', 'RealTabR-D', 'FTT-D', 'FTT-HPO',
                        'ResNet-RTDL-D', 'ResNet-RTDL-HPO', 'RF-SKL-D', 'RF-HPO', 'XGB-PBB-D', 'TabR-S-D', 'TabR-HPO'])

    alg_names_to_keep = ["MLP-RTDL-D", "MLP-PLR-D", "RealMLP-TD", "MLP-HPO", "MLP-PLR-HPO", "RealMLP-HPO",
                        "MLP-RTDL-HPO"]

    # #all
    # plot_pareto(paths, tables,
    #             coll_names=coll_names,
    #             alg_names=alg_names,
    #             use_ranks=False, use_normalized_errors=False,
    #             use_grinnorm_errors=False,
    #             use_geometric_mean=True, arrow_alg_names=arrow_alg_names,
    #             plot_pareto_frontier=False,
    #             filename_suffix='_1',
    #             subfolder='animations',
    #             alg_names_to_hide=[])#alg_name for alg_name in alg_names if alg_name not in black_border_alg_names])
    #
    # # show pareto frontier
    # plot_pareto(paths, tables,
    #             coll_names=coll_names,
    #             alg_names=alg_names,
    #             use_ranks=False, use_normalized_errors=False,
    #             use_grinnorm_errors=False,
    #             use_geometric_mean=True, arrow_alg_names=arrow_alg_names,
    #             pareto_frontier_width=4.,
    #             filename_suffix='_2',
    #             subfolder='animations',
    #             alg_names_to_hide=[])#alg_name for alg_name in alg_names if alg_name not in black_border_alg_names])
    #
    # # show only MLP models
    # plot_pareto(paths, tables,
    #             coll_names=coll_names,
    #             alg_names=alg_names,
    #             use_ranks=False, use_normalized_errors=False,
    #             use_grinnorm_errors=False,
    #             use_geometric_mean=True, arrow_alg_names=arrow_alg_names,
    #             pareto_frontier_width=4.,
    #             filename_suffix='_3',
    #             subfolder='animations',
    #             alg_names_to_hide=[alg_name for alg_name in alg_names if alg_name not in alg_names_to_keep])
    #
    # # add NN baselines
    # alg_names_to_keep = ["MLP-RTDL-D", "MLP-PLR-D", "RealMLP-TD", "MLP-HPO", "MLP-PLR-HPO", "RealMLP-HPO",
    #                     "MLP-RTDL-HPO", "TabR-S-D", "TabR-HPO", "FTT-D", "FTT-HPO"]
    #
    # plot_pareto(paths, tables,
    #             coll_names=coll_names,
    #             alg_names=alg_names,
    #             use_ranks=False, use_normalized_errors=False,
    #             use_grinnorm_errors=False,
    #             use_geometric_mean=True, arrow_alg_names=arrow_alg_names,
    #             pareto_frontier_width=4.,
    #             filename_suffix='_4',
    #             subfolder='animations',
    #             alg_names_to_hide=[alg_name for alg_name in alg_names if alg_name not in alg_names_to_keep])
    #
    # # show that we can also improve TabR with RealTabr
    # alg_names_to_keep = ["MLP-RTDL-D", "MLP-PLR-D", "RealMLP-TD", "MLP-HPO", "MLP-PLR-HPO", "RealMLP-HPO",
    #                     "MLP-RTDL-HPO", "TabR-S-D", "TabR-HPO", "RealTabR-D", "FTT-D", "FTT-HPO"]
    #
    # plot_pareto(paths, tables,
    #             coll_names=coll_names,
    #             alg_names=alg_names,
    #             use_ranks=False, use_normalized_errors=False,
    #             use_grinnorm_errors=False,
    #             use_geometric_mean=True, arrow_alg_names=arrow_alg_names,
    #             pareto_frontier_width=4.,
    #             filename_suffix='_5',
    #             subfolder='animations',
    #             alg_names_to_hide=[alg_name for alg_name in alg_names if alg_name not in alg_names_to_keep])
    #
    # #show that we can also create TD for trees
    # alg_names_to_keep = ["CatBoost-D", "CatBoost-TD", "CatBoost-HPO",
    #                     "XGB-D", "XGB-TD", "XGB-HPO",
    #                     "LGBM-D", "LGBM-TD", "LGBM-HPO"]
    #
    # plot_pareto(paths, tables,
    #             coll_names=coll_names,
    #             alg_names=alg_names,
    #             use_ranks=False, use_normalized_errors=False,
    #             use_grinnorm_errors=False,
    #             use_geometric_mean=True, arrow_alg_names=arrow_alg_names,
    #             pareto_frontier_width=4.,
    #             filename_suffix='_6',
    #             subfolder='animations',
    #             alg_names_to_hide=[alg_name for alg_name in alg_names if alg_name not in alg_names_to_keep])
    #
    # # show that ensembles work well for td
    # alg_names_to_keep = ["CatBoost-TD", "CatBoost-HPO",
    #                     "XGB-TD", "XGB-HPO",
    #                     "LGBM-TD", "LGBM-HPO",
    #                     "RealMLP-TD", "RealMLP-HPO",
    #                     "Ensemble-D", "BestModel-D",
    #                     "Ensemble-TD", "Ensemble-HPO",
    #                     "BestModel-TD", "BestModel-HPO"]
    #
    # plot_pareto(paths, tables,
    #             coll_names=coll_names,
    #             alg_names=alg_names,
    #             use_ranks=False, use_normalized_errors=False,
    #             use_grinnorm_errors=False,
    #             use_geometric_mean=True, arrow_alg_names=arrow_alg_names,
    #             pareto_frontier_width=4.,
    #             filename_suffix='_7',
    #             subfolder='animations',
    #             alg_names_to_hide=[alg_name for alg_name in alg_names if alg_name not in alg_names_to_keep])

    # alg_names_to_keep = ["CatBoost-D", "CatBoost-TD", #"CatBoost-HPO",
    #                      "XGB-D", "XGB-TD", #"XGB-HPO",
    #                      "LGBM-D", "LGBM-TD", #"LGBM-HPO",
    #                      "MLP-PLR-D", "MLP-PLR-HPO",
    #                      "RealMLP-TD", "RealMLP-HPO",
    #                      "TabR-S-D", "RealTabR-D"]
    #
    # plot_pareto(paths, tables,
    #             coll_names=coll_names,
    #             alg_names=alg_names,
    #             use_ranks=False, use_normalized_errors=False,
    #             use_grinnorm_errors=False,
    #             use_geometric_mean=True, arrow_alg_names=arrow_alg_names,
    #             pareto_frontier_width=4.,
    #             filename_suffix='_8',
    #             subfolder='animations',
    #             alg_names_to_hide=[alg_name for alg_name in alg_names if alg_name not in alg_names_to_keep])

    alg_names_to_keep = ["CatBoost-D", "CatBoost-TD", "CatBoost-HPO",
                         "MLP-PLR-D", "MLP-PLR-HPO",
                         "RealMLP-TD", "RealMLP-HPO",
                         "TabR-S-D", "RealTabR-D", "TabR-HPO",
                         "BestModel-D", "BestModel-TD", "BestModel-HPO"]

    plot_pareto(paths, tables,
                coll_names=coll_names,
                alg_names=alg_names,
                use_ranks=False, use_normalized_errors=False,
                use_grinnorm_errors=False,
                use_geometric_mean=True, arrow_alg_names=arrow_alg_names,
                pareto_frontier_width=4.,
                filename_suffix='_9',
                subfolder='animations',
                alg_names_to_hide=[alg_name for alg_name in alg_names if alg_name not in alg_names_to_keep])


    # animation
    # everything
    # then bigger pareto front
    # then remove everything except the algorithms of interest


if __name__ == '__main__':
    coll_names = ['meta-train-class', 'meta-train-reg', 'meta-test-class', 'meta-test-reg', 'grinsztajn-class-filtered',
                    'grinsztajn-reg']

    plot_animations(['meta-test-class', 'meta-test-reg'])
    plot_animations(['grinsztajn-class-filtered', 'grinsztajn-reg'])
    plot_animations(['meta-train-class', 'meta-train-reg'])
    plot_animations(['meta-test-class', 'grinsztajn-class-filtered'])
    plot_animations(['meta-test-reg', 'grinsztajn-reg'])
