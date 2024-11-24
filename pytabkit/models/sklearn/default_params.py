class DefaultParams:
    RealMLP_TD_CLASS = dict(
        hidden_sizes=[256] * 3,
        max_one_hot_cat_size=9, embedding_size=8,
        weight_param='ntk', bias_lr_factor=0.1,
        act='selu', use_parametric_act=True, act_lr_factor=0.1,
        block_str='w-b-a-d', p_drop=0.15, p_drop_sched='flat_cos',
        add_front_scale=True,
        scale_lr_factor=6.0,
        bias_init_mode='he+5', weight_init_mode='std',
        wd=2e-2, wd_sched='flat_cos', bias_wd_factor=0.0,
        use_ls=True, ls_eps=0.1,
        num_emb_type='pbld', plr_sigma=0.1, plr_hidden_1=16, plr_hidden_2=4, plr_lr_factor=0.1,
        lr=4e-2,
        tfms=['one_hot', 'median_center', 'robust_scale', 'smooth_clip', 'embedding'],
        n_epochs=256, lr_sched='coslog4', opt='adam', sq_mom=0.95
    )

    RealMLP_TD_S_CLASS = dict(
        hidden_sizes=[256] * 3,
        weight_param='ntk', bias_lr_factor=0.1,
        act='selu',
        block_str='w-b-a',
        add_front_scale=True, scale_lr_factor=6.0,
        bias_init_mode='normal', weight_init_mode='normal',
        last_layer_config=dict(bias_init_mode='zeros', weight_init_mode='zeros'),
        use_ls=True, ls_eps=0.1,
        tfms=['one_hot', 'median_center', 'robust_scale', 'smooth_clip'],
        n_epochs=256, lr=4e-2, lr_sched='coslog4', opt='adam', sq_mom=0.95
    )

    RealMLP_TD_REG = dict(
        hidden_sizes=[256] * 3,
        max_one_hot_cat_size=9, embedding_size=8,
        weight_param='ntk', weight_init_mode='std',
        bias_init_mode='he+5', bias_lr_factor=0.1,
        act='mish', use_parametric_act=True, act_lr_factor=0.1,
        wd=2e-2, wd_sched='flat_cos', bias_wd_factor=0.0,
        block_str='w-b-a-d', p_drop=0.15, p_drop_sched='flat_cos',
        add_front_scale=True, scale_lr_factor=6.0,
        tfms=['one_hot', 'median_center', 'robust_scale', 'smooth_clip', 'embedding'],
        num_emb_type='pbld', plr_sigma=0.1, plr_hidden_1=16, plr_hidden_2=4, plr_lr_factor=0.1,
        clamp_output=True, normalize_output=True,
        lr=0.2,
        n_epochs=256, lr_sched='coslog4', opt='adam', sq_mom=0.95
    )

    RealMLP_TD_S_REG = dict(
        hidden_sizes=[256] * 3,
        weight_param='ntk', bias_lr_factor=0.1,
        bias_init_mode='normal', weight_init_mode='normal',
        last_layer_config=dict(bias_init_mode='zeros', weight_init_mode='zeros'),
        act='mish', normalize_output=True,
        block_str='w-b-a',
        add_front_scale=True, scale_lr_factor=6.0,
        tfms=['one_hot', 'median_center', 'robust_scale', 'smooth_clip'],
        n_epochs=256, lr=7e-2, lr_sched='coslog4', opt='adam', sq_mom=0.95
    )

    # -------- GBDTs ------------

    LGBM_TD_CLASS = dict(
        n_estimators=1000, lr=4e-2, subsample=0.75, colsample_bytree=1.0, num_leaves=50, bagging_freq=1,
        min_data_in_leaf=40, min_sum_hessian_in_leaf=1e-7, max_bin=255, early_stopping_rounds=300,
    )

    LGBM_TD_REG = dict(
        n_estimators=1000, lr=5e-2, subsample=0.7, colsample_bytree=1.0, num_leaves=100, max_bin=255, bagging_freq=1,
        min_data_in_leaf=3, min_sum_hessian_in_leaf=1e-7, early_stopping_rounds=300,
    )

    XGB_TD_CLASS = dict(
        n_estimators=1000, lr=8e-2, min_child_weight=5e-6, reg_lambda=0.0, max_depth=6,
        colsample_bylevel=0.9, subsample=0.65, tree_method='hist', max_bin=256, early_stopping_rounds=300,
    )

    XGB_TD_REG = dict(
        n_estimators=1000, max_depth=9, tree_method='hist', max_bin=256, lr=5e-2, min_child_weight=2.0, reg_lambda=0.0,
        subsample=0.7, early_stopping_rounds=300,
    )

    # from Probst, Boulestix, and Bischl, "Tunability: Importance of ..."
    XGB_PBB_CLASS = dict(
        n_estimators=4168, lr=0.018, min_child_weight=2.06,
        max_depth=13, reg_lambda=0.982, reg_alpha=1.113, subsample=0.839,
        colsample_bytree=0.752, colsample_bylevel=0.585,
        tree_method='hist', max_n_threads=64,
        tfms=['one_hot'], max_one_hot_cat_size=20
    )

    CB_TD_CLASS = dict(
        n_estimators=1000, lr=8e-2, l2_leaf_reg=1e-5, boosting_type='Plain',
        bootstrap_type='Bernoulli', subsample=0.9,
        max_depth=7, random_strength=0.8, one_hot_max_size=15,
        leaf_estimation_iterations=1, max_bin=254, early_stopping_rounds=300,
    )

    CB_TD_REG = dict(
        n_estimators=1000, lr=9e-2, l2_leaf_reg=1e-5, boosting_type='Plain',
        bootstrap_type='Bernoulli', subsample=0.9,
        max_depth=9, random_strength=0.0, max_bin=254,
        one_hot_max_size=20, leaf_estimation_iterations=20, early_stopping_rounds=300,
    )

    # RTDL params

    RESNET_RTDL_D_CLASS_Grinsztajn = {
        "lr_scheduler": False,
        "module_activation": "reglu",
        "module_normalization": "batchnorm",
        "module_n_layers": 8,
        "module_d": 256,
        "module_d_hidden_factor": 2,
        "module_hidden_dropout": 0.2,
        "module_residual_dropout": 0.2,
        "lr": 1e-3,
        "optimizer_weight_decay": 1e-7,
        "optimizer": "adamw",
        "module_d_embedding": 128,
        "batch_size": 256,
        "max_epochs": 300,
        "use_checkpoints": True,
        "es_patience": 40,
        "lr_patience": 30,
        "verbose": 0,
        'tfms': ['quantile'],
    }

    RESNET_RTDL_D_REG_Grinsztajn = {**RESNET_RTDL_D_CLASS_Grinsztajn,
                                    "transformed_target": True}

    MLP_RTDL_D_CLASS_Grinsztajn = {
        "lr_scheduler": False,
        "module_n_layers": 8,
        "module_d_layers": 256,
        "module_d_first_layer": 128,
        "module_d_last_layer": 128,
        "module_dropout": 0.2,
        "lr": 1e-3,
        "optimizer": "adamw",
        "module_d_embedding": 128,
        "batch_size": 256,
        "max_epochs": 300,
        "use_checkpoints": True,
        "es_patience": 40,
        "lr_patience": 30,
        "verbose": 0,
        'tfms': ['quantile'],
    }

    MLP_RTDL_D_REG_Grinsztajn = {**MLP_RTDL_D_CLASS_Grinsztajn,
                                 "transformed_target": True}

    FTT_D_CLASS = {
        "lr_scheduler": False,
        "module_d_token": 192,
        "module_d_ffn_factor": 4. / 3.,
        "module_n_layers": 3,
        "module_n_heads": 8,
        "module_activation": "reglu",
        "module_token_bias": True,
        "module_attention_dropout": 0.2,
        "module_initialization": "kaiming",
        "module_ffn_dropout": 0.1,
        "module_residual_dropout": 0.0,
        "module_prenormalization": True,
        "module_kv_compression": None,
        "module_kv_compression_sharing": None,
        "lr": 1e-4,
        "optimizer": "adamw",
        "optimizer_weight_decay": 1e-5,
        "batch_size": 256,  # default in Grinsztajn is 512?
        "max_epochs": 300,  # todo: keep it?
        "use_checkpoints": True,
        "es_patience": 16,  # value from Gorishniy et al.
        "lr_patience": 30,
        "verbose": 0,
        "tfms": ['quantile_tabr'],
    }

    FTT_D_REG = {**FTT_D_CLASS, "transformed_target": True}

    # Default parameters for rtdl models based on https://github.com/naszilla/tabzilla/blob/main/TabZilla/models/rtdl.py
    RESNET_RTDL_D_CLASS_TabZilla = {
        "lr_scheduler": False,
        "module_activation": "relu",
        "module_normalization": "batchnorm",
        "module_n_layers": 2,
        "module_d": 128,
        "module_d_hidden_factor": 2,
        "module_hidden_dropout": 0.25,  # DROPOUT_FIRST
        "module_residual_dropout": 0.1,  # DROPOUT_SECOND
        "lr": 1e-3,
        "optimizer_weight_decay": 0.01,  # for tabzilla they don't set it which means 0.01 (which seems high compared
        # to rtdl hp space?)
        "optimizer": "adamw",
        "module_d_embedding": 8,
        "batch_size": 128,
        # default param in https://github.com/naszilla/tabzilla/blob/4949a1dea3255c1a794d89aa2422ef1f8c9ae265/README.md?plain=1#L129
        "max_epochs": 1000,  # same
        "use_checkpoints": True,
        "es_patience": 20,  # same
        "lr_patience": 30,
        "verbose": 0,
        'tfms': ['quantile_tabr'],
    }

    RESNET_RTDL_D_REG_TabZilla = {**RESNET_RTDL_D_CLASS_TabZilla,
                                  "transformed_target": True}

    MLP_RTDL_D_CLASS_TabZilla = {
        "lr_scheduler": False,
        "module_n_layers": 3,
        "module_d_first_layer": 128,  # ignored by the code since d_layers is a list
        "module_d_last_layer": 128,  # ignored by the code since d_layers is a list
        "module_d_layers": [128, 256, 128],
        "module_dropout": 0.1,
        # module_activation
        # module_dropout
        # optimizer_weight_decay
        "lr": 1e-3,
        "optimizer": "adamw",
        "module_d_embedding": 8,
        "batch_size": 128,
        # default param in https://github.com/naszilla/tabzilla/blob/4949a1dea3255c1a794d89aa2422ef1f8c9ae265/README.md?plain=1#L129
        "max_epochs": 1000,  # same
        "use_checkpoints": True,
        "es_patience": 20,  # same
        "lr_patience": 30,
        "verbose": 0,
        'tfms': ['quantile_tabr'],
    }

    MLP_RTDL_D_REG_TabZilla = {**MLP_RTDL_D_CLASS_TabZilla,
                               "transformed_target": True}

    MLP_PLR_D_CLASS = {
        # adapted from TabZilla version of MLP_RTDL_D and the defaults of the rtdl_num_embeddings library
        "lr_scheduler": False,
        "module_n_layers": 3,
        "module_d_first_layer": 128,  # ignored by the code since d_layers is a list
        "module_d_last_layer": 128,  # ignored by the code since d_layers is a list
        "module_d_layers": [128, 256, 128],
        "module_dropout": 0.1,
        "lr": 1e-3,
        "optimizer": "adamw",
        "module_d_embedding": 8,
        "batch_size": 128,
        # default param in https://github.com/naszilla/tabzilla/blob/4949a1dea3255c1a794d89aa2422ef1f8c9ae265/README.md?plain=1#L129
        "max_epochs": 1000,  # same
        "use_checkpoints": True,
        "es_patience": 20,  # same
        "lr_patience": 30,
        "verbose": 0,
        'tfms': ['quantile_tabr'],
        "module_num_emb_type": 'plr',
        "module_num_emb_dim": 24,
        "module_num_emb_hidden_dim": 48,
        "module_num_emb_sigma": 0.01,
        "module_num_emb_lite": False
    }

    MLP_PLR_D_REG = {**MLP_PLR_D_CLASS,
                     "transformed_target": True}

    TABR_S_D_CLASS = {
        "num_embeddings": None,
        "d_main": 265,
        "context_dropout": 0.38920071545944357,  # named mixer_dropout sometimes I think
        "d_multiplier": 2.0,
        "encoder_n_blocks": 0,
        "predictor_n_blocks": 1,
        "mixer_normalization": "auto",
        "dropout0": 0.38852797479169876,
        "dropout1": 0.0,
        "normalization": "LayerNorm",
        "activation": "ReLU",
        "batch_size": "auto",  # adapt given the dataset size
        "eval_batch_size": 4096,  # TODO: automatically infer given memory
        "patience": 16,
        "n_epochs": 100_000,  # inf in paper
        "context_size": 96,
        "freeze_contexts_after_n_epochs": None,
        "optimizer": {
            "type": "AdamW",
            "lr": 0.0003121273641315169,
            "weight_decay": 1.2260352006404615e-06
        },
        'tfms': ['quantile_tabr'],
    }

    TABR_S_D_REG = {**TABR_S_D_CLASS,
                    "transformed_target": True}

    TABR_S_D_CLASS_FREEZE = {
        **TABR_S_D_CLASS,
        "freeze_contexts_after_n_epochs": 4,
    }

    TABR_S_D_REG_FREEZE = {
        **TABR_S_D_REG,
        "freeze_contexts_after_n_epochs": 4,
    }

    RealTABR_D_CLASS = {
        "d_main": 265,
        "context_dropout": 0.38920071545944357,  # named mixer_dropout sometimes I think
        "d_multiplier": 2.0,
        "encoder_n_blocks": 0,
        "predictor_n_blocks": 1,
        "mixer_normalization": "auto",
        "dropout0": 0.38852797479169876,
        "dropout1": 0.0,
        "normalization": "LayerNorm",
        "activation": "ReLU",
        "batch_size": "auto",  # adapt given the dataset size
        "eval_batch_size": 4096,
        "patience": 16,
        "n_epochs": 100_000,  # inf in paper
        "context_size": 96,
        "freeze_contexts_after_n_epochs": None,
        'num_embeddings': {
            'type': "PBLDEmbeddings",
            'n_frequencies': 8,  # not 16 because of RAM issues on meta-test
            'd_embedding': 4,
            'frequency_scale': 0.1,
        },
        'tfms': ['median_center', 'robust_scale', 'smooth_clip'],
        'optimizer': {
            "type": "AdamW",
            "lr": 0.0003121273641315169,
            "weight_decay": 1.2260352006404615e-06,
            "betas": (0.9, 0.95),
        },
        'add_scaling_layer': True,
        'scale_lr_factor': 96,
        'ls_eps': 0.1,
    }

    RealTABR_D_REG = {**RealTABR_D_CLASS,
                      "transformed_target": True}

    TABM_D_CLASS = {
        # from https://github.com/yandex-research/tabm/blob/main/example.ipynb
        'arch_type': 'tabm',
        'tabm_k': 32,
        'num_emb_type': 'none',
        'num_emb_n_bins': 48,
        'batch_size': 256,
        'lr': 2e-3,
        'weight_decay': 0.0,
        'n_epochs': 1_000_000_000,
        'patience': 16,
        'd_embedding': 16,
        'd_block': 512,
        'n_blocks': 'auto',
        'dropout': 0.1,
        'compile_model': False,
        'allow_amp': False,
        'tfms': ['quantile_tabr'],
        'gradient_clipping_norm': None,  # set to 1.0 in TabR paper experiments
    }

    TABM_D_REG = TABM_D_CLASS

    # ----- sklearn versions ------

    LGBM_D = dict(
        n_estimators=100,
    )

    XGB_D = dict(
        n_estimators=100, tree_method='hist',
    )

    CB_D = dict(
        n_estimators=1000,
    )

    RF_SKL_D = dict(
        tfms=['ordinal_encoding'], permute_ordinal_encoding=True,
    )

    MLP_SKL_D = dict(
        tfms=['mean_center', 'l2_normalize', 'one_hot']
    )
