import copy
import os
from typing import List, Any, Optional

import numpy as np
import pandas as pd
import torch
from pytabkit.models import utils
from pytabkit.models.alg_interfaces.base import RequiredResources, InterfaceResources
from pytabkit.models.alg_interfaces.resource_computation import ResourcePredictor
from pytabkit.models.alg_interfaces.sub_split_interfaces import SklearnSubSplitInterface
from pytabkit.models.data.data import DictDataset
from pytabkit.models.utils import FunctionProcess


class AutoGluonModelAlgInterface(SklearnSubSplitInterface):
    # parameters: use_gpu?, hp_family?, model_types, max_n_models_per_type
    # possible values for hp_family: default, zeroshot, zeroshot_hpo, zeroshot_hpo_hybrid, default_FTT, light
    # possible values for model_types: 'FASTAI', 'NN_TORCH', 'FT_TRANSFORMER', 'XGB', 'CAT', 'GBM', 'RF', 'XT'
    def _create_sklearn_model(self, seed: int, n_threads: int, gpu_devices: List[str]) -> Any:
        from autogluon.tabular import TabularPredictor

        params_config = []
        params = utils.extract_params(self.config, params_config)
        params['device'] = 'cpu' if len(gpu_devices) == 0 else gpu_devices[0]
        val_metric_name = self.config.get('val_metric_name')

        # todo: random_state?

        other_kwargs = dict()

        if self.n_classes > 0:
            problem_type = 'binary' if self.n_classes == 2 else 'multiclass'
            if val_metric_name is None or val_metric_name == 'class_error':
                eval_metric = 'accuracy'
            elif val_metric_name == 'cross_entropy':
                eval_metric = 'log_loss'
            else:
                raise ValueError(f'{val_metric_name=} not implemented')
        else:
            problem_type = 'regression'
            if val_metric_name is None or val_metric_name == 'rmse':
                eval_metric = 'rmse'
            elif val_metric_name.startswith('pinball('):
                problem_type = 'quantile'
                eval_metric = 'pinball_loss'
                other_kwargs = dict(quantile_levels=[float(val_metric_name[len('pinball('):-1])])
            else:
                raise ValueError(f'{val_metric_name=} not implemented')

        self.eval_metric = eval_metric

        return TabularPredictor(label='label', eval_metric=eval_metric,
                                problem_type=problem_type,
                                path=self.config.get('tmp_folder', None),
                                verbosity=self.config.get('verbosity', 0),
                                log_to_file=False, **other_kwargs)

    def _create_df(self, X: pd.DataFrame, y: Optional[np.ndarray]):
        new_columns = {'input_' + col_name: X[col_name] for col_name in X.columns}
        if y is not None:
            new_columns['label'] = y
        df = pd.DataFrame(new_columns)
        if y is not None:
            is_reg = y.dtype.kind == 'f'
            df['label'] = df['label'].astype('float64' if is_reg else 'category')
        return df

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        assert n_cv == 1
        assert n_refit == 0
        assert n_splits == 1

        use_gpu = self.config.get('use_gpu', False)
        model_types = self.config['model_types']
        if isinstance(model_types, str):
            model_types = [model_types]
        has_ft_transformer = 'FT_TRANSFORMER' in model_types

        updated_config = utils.join_dicts(dict(n_estimators=100, max_n_threads=2), self.config)
        time_params = {'': 10, 'ds_onehot_size_gb': 10.0, 'n_samples': 8e-5, 'n_samples*n_features': 5e-6}
        ram_params = {'': 0.5 if use_gpu else 3.0, 'ds_onehot_size_gb': 1.5}
        gpu_ram_params = {'': 0.4, 'ds_onehot_size_gb': 1.5,
                          'n_features': 3e-2 if has_ft_transformer else 1e-4} if use_gpu else None
        rc = ResourcePredictor(config=updated_config, time_params=time_params, gpu_ram_params=gpu_ram_params,
                               cpu_ram_params=ram_params, n_gpus=1 if use_gpu else 0,
                               gpu_usage=0.02 if use_gpu else 0.0)
        return rc.get_required_resources(ds)

    def _fit_sklearn(self, x_df: pd.DataFrame, y: np.ndarray, val_idxs: np.ndarray,
                     cat_col_names: Optional[List[str]] = None):
        df = self._create_df(x_df, y)
        # by default, we ignore the validation set since most sklearn methods do not support it
        n_samples = len(x_df)
        train_mask = np.ones(shape=(n_samples,), dtype=np.bool_)
        train_mask[val_idxs] = False

        hparams_selected = dict()

        from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

        hparams = copy.deepcopy(get_hyperparameter_config(self.config.get('hp_family', 'default')))
        interface_resources: InterfaceResources = self.config['interface_resources']
        cuda_ids = [device[len('cuda:'):] for device in interface_resources.gpu_devices if device.startswith('cuda:')]
        use_gpu = len(cuda_ids) > 0
        # todo: this is only correct if the variable wasn't already set before
        # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(cuda_ids)
        print(f'_fit_sklearn: {torch.cuda.is_initialized()=}')
        # todo: does it work?
        print(f'{torch.cuda.device_count()=}')
        print(f'{cuda_ids=}')
        print(f'{os.getenv("CUDA_VISIBLE_DEVICES")=}')

        max_n_models_per_type = self.config.get('max_n_models_per_type', 0)
        hparams_idx = self.config.get('hparams_idx', None)
        model_types = self.config['model_types']
        if isinstance(model_types, str):
            model_types = [model_types]

        for key, value in hparams.items():
            # if key in ['FASTAI', 'NN_TORCH', 'FT_TRANSFORMER']:
            if key not in model_types:
                continue
            if not isinstance(value, list):
                value = [value]
            if hparams_idx is not None:
                value = [value[hparams_idx]]
            if max_n_models_per_type > 0 and len(value) > max_n_models_per_type:
                value = value[:max_n_models_per_type]
            for config in value:
                config['ag_args_fit'] = dict(num_gpus=1 if use_gpu else 0)
                if key == 'FT_TRANSFORMER':
                    config['ag_args_fit']['_max_features'] = 100_000
                    config['_max_features'] = 100_000

            hparams_selected[key] = value

        print(f'{hparams_selected=}')

        self.model.fit(df.iloc[train_mask], tuning_data=df.iloc[~train_mask],
                               presets='medium_quality',
                               fit_weighted_ensemble=False,
                               fit_full_last_level_weighted_ensemble=False,
                               hyperparameters=hparams_selected,
                               )

        # fit_func = lambda df, hparams_selected, train_mask, model: model.fit(df.iloc[train_mask], tuning_data=df.iloc[~train_mask],
        #                        presets='medium_quality',
        #                        fit_weighted_ensemble=False,
        #                        fit_full_last_level_weighted_ensemble=False,
        #                        hyperparameters=hparams_selected,
        #                        )
        #
        # print(f'Running fit on autogluon model')
        #
        # # fit_func(df, hparams_selected, train_mask, self.model)
        # self.model = FunctionProcess(fit_func, df, hparams_selected, train_mask, self.model).start().pop_result()
        # print(f'fit completed')

    def _predict_sklearn(self, x_df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._create_df(x_df, None)).to_numpy()

    def _predict_proba_sklearn(self, x_df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._create_df(x_df, None)).to_numpy()
