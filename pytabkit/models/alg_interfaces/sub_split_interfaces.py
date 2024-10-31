import random
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch

from pytabkit.models import utils
from pytabkit.models.alg_interfaces.alg_interfaces import SingleSplitAlgInterface, AlgInterface
from pytabkit.models.alg_interfaces.base import SplitIdxs, InterfaceResources, RequiredResources
from pytabkit.models.data.data import DictDataset
from pytabkit.models.nn_models.models import PreprocessingFactory
from pytabkit.models.training.logging import Logger
from pytabkit.models.training.metrics import insert_missing_class_columns


class SingleSplitWrapperAlgInterface(SingleSplitAlgInterface):
    """
    AlgInterface that takes multiple AlgInterfaces that can only handle a single train-val-test split
    and wraps them to handle a trainval-test split (possibly with multiple train-val splits)
    """
    def __init__(self, sub_split_interfaces: List[AlgInterface], fit_params: Optional[List[Dict[str, Any]]] = None,
                 **config):
        """
        :param sub_split_interfaces: Interfaces for each sub-split (train-val split).
        """
        super().__init__(fit_params=fit_params, **config)
        self.sub_split_interfaces = sub_split_interfaces

    def get_refit_interface(self, n_refit: int, fit_params: Optional[List[Dict]] = None) -> 'AlgInterface':
        if fit_params is not None:
            assert len(fit_params) == 1  # single split required
        orig_fit_params = fit_params
        fit_params = fit_params or self.fit_params

        config = utils.join_dicts(self.sub_split_interfaces[0].config, self.config)
        if config.get('use_best_mean_iteration_for_refit', True):
            return SingleSplitWrapperAlgInterface(
                [self.sub_split_interfaces[0].get_refit_interface(n_refit=1, fit_params=fit_params) for i in
                 range(n_refit)], fit_params=fit_params)
        else:
            if n_refit != len(self.sub_split_interfaces):
                raise ValueError('When use_best_mean_iteration_for_refit==False, we must have n_cv==n_refit, '
                                 f'but got n_cv={len(self.sub_split_interfaces)} and {n_refit=}')
            if orig_fit_params is not None:
                raise ValueError('When use_best_mean_iteration_for_refit==False, '
                                 'fit_params in get_refit_interface() should be None')
            return SingleSplitWrapperAlgInterface(
                [ssi.get_refit_interface(n_refit=1) for ssi in self.sub_split_interfaces], fit_params=fit_params)

    def fit(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources,
            logger: Logger, tmp_folders: List[Optional[Path]], name: str) -> Optional[
        List[List[List[Tuple[Dict, float]]]]]:
        assert len(idxs_list) == 1  # this is a SingleSplitAlgInterface
        assert len(tmp_folders) == 1  # this is a SingleSplitAlgInterface
        split_idxs = idxs_list[0]
        tmp_folder = tmp_folders[0]
        hyper_results_list = []
        # todo: this could be parallelized if necessary, but not for now
        for i in range(split_idxs.n_trainval_splits):
            sub_split_idxs = [split_idxs.get_sub_split_idxs_alt(i)]
            sub_tmp_folder = tmp_folder / f'sub_split_{i}' if tmp_folder is not None else None
            # don't set fit_params here
            # because we might intentionally not want to set them if use_best_mean_iteration_for_refit==False
            # see get_refit_interfaces()
            # if self.fit_params is not None:
            #     self.sub_split_interfaces[i].fit_params = self.fit_params
            hyper_results = self.sub_split_interfaces[i].fit(ds, sub_split_idxs, interface_resources, logger,
                                                             [sub_tmp_folder], name=name)
            hyper_results = hyper_results[0][0] if hyper_results is not None else []
            hyper_results_list.append(hyper_results)

        if self.fit_params is None:
            # determine best fit parameters (early stopping epoch or so)
            # by averaging losses across cv splits and then taking the minimum of that

            n_hyper_results = [len(hyper_result) for hyper_result in hyper_results_list]
            # print(f'{n_hyper_results=}')

            # truncate all hyper results to minimum length (could be different in case of early stopping)
            min_n_hyper_results = min(n_hyper_results)

            if min_n_hyper_results > 0:
                for i in range(len(hyper_results_list)):
                    hyper_results_list[i] = hyper_results_list[i][:min_n_hyper_results]

                n_hyper_results = [len(hyper_result) for hyper_result in hyper_results_list]
                if not utils.all_equal(n_hyper_results):
                    raise RuntimeError(f'Got hyperparameter results of different lengths: {n_hyper_results}')
                for i in range(n_hyper_results[0]):
                    if not utils.all_equal([frozenset(hyper_result[i][0]) for hyper_result in hyper_results_list]):
                        raise RuntimeError(f'Hyperparameter result lists did not use the same hyperparameters')
                mean_hyper_results = np.asarray([np.mean([hyper_result[i][1] for hyper_result in hyper_results_list])
                                                 for i in range(n_hyper_results[0])])
                # use reverse argmin for ties since it sometimes gives better results
                best_idx = utils.reverse_argmin(mean_hyper_results)
                self.fit_params = [hyper_results_list[0][best_idx][0]]

                # steal the config from the sub_split_interface because it usually gets all the kwargs
                config = utils.join_dicts(self.sub_split_interfaces[0].config, self.config)

                if config.get('use_best_mean_iteration_for_cv', False):
                    for ssi in self.sub_split_interfaces:
                        ssi.fit_params = self.fit_params
            else:
                self.fit_params = [dict()]

        return None

    def predict(self, ds: DictDataset) -> torch.Tensor:
        # todo: pay attention to dimensions
        return torch.cat([s.predict(ds) for s in self.sub_split_interfaces], dim=0)

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        assert n_splits == 1
        assert n_cv == len(self.sub_split_interfaces)
        # todo: this is ignoring the refit stage
        single_resources = [ssi.get_required_resources(ds, n_cv=1, n_refit=0, n_splits=1, split_seeds=[split_seed], n_train=n_train)
                            for ssi, split_seed in zip(self.sub_split_interfaces, split_seeds)]
        return RequiredResources.combine_sequential(single_resources)


class SklearnSubSplitInterface(SingleSplitAlgInterface):  # todo: have another base class
    """
    Base class for AlgInterfaces based on scikit-learn methods.
    """
    def __init__(self, fit_params: Optional[List[Dict[str, Any]]] = None, **config):
        super().__init__(fit_params=fit_params, **config)
        self.tfm = None
        self.n_classes = None
        self.model = None
        self.train_ds = None

    def fit(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources,
            logger: Logger, tmp_folders: List[Optional[Path]], name: str) -> Optional[
        List[List[List[Tuple[Dict, float]]]]]:
        assert len(idxs_list) == 1
        assert idxs_list[0].n_trainval_splits == 1

        print(f'fit(): {torch.cuda.is_initialized()=}')

        # return List[Tuple[Dict, float]]], i.e., validation scores for every hyperparameter combination
        # (could be number of trees, early stopping epoch, or hyperparameters from hyperparameter optimization)
        # if hyperparams is not None, use these and maybe only return one list element?
        seed = idxs_list[0].sub_split_seeds[0]
        torch.manual_seed(seed)  # can be useful for label encoding with randomized permutation
        np.random.seed(seed)
        random.seed(seed)
        # print(f'Seeding with seed {seed}')
        # print(f'{type(seed)=}')
        self.n_classes = ds.get_n_classes()
        if idxs_list[0].val_idxs is None:
            trainval_idxs = idxs_list[0].train_idxs[0]
            # validation indices such that trainval_idxs[rel_val_idxs] is the val_idxs
            # can be used to index trainval_ds later
            rel_val_idxs = torch.zeros(0, dtype=torch.long)
        else:
            trainval_idxs = torch.cat([idxs_list[0].train_idxs[0], idxs_list[0].val_idxs[0]], dim=0)
            rel_val_idxs = torch.arange(idxs_list[0].n_train, trainval_idxs.shape[0], dtype=torch.long)

        trainval_ds = ds.get_sub_dataset(trainval_idxs)
        # for filling in missing classes in the train dataset later
        # might not work when the validation set contains classes that the training set doesn't contain
        self.train_ds = ds.get_sub_dataset(idxs_list[0].train_idxs[0])

        self.config["tmp_folder"] = tmp_folders[0]
        self.config['interface_resources'] = interface_resources

        # create preprocessing factory
        factory = self.config.get('factory', None)
        if factory is None:
            factory = PreprocessingFactory(**self.config)

        # transform according to factory
        fitter = factory.create(ds.tensor_infos)
        self.tfm, trainval_ds = fitter.fit_transform(trainval_ds)

        y = trainval_ds.tensors['y']

        self.model = self._create_sklearn_model(seed=seed,
                                                n_threads=interface_resources.n_threads,
                                                gpu_devices=interface_resources.gpu_devices)
        if self.n_classes == 0 and trainval_ds.tensor_infos['y'].get_n_features() > 1 \
                and self.config.get('use_multioutput_regressor', False):
            from sklearn.multioutput import MultiOutputRegressor
            self.model = MultiOutputRegressor(self.model)  # todo: test this
            y = y.numpy()
        else:
            y = y[:, 0].numpy()

        x_df = trainval_ds.without_labels().to_df()
        cat_col_names = list(x_df.select_dtypes(include='category').columns)
        self._fit_sklearn(x_df=x_df, y=y, val_idxs=rel_val_idxs.numpy(), cat_col_names=cat_col_names)

        return None

    def _fit_sklearn(self, x_df: pd.DataFrame, y: np.ndarray, val_idxs: np.ndarray,
                     cat_col_names: Optional[List[str]] = None):
        # by default, we ignore the validation set since most sklearn methods do not support it
        n_samples = len(x_df)
        train_mask = np.ones(shape=(n_samples,), dtype=np.bool_)
        train_mask[val_idxs] = False
        x_df = x_df.iloc[train_mask, :]
        y = y[train_mask]
        if cat_col_names is not None and len(cat_col_names) > 0:
            self.model.fit(x_df, y, **{self._get_cat_indexes_arg_name(): cat_col_names})
        else:
            self.model.fit(x_df, y)

    def predict(self, ds: DictDataset) -> torch.Tensor:
        # should return tensor of shape len(ds) x output_shape
        if self.tfm is not None:
            ds = self.tfm.forward_ds(ds)

        x_df = ds.without_labels().to_df()

        if self.n_classes > 0:
            # classification
            y_pred = np.log(self._predict_proba_sklearn(x_df) + 1e-30)
        else:
            # regression
            y_pred = self._predict_sklearn(x_df)
            if len(y_pred.shape) == 1:
                y_pred = y_pred[:, None]

        y_pred = torch.as_tensor(y_pred, dtype=torch.float32)
        # guard against missing classes in the training set
        # (GBDT interfaces don't need this because they get passed n_classes as a parameter)
        y_pred = insert_missing_class_columns(y_pred, self.train_ds)
        return y_pred[None]  # add n_models dimension

    def _predict_sklearn(self, x_df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(x_df)

    def _predict_proba_sklearn(self, x_df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(x_df)

    def _create_sklearn_model(self, seed: int, n_threads: int, gpu_devices: List[str]) -> Any:
        # override this in subclasses
        raise NotImplementedError()

    def _get_cat_indexes_arg_name(self) -> str:
        # override this in subclasses if categorical features are supported
        raise NotImplementedError()


class TreeBasedSubSplitInterface(SingleSplitAlgInterface):  # todo: insert more appropriate class to inherit from?
    """
    Base class for tree-based ML models (XGB, LGBM, CatBoost).
    """
    def __init__(self, fit_params: Optional[List[Dict[str, Any]]] = None, **config):
        super().__init__(fit_params=fit_params, **config)
        self.config = config
        self.tfm = None
        self.n_classes = None
        self.model = None

    def fit(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources,
            logger: Logger, tmp_folders: List[Optional[Path]], name: str) -> Optional[
        List[List[List[Tuple[Dict, float]]]]]:
        assert len(idxs_list) == 1
        assert idxs_list[0].n_trainval_splits == 1
        # return List[Tuple[Dict, float]]], i.e., validation scores for every hyperparameter combination
        # (could be number of trees, early stopping epoch, or hyperparameters from hyperparameter optimization)
        # if hyperparams is not None, use these and maybe only return one list element?
        seed = idxs_list[0].sub_split_seeds[0]
        torch.manual_seed(seed)  # can be useful for label encoding with randomized permutation
        np.random.seed(seed)
        random.seed(seed)
        self.n_classes = ds.get_n_classes()
        train_idxs = idxs_list[0].train_idxs[0]
        val_idxs = idxs_list[0].val_idxs[0] if idxs_list[0].val_idxs is not None else None
        train_ds = ds.get_sub_dataset(train_idxs)
        is_cv = val_idxs is not None
        val_ds = ds.get_sub_dataset(val_idxs) if is_cv else None

        # create preprocessing factory
        factory = self.config.get('factory', None)
        if factory is None:
            factory = PreprocessingFactory(**self.config)

        # transform according to factory
        fitter = factory.create(ds.tensor_infos)
        if is_cv:
            trainval_ds = ds.get_sub_dataset(torch.cat([train_idxs, val_idxs], dim=0))
        else:
            trainval_ds = train_ds
        self.tfm = fitter.fit(trainval_ds)
        train_ds = self.tfm.forward_ds(train_ds)
        if is_cv:
            val_ds = self.tfm.forward_ds(val_ds)

        params = self._get_params()
        if self.fit_params is not None:
            params = utils.update_dict(params, self.fit_params[0])
        gpu_ids = [int(dev_str[len('cuda:'):])
                   for dev_str in interface_resources.gpu_devices if dev_str.startswith('cuda:')]
        if len(gpu_ids) > 0 and self.config.get('allow_gpu', True):
            params['device'] = f'cuda:{gpu_ids[0]}'  # this is for XGBoost 2.0
        self.model, val_errors = self._fit(train_ds, val_ds, params=params, seed=seed,
                                           n_threads=interface_resources.n_threads,
                                           val_metric_name=self.config.get('val_metric_name', None),
                                           tmp_folder=tmp_folders[0])
        if val_errors is None:
            return None
        else:
            if self.config.get('use_best_checkpoint', True):
                self.fit_params = [dict(n_estimators=utils.reverse_argmin(val_errors) + 1)]
            else:
                self.fit_params = [dict(n_estimators=len(val_errors))]
            return [[[(dict(n_estimators=i + 1), err) for i, err in enumerate(val_errors)]]]

    def predict(self, ds: DictDataset) -> torch.Tensor:
        # should return tensor of shape len(ds) x output_shape
        if self.tfm is not None:
            ds = self.tfm.forward_ds(ds)
        return self._predict(self.model, ds, self.n_classes,
                             self.fit_params[0] if self.fit_params is not None else dict())[None]

    def _fit(self, train_ds: DictDataset, val_ds: Optional[DictDataset], params: Dict[str, Any], seed: int,
             n_threads: int, val_metric_name: Optional[str] = None,
             tmp_folder: Optional[Path] = None) -> Tuple[Any, Optional[List[float]]]:
        raise NotImplementedError()

    def _predict(self, bst: Any, ds: DictDataset, n_classes: int, other_params: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError()

    def _get_params(self) -> Dict[str, Any]:
        raise NotImplementedError()


