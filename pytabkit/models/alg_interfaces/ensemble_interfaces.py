from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import torch

from pytabkit.models.alg_interfaces.alg_interfaces import SingleSplitAlgInterface, AlgInterface
from pytabkit.models.alg_interfaces.base import SplitIdxs, InterfaceResources, RequiredResources
from pytabkit.models.data.data import DictDataset, TaskType
from pytabkit.models.torch_utils import cat_if_necessary
from pytabkit.models.training.logging import Logger
from pytabkit.models.training.metrics import Metrics


class WeightedPrediction:
    def __init__(self, y_pred_list: List[torch.Tensor], task_type: TaskType):
        self.task_type = task_type
        self.y_pred_converted_list = y_pred_list if task_type == TaskType.REGRESSION \
            else [torch.softmax(y_pred, dim=-1) for y_pred in y_pred_list]

    def predict_for_weights(self, weights: np.ndarray):
        weights = weights.astype(np.float32)
        norm_weights = weights / np.sum(weights)
        weighted_sum = sum([w * y_pred for w, y_pred in zip(norm_weights, self.y_pred_converted_list)])
        if self.task_type == TaskType.CLASSIFICATION:
            weighted_sum = torch.log(weighted_sum + 1e-30)
        return weighted_sum


class CaruanaEnsembleAlgInterface(SingleSplitAlgInterface):
    """
    Following a simple variant of Caruana et al. (2004), "Ensemble selection from libraries of models"
    without pre-selection of candidates
    """

    def __init__(self, alg_interfaces: List[AlgInterface], fit_params: Optional[List[Dict]] = None, **config):
        super().__init__(fit_params=fit_params, **config)
        self.alg_interfaces = alg_interfaces
        self.task_type = None

    def get_refit_interface(self, n_refit: int, fit_params: Optional[List[Dict]] = None) -> 'AlgInterface':
        return CaruanaEnsembleAlgInterface([alg_interface.get_refit_interface(n_refit=n_refit)
                                            for alg_interface in self.alg_interfaces],
                                           fit_params=fit_params or self.fit_params)

    def fit(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources,
            logger: Logger, tmp_folders: List[Optional[Path]], name: str) -> None:
        # train sub-models
        for alg_idx, alg_interface in enumerate(self.alg_interfaces):
            sub_tmp_folders = [tmp_folder / str(alg_idx) if tmp_folder is not None else None for tmp_folder in
                               tmp_folders]
            alg_interface.fit(ds, idxs_list, interface_resources, logger, sub_tmp_folders, name + f'sub-alg-{alg_idx}')

        if self.fit_params is not None:
            # this is the refit stage, there is no validation data set to determine the weights on,
            # instead the weights are already in fit_params
            return
        if idxs_list[0].val_idxs is None:
            raise ValueError('CaruanaEnsembleAlgInterface.fit(): Neither a validation set '
                             'nor ensemble weights were provided')

        self.task_type = TaskType.CLASSIFICATION if ds.tensor_infos[
                                                        'y'].get_cat_size_product() > 0 else TaskType.REGRESSION
        val_metric_name = self.config.get('ens_weight_metric_name', self.config.get('val_metric_name', None))
        if val_metric_name is None:
            val_metric_name = Metrics.default_val_metric_name(task_type=self.task_type)

        n_caruana_steps = self.config.get('n_caruana_steps', 40)  # default value is taken from TaskRepo paper (IIRC)

        y_preds_oob_list = []
        for alg_idx, alg_interface in enumerate(self.alg_interfaces):
            y_preds = alg_interface.predict(ds)
            # get out-of-bag predictions
            y_preds_oob_list.append(cat_if_necessary([y_preds[j, idxs_list[0].val_idxs[j]]
                                                      for j in range(idxs_list[0].val_idxs.shape[0])], dim=0))

        # get out-of-bag labels
        y = ds.tensors['y']
        y_oob = cat_if_necessary([y[idxs_list[0].val_idxs[j]] for j in range(idxs_list[0].val_idxs.shape[0])], dim=0)

        weights = np.zeros(len(self.alg_interfaces), dtype=np.int32)
        best_weights = np.copy(weights)
        best_loss = np.inf

        wp = WeightedPrediction(y_preds_oob_list, self.task_type)

        for step_idx in range(n_caruana_steps):
            best_step_weights = None
            best_step_loss = np.inf
            for weight_idx in range(weights.shape[0]):
                weights[weight_idx] += 1

                y_pred_oob = wp.predict_for_weights(weights)
                loss = Metrics.apply(y_pred_oob, y_oob, val_metric_name).item()
                # print(f'{weights=}, {loss=}')
                if loss < best_step_loss:
                    best_step_loss = loss
                    best_step_weights = np.copy(weights)

                weights[weight_idx] -= 1

            if best_step_loss < best_loss:
                best_loss = best_step_loss
                best_weights = np.copy(best_step_weights)

            weights = best_step_weights

        logger.log(2, f'Obtained ensemble weights: {best_weights}')

        self.fit_params = [dict(alg_weights=best_weights.tolist(),
                                sub_fit_params=[alg_interface.get_fit_params()[0]
                                                for alg_interface in self.alg_interfaces])]

    def predict(self, ds: DictDataset) -> torch.Tensor:
        weights = self.fit_params[0]['alg_weights']
        sparse_weights = [w for w in weights if w != 0]
        sparse_algs = [alg_interface for w, alg_interface in zip(weights, self.alg_interfaces) if w != 0]
        sparse_preds = [ai.predict(ds) for ai in sparse_algs]
        wp = WeightedPrediction(sparse_preds, task_type=self.task_type)
        return wp.predict_for_weights(weights=np.asarray(sparse_weights))

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        single_resources = [
            ssi.get_required_resources(ds, n_cv, n_refit, n_splits=n_splits, split_seeds=split_seeds, n_train=n_train)
            for ssi in self.alg_interfaces]
        return RequiredResources.combine_sequential(single_resources)


class AlgorithmSelectionAlgInterface(SingleSplitAlgInterface):
    """
    Picks the best model out of a list of candidates.
    """

    def __init__(self, alg_interfaces: List[AlgInterface], fit_params: Optional[List[Dict]] = None, **config):
        super().__init__(fit_params=fit_params, **config)
        self.alg_interfaces = alg_interfaces
        self.task_type = None

    def get_refit_interface(self, n_refit: int, fit_params: Optional[List[Dict]] = None) -> 'AlgInterface':
        # todo: could use sub_fit_params
        return AlgorithmSelectionAlgInterface([alg_interface.get_refit_interface(n_refit=n_refit)
                                               for alg_interface in self.alg_interfaces],
                                              fit_params=fit_params or self.fit_params)

    def fit(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources,
            logger: Logger, tmp_folders: List[Optional[Path]], name: str) -> None:
        assert len(idxs_list) == 1

        if self.fit_params is not None:
            # this is the refit stage, there is no validation data set to determine the best model on,
            # instead the best model index is already in fit_params
            best_alg_idx = self.fit_params[0]['best_alg_idx']
            sub_tmp_folders = [tmp_folder / str(best_alg_idx) if tmp_folder is not None else None for tmp_folder in
                               tmp_folders]
            self.alg_interfaces[best_alg_idx].fit(ds, idxs_list, interface_resources, logger, sub_tmp_folders,
                                                  name + f'sub-alg-{best_alg_idx}')

            return

        if idxs_list[0].val_idxs is None:
            raise ValueError('CaruanaEnsembleAlgInterface.fit(): Neither a validation set '
                             'nor fit_params were provided')

        self.task_type = TaskType.CLASSIFICATION if ds.tensor_infos[
                                                        'y'].get_cat_size_product() > 0 else TaskType.REGRESSION
        val_metric_name = self.config.get('alg_sel_metric_name', self.config.get('val_metric_name', None))
        if val_metric_name is None:
            val_metric_name = Metrics.default_val_metric_name(task_type=self.task_type)

        # get out-of-bag labels
        y = ds.tensors['y']
        y_oob = cat_if_necessary([y[idxs_list[0].val_idxs[i]] for i in range(idxs_list[0].val_idxs.shape[0])], dim=0)

        best_alg_idx = 0
        best_alg_loss = np.inf

        for alg_idx, alg_interface in enumerate(self.alg_interfaces):
            sub_tmp_folders = [tmp_folder / str(alg_idx) if tmp_folder is not None else None for tmp_folder in
                               tmp_folders]
            alg_interface.fit(ds, idxs_list, interface_resources, logger, sub_tmp_folders, name + f'sub-alg-{alg_idx}')
            y_preds = alg_interface.predict(ds)
            # get out-of-bag predictions
            y_pred_oob = cat_if_necessary([y_preds[j, idxs_list[0].val_idxs[j]]
                                           for j in range(idxs_list[0].val_idxs.shape[0])], dim=0)
            loss = Metrics.apply(y_pred_oob, y_oob, val_metric_name).item()
            if loss < best_alg_loss:
                best_alg_loss = loss
                best_alg_idx = alg_idx

        self.fit_params = [dict(best_alg_idx=best_alg_idx,
                                sub_fit_params=self.alg_interfaces[best_alg_idx].get_fit_params()[0])]
        logger.log(2, f'Best algorithm has index {best_alg_idx}')
        logger.log(2, f'Algorithm selection fit parameters: {self.fit_params[0]}')

    def predict(self, ds: DictDataset) -> torch.Tensor:
        alg_idx = self.fit_params[0]['best_alg_idx']
        return self.alg_interfaces[alg_idx].predict(ds)

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        # too pessimistic for refit...
        single_resources = [
            ssi.get_required_resources(ds, n_cv, n_refit, n_splits=n_splits, split_seeds=split_seeds, n_train=n_train)
            for ssi in self.alg_interfaces]
        return RequiredResources.combine_sequential(single_resources)


class PrecomputedPredictionsAlgInterface(SingleSplitAlgInterface):
    def __init__(self, y_preds_cv: torch.Tensor, y_preds_refit: Optional[torch.Tensor],
                 fit_params_cv: Dict, fit_params_refit: Optional[Dict]):
        super().__init__()
        self.y_preds_cv = y_preds_cv
        self.y_preds_refit = y_preds_refit
        self.is_refit = None
        self.fit_params_cv = fit_params_cv
        self.fit_params_refit = fit_params_refit

    def get_refit_interface(self, n_refit: int, fit_params: Optional[List[Dict]] = None) -> 'AlgInterface':
        return self  # todo: does this work?

    def fit(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources,
            logger: Logger, tmp_folders: List[Optional[Path]], name: str) -> None:
        self.is_refit = idxs_list[0].val_idxs is None
        self.fit_params = [self.fit_params_refit] if self.is_refit else [self.fit_params_cv]

    def predict(self, ds: DictDataset) -> torch.Tensor:
        if ds.n_samples != self.y_preds_cv.shape[1]:
            raise ValueError('Prediction can only be performed on the exact same dataset '
                             'because this uses precomputed predictions')
        return self.y_preds_refit if self.is_refit else self.y_preds_cv

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        return RequiredResources(time_s=1e-5 * ds.n_samples, cpu_ram_gb=2.0, n_threads=1)
