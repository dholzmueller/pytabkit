import traceback
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Callable

import numpy as np
import scipy
import sklearn
import torch
import torch.nn as nn
from dask.array import greater
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, make_scorer
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from pytabkit.models.alg_interfaces.alg_interfaces import AlgInterface
from pytabkit.models.alg_interfaces.base import SplitIdxs, InterfaceResources, RequiredResources
from pytabkit.models.data.data import DictDataset
from pytabkit.models.training.logging import Logger

import math


class PostHocCalibrationAlgInterface(AlgInterface):
    def __init__(self, alg_interface: AlgInterface, fit_params: Optional[List[Dict[str, Any]]] = None, **config):
        super().__init__(fit_params=fit_params, **config)
        self.alg_interface = alg_interface
        self.calibrators = []
        self.n_calibs = []

    def _transform_probs(self, probs: np.ndarray) -> np.ndarray:
        offset = self.config.get('calib_input_offset', 0.0)
        if offset != 0.0:
            probs = probs + offset
            probs = probs / np.sum(probs, axis=-1, keepdims=True)
        return probs

    def fit(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources,
            logger: Logger, tmp_folders: List[Optional[Path]], name: str) \
            -> Optional[List[List[List[Tuple[Dict, float]]]]]:
        self.alg_interface.fit(ds, idxs_list, interface_resources, logger, tmp_folders, name)
        y_preds = self.alg_interface.predict(ds)

        for tt_split_idx, split_idxs in enumerate(idxs_list):
            for tv_split_idx in range(split_idxs.n_trainval_splits):
                val_idxs = split_idxs.val_idxs[tv_split_idx]
                y = ds.tensors['y'][val_idxs]
                y_pred = y_preds[len(self.calibrators), val_idxs]
                y_pred_probs = torch.softmax(y_pred, dim=-1)

                import probmetrics.calibrators
                import probmetrics.distributions
                calib = probmetrics.calibrators.get_calibrator(**self.config)
                if self.config.get('calibrate_with_logits', True):
                    calib.fit_torch(y_pred=probmetrics.distributions.CategoricalLogits(y_pred.detach().cpu()),
                                    y_true_labels=y[:, 0])
                else:
                    calib.fit(self._transform_probs(y_pred_probs.detach().cpu().numpy()), y.cpu().numpy()[:, 0])

                self.calibrators.append(calib)
                self.n_calibs.append(val_idxs.shape[-1])

        self.fit_params = [dict(sub_fit_params=fp) for fp in self.alg_interface.fit_params]

        return None

    def predict(self, ds: DictDataset) -> torch.Tensor:
        y_preds = self.alg_interface.predict(ds)
        y_preds_probs = torch.softmax(y_preds, dim=-1)
        y_preds_calib = []
        for i in range(y_preds.shape[0]):
            if self.config.get('calibrate_with_logits', True):
                from probmetrics.distributions import CategoricalLogits
                y_pred_calib = self.calibrators[i].predict_proba_torch(
                    CategoricalLogits(y_preds[i].detach().cpu())).get_probs()
            else:
                y_pred_calib = self.calibrators[i].predict_proba(
                    self._transform_probs(y_preds_probs[i].detach().cpu().numpy()))
                # the np.array(...) is for avoiding read-only array warnings
                y_pred_calib = torch.as_tensor(np.array(y_pred_calib), dtype=torch.float32)

            if self.config.get('use_calib_offset', False):
                y_pred_calib += 1. / self.n_calibs[i]
            y_pred_calib = torch.log(y_pred_calib + 1e-30)
            y_preds_calib.append(y_pred_calib)
        result = torch.stack(y_preds_calib, dim=0)
        # print(f'{y_preds.shape=}, {result.shape=}')
        return result

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        return self.alg_interface.get_required_resources(ds, n_cv, n_refit, n_splits, split_seeds, n_train=n_train)
