from pytabkit.models.training.lightning_callbacks import ModelCheckpointCallback

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl
from typing import List, Optional, Dict, Any
import numpy as np
import torch

from pytabkit.models.data.data import ParallelDictDataLoader, DictDataset
from pytabkit.models.alg_interfaces.base import SplitIdxs, InterfaceResources
from pytabkit.models.nn_models.base import Layer
from pytabkit.models.optim.optimizers import get_opt_class
from pytabkit.models.training.nn_creator import NNCreator
from pytabkit.models.training.logging import StdoutLogger, Logger
from pytabkit.models.training.metrics import Metrics
from pytabkit.models.training.scheduling import LearnerProgress


def postprocess_multiquantile(y_pred: torch.Tensor, val_metric_name: Optional[str] = None,
                              sort_quantile_predictions: bool = True,
                              **config):
    if val_metric_name is None or not val_metric_name.startswith('multi_pinball(') or not sort_quantile_predictions:
        return y_pred

    quantiles = [float(q_str) for q_str in val_metric_name[len('multi_pinball('):-1].split(',')]
    if not all([a <= b for a, b in zip(quantiles[:-1], quantiles[1:])]):
        raise ValueError(f'Quantiles {quantiles} must be sorted')

    return y_pred.sort(dim=-1)[0]


class TabNNModule(pl.LightningModule):
    def __init__(self, n_epochs: int = 256, logger: Optional[Logger] = None,
                 fit_params: Optional[List[Dict[str, Any]]] = None,
                 **config):
        """
        Pytorch Lightning Module for building and training a pytorch NN for tabular data.
        The core of the module is the NNCreatorInterface, which is used to create the model, the callbacks,
        the hyperparameter manager and the dataloaders. The TabNNModule is responsible for the training loop,
        (optional) validation and inference.
        """
        super().__init__()
        self.my_logger = logger or StdoutLogger(verbosity_level=config.get('verbosity', 0))
        # todo: improve this
        self.creator = NNCreator(
            n_epochs=n_epochs, fit_params=fit_params, **config
        )

        self.hp_manager = self.creator.hp_manager
        self.model: Optional[Layer] = None
        self.criterion = None
        self.train_dl = None

        self.progress = LearnerProgress()
        self.progress.max_epochs = n_epochs
        self.fit_params = fit_params

        # Validation
        self.val_preds = []
        self.old_training = None
        self.val_dl = None
        self.save_best_params = True
        self.val_metric_names = None
        self.epoch_mean_val_errors = None
        self.best_mean_val_errors = None
        self.best_mean_val_epochs = None
        self.best_val_errors = None
        self.best_val_epochs = None
        self.has_stopped_list = None
        self.callbacks = None
        # will contain {val_metric_name: ModelCheckpointCallback(..., val_metric_name)}
        self.ckpt_callbacks = dict()

        # LightningModule
        self.automatic_optimization = False

        self.config = config

    def compile_model(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources):
        """
        Method to create the model and all other training dependencies given the dataset and the assigned resources.
        Once this is called, the module is ready for training.
        """
        self.creator.setup_from_dataset(
            ds, idxs_list=idxs_list, interface_resources=interface_resources
        )
        self.model = self.creator.create_model(ds, idxs_list=idxs_list)
        self.train_dl, self.val_dl = self.creator.create_dataloaders(ds)
        self.criterion, self.val_metric_names = self.creator.get_criterions()

    def create_callbacks(self):
        """ Helper method to return callbacks for the trainer.fit callback argument."""
        assert self.val_metric_names is not None
        self.callbacks = self.creator.create_callbacks(self.model, self.my_logger, self.val_metric_names)
        self.ckpt_callbacks = {}
        for callback in self.callbacks:
            if isinstance(callback, ModelCheckpointCallback):
                self.ckpt_callbacks[callback.val_metric_name] = callback
        return self.callbacks

    def get_predict_dataloader(self, ds: DictDataset):
        """ Helper method to create a dataloader for inference."""
        ds_x, _ = ds.split_xy()
        ds_x = self.creator.static_model.forward_ds(ds_x)
        idxs_single = torch.arange(ds.n_samples, dtype=torch.long)
        idxs = idxs_single[None, :].expand(
            self.creator.n_tt_splits * self.creator.n_tv_splits, -1
        )

        return ParallelDictDataLoader(ds=ds_x, idxs=idxs,
                                      batch_size=self.creator.config.get("predict_batch_size", 1024))

    # ----- Start LightningModule Methods -----
    def on_fit_start(self):
        self.model.train()
        self.optimizers().train()
        # mean val errors will not be accurate if all epochs after this yield NaN
        self.best_mean_val_errors = {val_metric_name: [np.inf] * self.creator.n_tt_splits for val_metric_name in
                                     self.val_metric_names}
        # epoch 0 counts as before training, epoch 1 is first epoch
        self.best_mean_val_epochs = {val_metric_name: [0] * self.creator.n_tt_splits for val_metric_name in
                                     self.val_metric_names}
        # don't use simpler notation of the form [[]] * 2 because this will have two references to the same inner array!
        self.best_val_errors = {
            val_metric_name: [[np.inf] * self.creator.n_tv_splits for i in range(self.creator.n_tt_splits)] for
            val_metric_name in self.val_metric_names}
        self.best_val_epochs = {
            val_metric_name: [[0] * self.creator.n_tv_splits for i in range(self.creator.n_tt_splits)] for
            val_metric_name in self.val_metric_names}
        self.has_stopped_list = {
            val_metric_name: [[False] * self.creator.n_tv_splits for i in range(self.creator.n_tt_splits)] for
            val_metric_name in self.val_metric_names}

    def training_step(self, batch, batch_idx):
        output = self.model(batch)
        opt = self.optimizers()
        # do sum() over models dimension
        loss = self.criterion(output["x_cont"], output["y"]).sum()
        # Callbacks for regularization are called before the backward pass
        self.manual_backward(loss)
        opt.step(loss=loss)
        opt.zero_grad()

        self.progress.total_samples += batch["y"].shape[-2]
        self.progress.epoch_float = (
                self.progress.total_samples / self.train_dl.get_num_iterated_samples()
        )
        return loss

    def on_validation_start(self):
        self.old_training = self.model.training
        self.val_preds = []
        self.model.eval()

    def validation_step(self, batch, batch_idx):
        self.val_preds.append(self.model(batch)["x_cont"])

    def on_validation_epoch_end(self):
        self.model.train(self.old_training)
        self.old_training = None
        y_pred = torch.cat(self.val_preds, dim=-2)

        y_pred = postprocess_multiquantile(y_pred, **self.config)

        use_early_stopping = self.config.get('use_early_stopping', False)
        early_stopping_additive_patience = self.config.get('early_stopping_additive_patience', 20)
        early_stopping_multiplicative_patience = self.config.get('early_stopping_multiplicative_patience', 2)

        for val_metric_name in self.val_metric_names:
            val_errors = torch.as_tensor(
                [
                    Metrics.apply(
                        y_pred[i, :, :], self.val_dl.val_y[i, :, :], val_metric_name
                    )
                    for i in range(y_pred.shape[0])
                ]
            )
            val_errors = val_errors.view(
                self.creator.n_tt_splits, self.creator.n_tv_splits
            )
            mean_val_errors = val_errors.mean(dim=-1)  # mean over cv/refit dimension
            mean_val_error = mean_val_errors.mean().item()

            self.my_logger.log(
                2,
                f"Epoch {self.progress.epoch + 1}/{self.progress.max_epochs}: val {val_metric_name} = {mean_val_error:6.6f}",
            )

            current_epoch = self.progress.epoch + 1

            for tt_split_idx in range(self.creator.n_tt_splits):
                use_last_best_epoch = self.config.get('use_last_best_epoch', True)

                has_stopped = self.has_stopped_list[val_metric_name][tt_split_idx]

                # compute best single-split validation errors
                for tv_split_idx in range(self.creator.n_tv_splits):
                    if use_early_stopping and not has_stopped[tv_split_idx]:
                        if current_epoch > early_stopping_multiplicative_patience \
                                * self.best_val_epochs[val_metric_name][tt_split_idx][tv_split_idx] \
                                + early_stopping_additive_patience:
                            has_stopped[tv_split_idx] = True

                    if not has_stopped[tv_split_idx]:
                        # compute best validation errors
                        current_err = val_errors[tt_split_idx, tv_split_idx].item()
                        best_err = self.best_val_errors[val_metric_name][tt_split_idx][tv_split_idx]
                        # use <= on purpose such that latest epoch among tied best epochs is kept
                        # this has been slightly beneficial for accuracy in previous experiments
                        improved = current_err <= best_err if use_last_best_epoch \
                            else current_err < best_err
                        if improved:
                            self.best_val_errors[val_metric_name][tt_split_idx][tv_split_idx] = current_err
                            self.best_val_epochs[val_metric_name][tt_split_idx][tv_split_idx] = (
                                    self.progress.epoch + 1
                            )

                if not any(has_stopped):
                    # compute best mean validation errors (averaged over sub-splits (cv/refit))
                    # use <= on purpose such that latest epoch among tied best epochs is kept
                    # this has been slightly beneficial for accuracy in previous experiments
                    improved = mean_val_errors[tt_split_idx] <= self.best_mean_val_errors[val_metric_name][
                        tt_split_idx] if use_last_best_epoch \
                        else mean_val_errors[tt_split_idx] < self.best_mean_val_errors[val_metric_name][tt_split_idx]
                    if improved:
                        self.best_mean_val_errors[val_metric_name][tt_split_idx] = mean_val_errors[tt_split_idx]
                        self.best_mean_val_epochs[val_metric_name][tt_split_idx] = (
                                self.progress.epoch + 1
                        )
        self.progress.epoch += 1

        if use_early_stopping and all(all([all(sub_lst) for sub_lst in lst]) for lst in self.has_stopped_list.values()):
            self.trainer.should_stop = True

    def on_fit_end(self):
        # if self.creator.config.get("use_best_epoch", True):
        #     self.fit_params = [{'stop_epoch': mean_ep, 'best_indiv_stop_epochs': single_eps}
        #                        for mean_ep, single_eps in zip(self.best_mean_val_epochs, self.best_val_epochs)]
        # else:
        #     self.fit_params = [
        #         {"stop_epoch": self.progress.max_epochs}
        #         for i in range(self.creator.n_tt_splits)
        #     ]

        if self.creator.config.get("use_best_epoch", True):
            self.fit_params = [{'stop_epoch': {val_metric_name: self.best_mean_val_epochs[val_metric_name][i] for
                                               val_metric_name in self.val_metric_names},
                                'best_indiv_stop_epochs': {val_metric_name: self.best_val_epochs[val_metric_name][i] for
                                                           val_metric_name in self.val_metric_names}}
                               for i in range(self.creator.n_tt_splits)]
        else:
            self.fit_params = [
                {"stop_epoch": {val_metric_name: self.progress.max_epochs for val_metric_name in self.val_metric_names}}
                for i in range(self.creator.n_tt_splits)
            ]

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        self.model.eval()
        with torch.no_grad():
            return self.model(batch)["x_cont"].to("cpu")

    def configure_optimizers(self):
        param_groups = [{"params": [p], "lr": 0.01} for p in self.model.parameters()]
        return get_opt_class(self.config.get('opt', 'adam'))(param_groups, self.hp_manager)

    def restore_ckpt_for_val_metric_name(self, val_metric_name: str):
        self.ckpt_callbacks[val_metric_name].restore(self)

    # from https://github.com/Lightning-AI/pytorch-lightning/discussions/19759
    # def on_fit_start(self) -> None:
    #     self.optimizers().train()  # already abovef

    def on_predict_start(self) -> None:
        self.optimizers(use_pl_optimizer=False).eval()

    def on_validation_model_eval(self) -> None:
        self.model.eval()
        self.optimizers(use_pl_optimizer=False).eval()

    def on_validation_model_train(self) -> None:
        self.model.train()
        self.optimizers(use_pl_optimizer=False).train()

    def on_test_model_eval(self) -> None:
        self.model.eval()
        self.optimizers(use_pl_optimizer=False).eval()

    def on_test_model_train(self) -> None:
        self.model.train()
        self.optimizers(use_pl_optimizer=False).train()

    def on_predict_model_eval(self) -> None:  # redundant with on_predict_start()
        self.model.eval()
        self.optimizers(use_pl_optimizer=False).eval()
