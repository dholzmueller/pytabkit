from typing import List, Any, Optional, Dict, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

from pytabkit.models.alg_interfaces.resource_computation import ResourcePredictor
from pytabkit.models import utils
from pytabkit.models.alg_interfaces.base import RequiredResources
from pytabkit.models.alg_interfaces.alg_interfaces import AlgInterface, SingleSplitAlgInterface, \
    RandomParamsAlgInterface
from pytabkit.models.alg_interfaces.base import SplitIdxs, InterfaceResources, RequiredResources, SubSplitIdxs
from pytabkit.models.alg_interfaces.rtdl_interfaces import choose_batch_size_rtdl_new
from pytabkit.models.alg_interfaces.sub_split_interfaces import SingleSplitWrapperAlgInterface
from pytabkit.models.data.data import DictDataset
from pytabkit.models.sklearn.default_params import DefaultParams
from pytabkit.models.training.logging import Logger
from pytabkit.models.nn_models.models import PreprocessingFactory
from pytabkit.models.nn_models.tabr import TabrLightning, TabrModel
from pytabkit.models.nn_models.tabr_context_freeze import TabrModelContextFreeze, TabrLightningContextFreeze
from pytabkit.models.training.metrics import insert_missing_class_columns

import torch.utils.data
try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint




class ExceptionPrintingCallback(pl.callbacks.Callback):
    def on_exception(self, trainer, pl_module, exception):
        import traceback
        print(f'caught exception')
        traceback.print_exception(exception)


class TabRSubSplitInterface(AlgInterface):
    def __init__(self, **config):
        super().__init__(**config)
        self.tfm = None
        self.n_classes = None
        self.model = None
        self.train_ds = None
    
    def create_model(self, n_num_features, n_bin_features,
                     cat_cardinalities, n_classes, freeze_contexts_after_n_epochs: Optional[int]) -> Any:
        
        params_config = [
            ('num_embeddings', None, None),
            ('d_main', None),
            ('d_multiplier', None),
            ('encoder_n_blocks', None),
            ('predictor_n_blocks', None),
            ('mixer_normalization', None),
            ('context_dropout', None),
            ('dropout0', None),
            ('dropout1', None),
            ('normalization', None),
            ('activation', None),
            # The following options should be used only when truly needed.
            ('memory_efficient', None),
            ('candidate_encoding_batch_size', None),
            ('add_scaling_layer', None),
            ('scale_lr_factor', None),
            ('use_ntp_linear', None),
            ('linear_init_type', None),
            ('use_ntp_encoder', None),
        ]
        params = utils.extract_params(self.config, params_config)

        if freeze_contexts_after_n_epochs is not None:
            return TabrModelContextFreeze(
                n_num_features=n_num_features,
                n_bin_features=n_bin_features,
                cat_cardinalities=cat_cardinalities,
                n_classes=n_classes,
                **params
            )
        else:
            return TabrModel(
                n_num_features=n_num_features,
                n_bin_features=n_bin_features,
                cat_cardinalities=cat_cardinalities,
                n_classes=n_classes,
                **params
                )
    
    def infer_batch_size(self, n_samples_train: int) -> int:
        # taken from tabr paper table 14
        # the cutoffs might not be exactly the same
        if n_samples_train < 10_000:
            return 128
        elif n_samples_train < 30_000:
            return 256
        elif n_samples_train < 200_000:
            return 512
        else:
            return 1024
    
    def fit(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources,
            logger: Logger, tmp_folders: List[Optional[Path]], name: str) -> Optional[List[List[List[Tuple[Dict, float]]]]]:
        assert len(idxs_list) == 1
        assert idxs_list[0].n_trainval_splits == 1
        pl.seed_everything(idxs_list[0].sub_split_seeds[0])

        use_deterministic_before = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)

        self.n_classes = ds.get_n_classes()
        train_idxs = idxs_list[0].train_idxs[0]
        val_idxs = idxs_list[0].val_idxs[0] if idxs_list[0].val_idxs is not None else None
        train_ds = ds.get_sub_dataset(train_idxs)
        self.train_ds = train_ds
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

        y = train_ds.tensors['y']
        if is_cv:
            y_val = val_ds.tensors['y']
        # equivalent of sklearn's TransformedTargetRegressor
        transformed_target = self.config.get("transformed_target", False)
        if transformed_target:
            #do TransformedTargetRegressor by hand (because setting the
            # validation set in skorch conflicts with TransformedTargetRegressor)
            self.transformer_mean = y.mean()
            self.transformer_std = y.std()
            y = (y - self.transformer_mean) / self.transformer_std
            if is_cv:
                y_val = (y_val - self.transformer_mean) / self.transformer_std
        else:
            self.transformer_mean = None
            self.transformer_std = None     

        # create datasets for pytorch lightning
        X_num = train_ds.tensors['x_cont']
        X_cat = train_ds.tensors['x_cat']
        # separate bin and cat
        cat_sizes = train_ds.tensor_infos['x_cat'].get_cat_sizes()
        cat_sizes = cat_sizes - 1 # cat sizes contains the size + 1 for unknown values
        #TODO: I think we could do something cleaner
        binary_indicator = cat_sizes == 2
        to_drop_indicator = cat_sizes <= 1 #TODO: this should be dealt with in the converter or the factory
        cat_indicator = (~to_drop_indicator) & (~binary_indicator)
        X_bin = train_ds.tensors['x_cat'][:, binary_indicator]
        X_cat = train_ds.tensors['x_cat'][:, cat_indicator]
        cat_sizes_nonbinary = cat_sizes[cat_indicator].tolist()

        # create validation dataset
        if is_cv:
            X_num_val = val_ds.tensors['x_cont']
            X_cat_val = val_ds.tensors['x_cat']
            # separate bin and cat
            X_bin_val = val_ds.tensors['x_cat'][:, binary_indicator]
            X_cat_val = val_ds.tensors['x_cat'][:, cat_indicator]

        # We need to do ordinalEncoding again here to prevent holes in the categories
        if X_cat.shape[1] > 0:
            #missing values were encoded as 0 in ToDictDatasetConverter
            # missing values were encoded as zero, we need to make them missing again
            self.replace_zero_by_nans = SimpleImputer(missing_values=0.,
                                    strategy="constant",
                                    fill_value=np.nan)
            self.ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', 
                                            unknown_value=-1,
                                            encoded_missing_value=-1)
            # apparently it doesn't work on the integer tensor
            X_cat = self.replace_zero_by_nans.fit_transform(X_cat.float())
            X_cat = torch.from_numpy(self.ord_enc.fit_transform(X_cat))
            if is_cv:
                X_cat_val = self.replace_zero_by_nans.transform(X_cat_val.float())
                X_cat_val = torch.from_numpy(self.ord_enc.transform(X_cat_val))
        if X_bin.shape[1] > 0:
            # the ToDictDatasetConverter encoded binary features as 1 and 2
            # we need to encode them as 0 and 1
            X_bin = X_bin - 1
            assert torch.logical_or(
                torch.logical_or(
                            (X_bin == -1), # missing values were encoded as 0
                            (X_bin == 0)
                            ),
                            (X_bin == 1)).all()
            # replace -1 by 0.5
            X_bin[X_bin == -1] = 0.5
            if is_cv:
                X_bin_val = X_bin_val - 1
                X_bin_val[X_bin_val == -1] = 0.5

        from skorch.dataset import Dataset

        class TabrDataset(Dataset):
            def __init__(self, X_num, X_bin, X_cat, Y):
                self.data = {
                    "Y": Y.reshape(-1)
                }
                if X_num.shape[1] > 0:
                    self.data["X_num"] = X_num.float()
                if X_bin.shape[1] > 0:
                    self.data["X_bin"] = X_bin.long()
                if X_cat.shape[1] > 0:
                    self.data["X_cat"] = X_cat.long()
                self.size = len(Y)

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return {"indices": idx}


        train_dataset = TabrDataset(
            X_num,
            X_bin,
            X_cat,
            y,
        )
        if is_cv:
            val_dataset = TabrDataset(
                X_num_val,
                X_bin_val,
                X_cat_val,
                y_val,
            )
        else:
            assert NotImplementedError

        n_train = idxs_list[0].n_train
        min_context_freeze_train_size = self.config.get('min_context_freeze_train_size', 0)
        freeze_contexts_after_n_epochs = self.config.get('freeze_contexts_after_n_epochs', None)
        if n_train < min_context_freeze_train_size:
            freeze_contexts_after_n_epochs = None  # don't freeze

        torch_model = self.create_model(
            n_num_features=X_num.shape[1],
            n_bin_features=X_bin.shape[1],
            cat_cardinalities=cat_sizes_nonbinary, # we could save a little memory
            # by recomputing the cardinality on train only, but let's keep it simple
            n_classes=self.n_classes if self.n_classes > 0 else None,
            freeze_contexts_after_n_epochs=freeze_contexts_after_n_epochs
            )

        # set batch size if auto
        if self.config.get('batch_size', None) == 'auto':
            self.config['batch_size'] = self.infer_batch_size(len(y))

        self.config["n_threads"] = interface_resources.n_threads
        self.config["verbosity"] = self.config.get("verbosity", 0)

        class_to_use = TabrLightningContextFreeze if freeze_contexts_after_n_epochs is not None else TabrLightning
        self.model = class_to_use(
            torch_model, train_dataset, val_dataset, C=self.config,
            n_classes=self.n_classes,
            )

        if self.n_classes > 0:
            val_metric_name = self.config.get('val_metric_name', 'class_error')
            if val_metric_name == 'class_error':
                es_callback = EarlyStopping(monitor='val_accuracy',
                                                  patience=self.config["patience"], mode='max')
                checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_accuracy", mode="max",
                                                      dirpath=tmp_folders[0])
            elif val_metric_name == 'cross_entropy':
                print(f'Early stopping on cross-entropy loss')
                es_callback = EarlyStopping(monitor='val_loss',
                                            patience=self.config["patience"], mode='min')
                checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min",
                                                      dirpath=tmp_folders[0])
            else:
                raise ValueError(f'Validation metric {val_metric_name} not implemented for TabR')
        else:
            es_callback = EarlyStopping(monitor='val_loss', 
                                              patience=self.config["patience"], mode='min')
            checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min",
                                                  dirpath=tmp_folders[0])

        gpu_devices = interface_resources.gpu_devices
        print("gpu_devices", gpu_devices)
        self.device = gpu_devices[0] if len(gpu_devices) > 0 else 'cpu'
        if self.device == 'cpu':
            pl_accelerator = 'cpu'
            pl_devices = 'auto'
        elif self.device == 'mps':
            pl_accelerator = 'mps'
            pl_devices = 'auto'
        elif self.device == 'cuda':
            pl_accelerator = 'gpu'
            pl_devices = [0]
        elif self.device.startswith('cuda:'):
            pl_accelerator = 'gpu'
            pl_devices = [int(self.device[len('cuda:'):])]
        else:
            raise ValueError(f'Unknown device "{self.device}"')
        
        self.trainer = pl.Trainer(
                           accelerator=pl_accelerator,
                            devices=pl_devices,
                            deterministic=True,
                            callbacks=[es_callback, checkpoint_callback, ExceptionPrintingCallback()],
                            max_epochs=self.config["n_epochs"],
                            enable_progress_bar=self.config["verbosity"] > 0,
                            enable_model_summary=self.config["verbosity"] > 0,
                            logger=pl.loggers.logger.DummyLogger(),
                            )

        self.trainer.fit(self.model)

        if self.config["verbosity"] > 0:
            print("path to best model",
                checkpoint_callback.best_model_path)   # prints path to the best model's checkpoint
            print("best score",
                checkpoint_callback.best_model_score) # and prints it score
        # load best model
        class_to_use = TabrLightningContextFreeze if freeze_contexts_after_n_epochs is not None else TabrLightning
        self.model = class_to_use.load_from_checkpoint(checkpoint_callback.best_model_path,
                                                        model = torch_model,
                                                        train_dataset=train_dataset,
                                                        val_dataset=val_dataset,
                                                        C=self.config,
                                                        n_classes=self.n_classes,
                                                        )

        torch.use_deterministic_algorithms(use_deterministic_before)

        return None

    def predict(self, ds: DictDataset) -> torch.Tensor:
        # adapted from SklearnSubSplitLearner
        # should return tensor of shape len(ds) x output_shape

        use_deterministic_before = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)

        if self.tfm is not None:
            ds = self.tfm.forward_ds(ds)
        
        X_num = ds.tensors['x_cont']
        X_cat = ds.tensors['x_cat']
        # separate bin and cat
        cat_sizes = ds.tensor_infos['x_cat'].get_cat_sizes()
        cat_sizes = cat_sizes - 1 # cat sizes contains the size + 1 for missing values
        binary_indicator = cat_sizes == 2
        to_drop_indicator = cat_sizes <= 1
        cat_indicator = (~to_drop_indicator) & (~binary_indicator)
        X_bin = ds.tensors['x_cat'][:, binary_indicator]
        X_cat = ds.tensors['x_cat'][:, cat_indicator]

        # We need to do ordinalEncoding again here to prevent holes in the categories
        if X_cat.shape[1] > 0:
            X_cat = self.replace_zero_by_nans.transform(X_cat.float())
            X_cat = torch.from_numpy(self.ord_enc.transform(X_cat))
        if X_bin.shape[1] > 0:
            # the ToDictDatasetConverter encoded binary features as 1 and 2
            # we need to encode them as 0 and 1
            X_bin = X_bin - 1
            assert torch.logical_or(
                torch.logical_or(
                            (X_bin == -1), # missing values were encoded as 0
                            (X_bin == 0)
                            ),
                            (X_bin == 1)).all()
            # replace -1 by 0.5
            X_bin[X_bin == -1] = 0.5

        from skorch.dataset import Dataset

        class TabrDatasetTest(Dataset):
            def __init__(self, X_num, X_bin, X_cat):
                self.data = {}
                if X_num.shape[1] > 0:
                    self.data["X_num"] = X_num.float()
                    self.size = len(X_num)
                if X_bin.shape[1] > 0:
                    self.data["X_bin"] = X_bin.long()
                    self.size = len(X_bin)
                if X_cat.shape[1] > 0:
                    self.data["X_cat"] = X_cat.long()
                    self.size = len(X_cat)
            def __len__(self):
                return self.size
            def __getitem__(self, idx):
                return {
                    key: self.data[key][idx]
                    for key in self.data
        }

        test_dataset = TabrDatasetTest(
            X_num,
            X_bin,
            X_cat,
        )
        # create a dataloader
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config["eval_batch_size"],
            shuffle=False,
            num_workers=0, #min(self.config["n_threads"] - 1, 16)
        )

        y_pred = self.trainer.predict(self.model, test_dataloader)
        y_pred = torch.cat(y_pred, dim=0)

        # guard against missing classes in the training set
        # (GBDT interfaces don't need this because they get passed n_classes as a parameter)
        y_pred = insert_missing_class_columns(y_pred, self.train_ds)
        # inverse transform for y (like in TransformedTargetRegressor)
        if self.transformer_mean is not None:
            y_pred = y_pred * self.transformer_std + self.transformer_mean

        torch.use_deterministic_algorithms(use_deterministic_before)

        return y_pred[None]  # add vectorized dimension

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        assert n_cv == 1
        assert n_refit == 0
        assert n_splits == 1
        has_num_emb = self.config.get('num_embeddings', None) is not None
        if has_num_emb:
            num_emb_dict = self.config['num_embeddings']

            num_emb_size_factor = 1.0 + 0.2 * (num_emb_dict.get('n_frequencies', 8) + num_emb_dict.get('d_embedding', 4))
        else:
            num_emb_size_factor = 1.0

        updated_config = utils.join_dicts(dict(n_estimators=100, max_n_threads=1), self.config)
        time_params = {'': 10, 'ds_onehot_size_gb': 10.0, 'n_train': 1e-4}
        ram_params = {'': 4, 'ds_onehot_size_gb': 1.5}
        gpu_ram_params = {'': 5, 'n_features': num_emb_size_factor * 1e-4, "n_train": 3e-5,
                          'n_features*n_train': num_emb_size_factor * 0.5e-7, 'n_classes': 0.04}
        rc = ResourcePredictor(config=updated_config, time_params=time_params, gpu_ram_params=gpu_ram_params,
                               cpu_ram_params=ram_params, n_gpus=1, gpu_usage=0.3) #, gpu_ram_params)
        return rc.get_required_resources(ds, n_train=n_train)


class RandomParamsTabRAlgInterface(RandomParamsAlgInterface):
    def _sample_params(self, is_classification: bool, seed: int, n_train: int):
        rng = np.random.default_rng(seed)
        hpo_space_name = self.config.get('hpo_space_name', 'tabr')

        if hpo_space_name == 'tabr':
            params = {
                # reduced d_layers
                "d_main": rng.choice(np.arange(96, 385)),
                "context_dropout": rng.uniform(0.0, 0.6),
                "dropout0": rng.uniform(0.0, 0.6),
                "dropout1": 0.0,
                "optimizer": {
                    "type": "AdamW",
                    "lr": np.exp(rng.uniform(np.log(3e-5), np.log(1e-3))),
                    "weight_decay": rng.choice([0, np.exp(rng.uniform(np.log(1e-6), np.log(1e-4)))])
                    # paper says 1e-3 but logs on github say 1e-4 for upper bound
                },
                "encoder_n_blocks": rng.choice([0, 1]),
                "predictor_n_blocks": rng.choice([1, 2]),
                "num_embeddings": {
                    "type": "PLREmbeddings",
                    "n_frequencies": rng.choice(np.arange(16, 97)),
                    "d_embedding": rng.choice(np.arange(16, 65)),
                    "frequency_scale": np.exp(rng.uniform(np.log(1e-2), np.log(1e2))),
                    "lite": True,
                },
            }

            if is_classification:
                params = utils.join_dicts(DefaultParams.TABR_S_D_CLASS, params)
            else:
                params = utils.join_dicts(DefaultParams.TABR_S_D_REG, params)
        elif hpo_space_name == 'realtabr':
            tfms_list = [['quantile_tabr'], ['median_center', 'robust_scale', 'smooth_clip']]
            params = {
                # reduced d_layers
                "d_main": rng.choice(np.arange(96, 385)),
                "context_dropout": rng.uniform(0.0, 0.6),
                "dropout0": rng.uniform(0.0, 0.6),
                "dropout1": 0.0,
                "optimizer": {
                    "type": "AdamW",
                    "lr": np.exp(rng.uniform(np.log(3e-5), np.log(1e-3))),
                    "weight_decay": rng.choice([0, np.exp(rng.uniform(np.log(1e-6), np.log(1e-4)))]),
                    # paper says 1e-3 but logs on github say 1e-4 for upper bound
                    "betas": (0.9, rng.choice([0.95, 0.999])),
                },
                "encoder_n_blocks": rng.choice([0, 1]),
                "predictor_n_blocks": rng.choice([1, 2]),
                "num_embeddings": {
                    "type": "PBLDEmbeddings",
                    # use factor 2 since it results in the same hidden dimension
                    # as for PLR without the factor 2 because of the concat(sin, cos) thing
                    "n_frequencies": 2*rng.choice(np.arange(16, 97)),
                    "d_embedding": rng.choice(np.arange(16, 65)),
                    "frequency_scale": np.exp(rng.uniform(np.log(1e-2), np.log(1e2))),
                },
                "ls_eps": rng.choice([0.0, 0.1]),
                'tfms': tfms_list[rng.choice(np.arange(len(tfms_list)))],
                'add_scaling_layer': rng.choice([True, False]),
                'scale_lr_factor': 96,
            }

            if is_classification:
                params = utils.join_dicts(DefaultParams.RealTABR_D_CLASS, params)
            else:
                params = utils.join_dicts(DefaultParams.RealTABR_D_REG, params)
        else:
            raise ValueError(f'Unknown HPO space name "{hpo_space_name}"')

        return params

    def _create_interface_from_config(self, n_tv_splits: int, **config):
        return SingleSplitWrapperAlgInterface([TabRSubSplitInterface(**config) for i in range(n_tv_splits)])
