# Training directly with PyTorch Lightning

## Using PyTorch Lightning

The TabNN models are implemented using [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/).
It follows the following training implementation principle as described [here](https://lightning.ai/docs/pytorch/stable/model/train_model_basic.html):

```python
# define Dataloader
train_loader = DataLoader(x_train, y_train)
val_loader = DataLoader(x_val, y_val)
test_loader = DataLoader(x_test, y_test)

# define model using a Pytorch LightningModule
nn_model = MyModel(hyper_param1, hyper_param2, ...)

# train model using the Pytorch Lightning Trainer
trainer = pl.Trainer()
trainer.fit(model=nn_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# make predictions using the Trainer
pred = trainer.predict(nn_model, dataloaders=test_loader)
```

In our use case, adapted to the Tabular NN Network, the implementation looks like this:

``` { .python .annotate } 
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from pytabkit.models.alg_interfaces.base import SplitIdxs, InterfaceResources
from pytabkit.models.data.data import DictDataset, TensorInfo
from pytabkit.models.sklearn.default_params import DefaultParams
from pytabkit.models.training.lightning_modules import TabNNModule

import lightning.pytorch as pl  # or: import pytorch_lightning as pl
import numpy as np
import torch

n_epochs = 200

X, y = make_classification()

idxs = np.arange(len(X))
trainval_idxs, test_idxs = train_test_split(idxs, test_size=0.2)
n_trainval_splits = 5
train_idxs_list = []
val_idxs_list = []
for i in range(n_trainval_splits):
    train_idxs, val_idxs = train_test_split(trainval_idxs, test_size=0.2)
    train_idxs_list.append(train_idxs)
    val_idxs_list.append(val_idxs)

# define datasets
ds = DictDataset(tensors={'x_cont': torch.as_tensor(X, dtype=torch.float32),
                                    'x_cat': torch.zeros(len(X), 0),
                                    'y': torch.as_tensor(y, dtype=torch.long)[:, None]},
                           tensor_infos={'x_cont': TensorInfo(feat_shape=[X.shape[1]]),
                                         'x_cat': TensorInfo(cat_sizes=[]),
                                         'y': TensorInfo(cat_sizes=[np.max(y) + 1])}, )  # (1)
train_val_splitting_idxs_list = [
    SplitIdxs(train_idxs=torch.as_tensor(np.stack(train_idxs_list, axis=0), dtype=torch.long),
              val_idxs=torch.as_tensor(np.stack(val_idxs_list, axis=0), dtype=torch.long),
              test_idxs=torch.as_tensor(test_idxs, dtype=torch.long),
              split_seed=0, sub_split_seeds=list(range(len(train_idxs_list))), split_id=0)]

test_ds = ds.get_sub_dataset(torch.as_tensor(test_idxs, dtype=torch.long))

# Create assigned resources
# interface_resources = InterfaceResources(n_threads=4, gpu_devices=['cuda:0'])  # (2)
interface_resources = InterfaceResources(n_threads=4, gpu_devices=[])  # (2)

# define the model using our LightningModule TabNNModule
nn_model = TabNNModule(**DefaultParams.RealMLP_TD_CLASS)
# build and 'compile' the model using the data, now it is ready to use
nn_model.compile_model(ds, train_val_splitting_idxs_list, interface_resources)

# train the model using the Pytorch Lightning Trainer
trainer = pl.Trainer(
    callbacks=nn_model.create_callbacks(),
    max_epochs=n_epochs,
    enable_checkpointing=False,
    enable_progress_bar=False,
    num_sanity_val_steps=0,
    logger=pl.loggers.logger.DummyLogger(),
)  # (3)

trainer.fit(
    model=nn_model,
    train_dataloaders=nn_model.train_dl,
    val_dataloaders=nn_model.val_dl
)
# make predictions using the Trainer
pred = trainer.predict(
    model=nn_model,
    dataloaders=nn_model.get_predict_dataloader(test_ds)
)
```

1. The NN Models have special requirements for their dataloaders, therefore we need to use the `DictDataset` first to create a dataset for both training and validation.
2. We handle our resource management manually, not with Lightning, therefore we need to create an `InterfaceResources` object
3. We use the original [`Trainer`](https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api) Class from Lightning. However, all of the parameters specified here are obligatory for the TabNNModule to work properly.

