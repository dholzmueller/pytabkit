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
n_epochs = 200

# define Datasets
train_val_ds = DictDataset(tensors=..., tensor_infos=...,) # (1)
train_val_splitting_idxs_list = [SplitIdxs(train_idxs=.., val_idxs=..., ...)]
test_ds = DictDataset(tensors=..., tensor_infos=...,)

# Create assigned resources
interface_resources = InterfaceResources(..) # (2)

# define the model using our LightningModule TabNNModule
nn_model = TabNNModule(**DefaultParams.MLP_TD_CLASS) # (3)
# build and 'compile' the model using the data, now it is ready to use
nn_model.compile_model(ds, idxs_list, interface_resources)

# train the model using the Pytorch Lightning Trainer
trainer = pl.Trainer(
        callbacks=nn_model.create_callbacks(),
        max_epochs=n_epochs,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        logger=pl.loggers.logger.DummyLogger(),
    ) # (4)
    
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

