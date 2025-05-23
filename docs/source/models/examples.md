# Examples

## Refitting RealMLP on train+val data using the best epoch from a previous run

You can refit RealMLP by simply using $n_refit=1$
(or, better, larger values to ensemble multiple NNs). 
But in case you want more control, you can do it manually
(e.g., if you only want to refit the best configuration from HPO,
but you're not using the HPO within pytabkit).

```python
import numpy as np
from sklearn.model_selection import train_test_split

from pytabkit import RealMLP_TD_Regressor

np.random.seed(0)

X = np.random.randn(500, 5)
y = np.random.randn(500)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

reg = RealMLP_TD_Regressor(verbosity=2, random_state=0)
reg.fit(X_train, y_train, X_val, y_val)

refit = RealMLP_TD_Regressor(verbosity=2, stop_epoch=list(reg.fit_params_['stop_epoch'].values())[0], val_fraction=0.0, random_state=0)
refit.fit(X, y)
```

## Fitting again after HPO on a smaller subset

Here is an example on how to fit HPO on a smaller subset 
and fit the best configuration again with validation. 
(It might be better to just use `n_refit` in the HPO classifier/regressor instead.)

```python
import numpy as np
from sklearn.model_selection import train_test_split

from pytabkit import LGBM_HPO_TPE_Regressor, LGBM_TD_Regressor

# This is an example on how to fit a HPO method on a smaller subset of the data,
# and then refit the best hyperparams on the full dataset
np.random.seed(0)

X = np.random.randn(500, 5)
y = np.random.randn(500)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.9, random_state=0)

# use 90% for validation to train faster
# if there is too much validation data, validation data might be the bottleneck, then you should pass
model = LGBM_HPO_TPE_Regressor(val_fraction=0.9, n_hyperopt_steps=5)
model.fit(X, y)

# unfortunately params are not always called the same way, so we need to rename a few
params = model.fit_params_['hyper_fit_params']
params['subsample'] = params.pop('bagging_fraction')
params['colsample_bytree'] = params.pop('feature_fraction')
params['lr'] = params.pop('learning_rate')

# unfortunately, it is hard right now to check if this is exactly the same config,
# as this might set some default params that are not used in the HPO config
model_refit = LGBM_TD_Regressor(**params)
model_refit.fit(X, y)
```