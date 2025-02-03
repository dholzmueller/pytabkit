# Hyperparameter optimization

This is a guide how to perform hyperparameter optimization (HPO) 
to get the best results out of RealMLP. 
We consider RealMLP for classification here, but most of the guide 
applies to regression and other baselines as well.

## Option 1: Using the HPO interface

The easiest option is to use the direct HPO interface:

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_HPO_Classifier

X, y = make_classification(random_state=42, n_samples=200, n_features=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

clf = RealMLP_HPO_Classifier(n_hyperopt_steps=10, n_cv=1, verbosity=2, val_metric_name='brier')
clf.fit(X_train, y_train)
clf.predict(X_test)
```

The code above
- runs random search with 10 configurations from the HPO space in the paper 
(should be increased to, say, 50 for better results)
- only uses one training-validation split 
(should be increased to, say, 5 for better results)
- prints validation results of each epoch and best found parameters thanks to `verbosity=2`
- selects the best model and best epoch based on the Brier score 
(default would be classification error)

While using the interface directly is convenient, it has certain drawbacks:
- It is not possible to change the search space, 
e.g. to reduce label smoothing for other metrics than classification error.
- It is not possible to save and resume from an intermediate state.
- It is not possible to use another HPO method than random search.
- It is not (easily) possible to access intermediate results.

Therefore, we now look at a more manual approach.

## Option 2: Performing your own HPO

The following code provides an example on how to do HPO manually.

```python
import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold

from pytabkit.models.alg_interfaces.nn_interfaces import RealMLPParamSampler
from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier
from pytabkit.models.training.metrics import Metrics

n_hyperopt_steps = 10
n_cv = 1
is_classification = True

X, y = make_classification(random_state=42, n_samples=200, n_features=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# We compute train-validation splits here instead of letting the sklearn interface do it
# such that we can compute the validation error ourselves
if n_cv == 1:
    # we cannot do 1-fold CV, so we do an 80%-20% train-validation split
    _, val_idxs = train_test_split(np.arange(X_train.shape[0]), test_size=0.2, random_state=0)
    val_idxs = val_idxs[None, :]
else:
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)
    val_idxs_list = [val_idxs for train_idxs, val_idxs in skf.split(X_train, y_train)]

    # make sure that each validation set has the same length, so we can exploit vectorization
    max_len = max([len(val_idxs) for val_idxs in val_idxs_list])
    val_idxs_list = [val_idxs[:max_len] for val_idxs in val_idxs_list]
    val_idxs = np.asarray(val_idxs_list)

best_val_loss = np.Inf
best_clf = None
best_params = None

for hpo_step in range(n_hyperopt_steps):
    # sample random params according to the proposed search space, but this can be replaced by a custom HPO method
    params = RealMLPParamSampler(is_classification=is_classification).sample_params(seed=hpo_step)

    # we only use one classifier that will fit n_cv sub-models, since RealMLP can vectorize the fitting,
    # but it would also be possible to use one classifier per cross-validation split.
    clf = RealMLP_TD_Classifier(**params, n_cv=n_cv, verbosity=2, val_metric_name='brier')
    clf.fit(X_train, y_train, val_idxs=val_idxs)

    # evaluate validation loss
    # for n_cv >= 2, predict_proba() only outputs averaged predictions of the cross-validation models,
    # but we need separate predictions of each of the cross-validation members to extract the out-of-bag ones,
    # so we use predict_proba_ensemble().
    # There is also predict_ensemble() which replaces predict().
    y_pred_prob = clf.predict_proba_ensemble(X_train)
    val_predictions = np.concatenate([y_pred_prob[i, val_idxs[i, :]] for i in range(n_cv)], axis=0)
    val_labels = np.concatenate([y_train[val_idxs[i, :]] for i in range(n_cv)], axis=0)
    val_logits = np.log(val_predictions + 1e-30)

    val_loss = Metrics.apply(torch.as_tensor(val_logits, dtype=torch.float32), torch.as_tensor(val_labels),
                             metric_name='brier').item()

    # update best model if loss improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_clf = clf
        best_params = params

best_clf.predict(X_test)
print(f'best params: {best_params}')
```

Here is the equivalent search space for `hyperopt`:
```python
from hyperopt import hp
import numpy as np

space = {
    'num_emb_type': hp.choice('num_emb_type', ['none', 'pbld', 'pl', 'plr']),
    'add_front_scale': hp.pchoice('add_front_scale', [(0.6, True), (0.4, False)]),
    'lr': hp.loguniform('lr', np.log(2e-2), np.log(3e-1)),
    'p_drop': hp.pchoice('p_drop', [(0.3, 0.0), (0.5, 0.15), (0.2, 0.3)]),
    'wd': hp.choice('wd', [0.0, 2e-2]),
    'plr_sigma': hp.loguniform('plr_sigma', np.log(0.05), np.log(0.5)),
    'hidden_sizes': hp.pchoice('hidden_sizes', [(0.6, [256] * 3), (0.2, [64] * 5), (0.2, [512])]),
    'act': hp.choice('act', ['selu', 'mish', 'relu']),
    'ls_eps': hp.pchoice('ls_eps', [(0.3, 0.0), (0.7, 0.1)])
}
```


