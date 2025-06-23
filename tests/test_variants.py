import pytest
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import ClassifierMixin

from pytabkit import TabM_D_Classifier, RealMLP_HPO_Classifier, Ensemble_HPO_Classifier, TabM_HPO_Regressor, \
    TabM_HPO_Classifier, LGBM_HPO_Classifier, CatBoost_HPO_Classifier, XGB_HPO_Classifier, Ensemble_HPO_Regressor, \
    LGBM_HPO_TPE_Regressor, RealMLP_TD_Regressor, RealMLP_HPO_Regressor, TabM_D_Regressor


@pytest.mark.parametrize('estimator', [
    RealMLP_TD_Regressor(n_cv=2, n_refit=2, n_repeats=2, device='cpu'),
    RealMLP_HPO_Regressor(device='cpu', n_hyperopt_steps=2, train_metric_name='multi_pinball(0.1,0.9)',
                          val_metric_name='multi_pinball(0.1,0.9)'),
    TabM_D_Classifier(val_metric_name='cross_entropy', num_emb_type='pwl', tabm_k=16, device='cpu', random_state=0),
    TabM_D_Regressor(val_metric_name='cross_entropy', num_emb_type='pwl', tabm_k=16, device='cpu', random_state=0),
    TabM_HPO_Regressor(val_metric_name='mae', n_hyperopt_steps=2, hpo_space_name='tabarena', device='cpu',
                       random_state=0),
    TabM_HPO_Classifier(val_metric_name='mae', n_hyperopt_steps=2, hpo_space_name='default', device='cpu',
                        random_state=0, use_caruana_ensembling=True),
    LGBM_HPO_Classifier(use_caruana_ensembling=True, n_hyperopt_steps=2, hpo_space_name='tabarena', device='cpu'),
    XGB_HPO_Classifier(use_caruana_ensembling=True, n_hyperopt_steps=2, hpo_space_name='tabarena', device='cpu'),
    CatBoost_HPO_Classifier(use_caruana_ensembling=True, n_hyperopt_steps=2, hpo_space_name='tabarena', device='cpu'),
    RealMLP_HPO_Classifier(val_metric_name='cross_entropy', n_hyperopt_steps=3, use_caruana_ensembling=True,
                           hpo_space_name='tabarena', n_caruana_steps=10, random_state=0, device='cpu'),
    Ensemble_HPO_Classifier(val_metric_name='brier', device='cpu', n_hpo_steps=2, use_full_caruana_ensembling=True,
                            use_tabarena_spaces=True),
    Ensemble_HPO_Regressor(val_metric_name='brier', device='cpu', n_hpo_steps=2, use_full_caruana_ensembling=True,
                           use_tabarena_spaces=True),
    LGBM_HPO_TPE_Regressor(n_cv=2, n_refit=2, n_hyperopt_steps=2),
])
def test_sklearn_not_crash(estimator):
    np.random.seed(0)
    n_train = 100
    X = pd.DataFrame({'a': np.random.randn(n_train), 'b': np.random.randint(5, size=(n_train,))})
    X['b'] = X['b'].astype('category')

    est = sklearn.base.clone(estimator)
    est.device = 'cpu'
    if isinstance(est, ClassifierMixin):
        y = np.random.randint(3, size=(n_train,))
    else:
        y = np.random.randn(n_train)

    est.fit(X, y)
    est.predict(X)
