import pytest
import sklearn.base
import numpy as np

from pytabkit import Ensemble_TD_Classifier, Ensemble_TD_Regressor
from pytabkit.models.sklearn.sklearn_interfaces import Ensemble_HPO_Classifier, Ensemble_HPO_Regressor


@pytest.mark.parametrize('model', [
    Ensemble_TD_Classifier(calibration_method='ts-mix', val_metric_name='ref-ll-ts', device='cpu'),
    Ensemble_TD_Regressor(device='cpu'),
    Ensemble_HPO_Classifier(calibration_method='ts-mix',
                            val_metric_name='ref-ll-ts', n_hpo_steps=1, device='cpu'),
    Ensemble_HPO_Regressor(n_hpo_steps=1, device='cpu'),
    ])
def test_ensemble(model):
    np.random.seed(0)
    X = np.random.randn(100, 2)
    y = np.random.randn(100, 1)
    if sklearn.base.is_classifier(model):
        y = y > 0.0
    model.fit(X, y)
    model.predict(X)
