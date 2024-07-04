import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier, RealMLP_TD_Regressor, \
    RealMLP_TD_S_Regressor, LGBM_TD_Classifier, LGBM_TD_Regressor, XGB_TD_Classifier, XGB_TD_Regressor, CatBoost_TD_Classifier, \
    CatBoost_TD_Regressor, MLP_RTDL_D_Classifier, MLP_RTDL_D_Regressor, Resnet_RTDL_D_Classifier, TabR_S_D_Classifier, \
    Resnet_RTDL_D_Regressor, TabR_S_D_Regressor


# decrease min_data_in_leaf for LGBMTDClassifier since otherwise the test check_classifiers_classes fails,
# because LGBM only predicts a single class on the training set
# also increase subsample to 1.0 because otherwise LightGBM fails with n_samples=1.
@parametrize_with_checks([
    LGBM_TD_Classifier(min_data_in_leaf=2, subsample=1.0), LGBM_TD_Regressor(subsample=1.0),
    XGB_TD_Classifier(), XGB_TD_Regressor(),
    CatBoost_TD_Classifier(), CatBoost_TD_Regressor(),
    RealMLP_TD_Classifier(n_epochs=8), RealMLP_TD_Regressor(n_epochs=64),
    # MLP_RTDL_D_Classifier(), MLP_RTDL_D_Regressor(),
    # Resnet_RTDL_D_Classifier(), Resnet_RTDL_D_Regressor(),
    # Tabr_D_Classifier(), Tabr_D_Regressor(),
                          ])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)

