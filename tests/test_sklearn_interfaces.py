import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pytabkit import XRFM_D_Classifier, XRFM_D_Regressor
from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier, RealMLP_TD_Regressor, \
    RealMLP_TD_S_Regressor, LGBM_TD_Classifier, LGBM_TD_Regressor, XGB_TD_Classifier, XGB_TD_Regressor, \
    CatBoost_TD_Classifier, \
    CatBoost_TD_Regressor, MLP_RTDL_D_Classifier, MLP_RTDL_D_Regressor, Resnet_RTDL_D_Classifier, TabR_S_D_Classifier, \
    Resnet_RTDL_D_Regressor, TabR_S_D_Regressor, TabM_D_Classifier, TabM_D_Regressor, MLP_PLR_D_Regressor, \
    MLP_PLR_D_Classifier, FTT_D_Classifier, FTT_D_Regressor, RealMLP_TD_S_Classifier


# decrease min_data_in_leaf for LGBMTDClassifier since otherwise the test check_classifiers_classes fails,
# because LGBM only predicts a single class on the training set
# also increase subsample to 1.0 because otherwise LightGBM fails with n_samples=1.
@parametrize_with_checks([
    XRFM_D_Classifier(device='cpu'), XRFM_D_Regressor(device='cpu'),
    LGBM_TD_Classifier(min_data_in_leaf=2, subsample=1.0, calibration_method='ts-mix', val_metric_name='ref-ll-ts',
                       n_estimators=100),
    LGBM_TD_Classifier(min_data_in_leaf=2, subsample=1.0, n_estimators=100),
    LGBM_TD_Regressor(subsample=1.0, n_estimators=100),
    XGB_TD_Classifier(n_estimators=100), XGB_TD_Regressor(n_estimators=100),
    CatBoost_TD_Classifier(n_estimators=100), CatBoost_TD_Regressor(n_estimators=100),
    # use CPU to avoid Mac OS errors with MPS backend
    RealMLP_TD_Classifier(n_epochs=8, device='cpu'), RealMLP_TD_Regressor(n_epochs=64, device='cpu'),
    TabM_D_Classifier(device='cpu', tabm_k=2, num_emb_type='pwl', arch_type='tabm-mini', num_emb_n_bins=2),
    TabM_D_Regressor(device='cpu', tabm_k=2, num_emb_type='pwl', arch_type='tabm-mini', num_emb_n_bins=2),
    MLP_RTDL_D_Classifier(device='cpu', max_epochs=50), #MLP_RTDL_D_Regressor(device='cpu'),
    Resnet_RTDL_D_Classifier(device='cpu'), Resnet_RTDL_D_Regressor(device='cpu'),
    MLP_PLR_D_Classifier(device='cpu'), MLP_PLR_D_Regressor(device='cpu'),
    FTT_D_Classifier(device='cpu', module_d_token=128, module_n_heads=8, max_epochs=32),
    FTT_D_Regressor(device='cpu', module_d_token=128, module_n_heads=8, max_epochs=32),
    # Tabr_D_Classifier(), Tabr_D_Regressor(),  # needs faiss which is not in the dependencies, so don't test
                          ])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)

