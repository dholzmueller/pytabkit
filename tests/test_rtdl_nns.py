import numpy as np
import pandas as pd
from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from pytabkit.models.sklearn.sklearn_interfaces import Resnet_RTDL_D_Classifier, Resnet_RTDL_D_Regressor, \
    MLP_RTDL_D_Classifier, MLP_RTDL_D_Regressor, FTT_D_Classifier, FTT_D_Regressor
from sklearn.datasets import make_classification, make_regression
import pytest
import torch
# def test_estimator_compliance():
#     # Check if the custom estimators comply with scikit-learn's conventions
#     check_estimator(Resnet_RTDL_D_Classifier())
#     check_estimator(Resnet_RTDL_D_Regressor())

# @pytest.mark.parametrize("n_classes", [2, 3])
# @pytest.mark.parametrize("model_name", ["resnet", "mlp", "ft_transformer"])
# def test_numerical_data(n_classes, model_name):
#     # Generate synthetic data
#     X, y = make_classification(n_samples=1000, n_features=20, n_informative=3,n_classes=n_classes, random_state=42)
#     X = pd.DataFrame(X)
#     y = pd.Series(y)
#
#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # Train the classifier
#     if model_name == "resnet":
#         clf = Resnet_RTDL_D_Classifier(device="cpu")
#     elif model_name == "mlp":
#         clf = MLP_RTDL_D_Classifier(device="cpu")
#     elif model_name == "ft_transformer":
#         clf = FTT_D_Classifier(device="cpu")
#     clf.fit(X_train, y_train, cat_indicator=[False] * 20)  # Assuming no categorical features
#
#     # Predict and evaluate
#     predictions = clf.predict(X_test)
#     accuracy = accuracy_score(y_test, predictions)
#     assert accuracy > 0.5, "Accuracy should be greater than 50%"
#
#
# @pytest.mark.parametrize("n_classes", [2, 3])
# @pytest.mark.parametrize("model_name", ["resnet", "mlp", "ft_transformer"])
# def test_categorical_data(n_classes, model_name):
#     # Generate synthetic data with a categorical feature
#     X, y = make_classification(n_samples=1000, n_features=20, n_informative=3, n_classes=n_classes, random_state=42)
#     # Add a categorical feature
#     cat_col = np.random.choice([0, 1, 2], size=X.shape[0])
#     X = np.hstack((X, cat_col.reshape(-1, 1)))
#
#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # Train the classifier with categorical feature
#     if model_name == "resnet":
#         clf = Resnet_RTDL_D_Classifier(device="cpu")
#     elif model_name == "mlp":
#         clf = MLP_RTDL_D_Classifier(device="cpu")
#     elif model_name == "ft_transformer":
#         clf = FTT_D_Classifier(device="cpu")
#     clf.fit(X_train, y_train, cat_indicator=[False] * 20 + [True])
#
#     # Predict and evaluate
#     predictions = clf.predict(X_test)
#     accuracy = accuracy_score(y_test, predictions)
#     assert accuracy > 0.5, "Accuracy should be greater than 50%"
#
#     # Check if the classifier can handle unseen categories
#     X_test[0, -1] = -1  # Unseen category
#     predictions = clf.predict(X_test)
#     # If no error is raised, the classifier can handle unseen categories
#
# @pytest.mark.parametrize("tranformed_target", [True, False])
# @pytest.mark.parametrize("model_name", ["resnet", "mlp", "ft_transformer"])
# def test_regressor_numerical_categorical(tranformed_target, model_name):
#     # Generate synthetic data with a mix of numerical and categorical features
#     X, y = make_regression(n_samples=1000, n_features=3, n_informative=2, random_state=43)
#     cat_feature = np.random.choice([1, 2, 3], size=X.shape[0])
#     X = np.column_stack((X, cat_feature))
#
#     X = pd.DataFrame(X, columns=[f"num_{i}" for i in range(X.shape[1] - 1)] + ['cat'])
#     cat_features = [False]*3 + [True]
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
#
#     if model_name == "resnet":
#         regressor = Resnet_RTDL_D_Regressor(transformed_target=tranformed_target, random_state=41, device="cpu")
#     elif model_name == "mlp":
#         regressor = MLP_RTDL_D_Regressor(transformed_target=tranformed_target, random_state=41, device="cpu")
#     elif model_name == "ft_transformer":
#         regressor = FTT_D_Regressor(transformed_target=tranformed_target, random_state=41, device="cpu")
#     regressor.fit(X_train, y_train, cat_indicator=cat_features)
#     predictions = regressor.predict(X_test)
#
#     # Evaluate the regressor with R2 score
#     score = r2_score(y_test, predictions)
#     assert score > 0.1, f"Regressor R2 score too low with mixed features, got {score}"
#
#     # Test handling of unseen categories
#     X_test.iloc[0, -1] = 4  # Introduce a new category
#     predictions = regressor.predict(X_test)
#     # If no errors and predictions are returned, the regressor can handle unseen categories during test time
#
#
# def create_model(regression, model_name, **kwargs):
#     if model_name == "resnet":
#         model = Resnet_RTDL_D_Regressor(device="cpu", **kwargs) if regression else Resnet_RTDL_D_Classifier(device="cpu", **kwargs)
#     elif model_name == "mlp":
#         model = MLP_RTDL_D_Regressor(device="cpu", **kwargs) if regression else MLP_RTDL_D_Classifier(device="cpu", **kwargs)
#     elif model_name == "ft_transformer":
#         model = FTT_D_Regressor(device="cpu", **kwargs) if regression else FTT_D_Classifier(device="cpu", **kwargs)
#     return model
#
#
# # @pytest.mark.parametrize("regression", [True, False])
# # @pytest.mark.parametrize("resnet_or_mlp", ["resnet", "mlp"])
# # def test_determinist(regression, resnet_or_mlp):
# #     # generate toy data
# #     if regression:
# #         X, y = make_regression(n_samples=300, n_features=20, n_informative=2, random_state=42)
# #     else:
# #         X, y = make_classification(n_samples=300, n_features=20, n_informative=2, random_state=42)
# #
# #     # add categorical feature
# #     cat_feature = np.random.choice([1, 2, 3], size=X.shape[0])
# #     X = np.column_stack((X, cat_feature))
# #
# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #
# #
# #     random_states = [42, 42, 43]
# #     res_list = []
# #     for random_state in random_states:
# #         model = create_model(regression, resnet_or_mlp, random_state=random_state)
# #         model.fit(X_train, y_train, cat_features=[False]*20 + [True])
# #         predictions = model.predict(X_test)
# #         res_list.append(predictions)
# #
# #     assert np.allclose(res_list[0], res_list[1]), "Predictions should be the same with the same random_state"
# #     assert not np.allclose(res_list[0], res_list[2]), "Predictions should be different with different random_state"
#
#
# @pytest.mark.parametrize("regression", [True, False])
# @pytest.mark.parametrize("model_name", ["resnet", "mlp", "ft_transformer"])
# @pytest.mark.parametrize("n_classes", [2, 3])
# def test_all_categorical(regression, model_name, n_classes):
#     X = np.random.randint(n_classes, size=(1000, 10))
#     if regression:
#         y = np.random.rand(1000)
#     else:
#         y = np.random.randint(n_classes, size=(1000,))
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
#     model = create_model(regression, model_name, random_state=42)
#     model.fit(X_train, y_train, cat_indicator=[True] * 10)
#
#     model.predict(X_test)
#
#
# @pytest.mark.parametrize("seed", list(range(10)))
# @pytest.mark.parametrize("model_name", ["resnet", "mlp", "ft_transformer"])
# def test_high_cardinality(seed, model_name):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#
#     x_df = pd.DataFrame({'cat_1': [270, 86, 154, 80, 56, 80, 80, 283, 199, 291]}).astype('category')
#     y = np.zeros(len(x_df))
#
#     reg = create_model(True, model_name, random_state=seed)
#     reg.fit(x_df, y, cat_indicator=[True])
#     reg.predict(x_df)
#
#
# # @pytest.mark.parametrize("resnet_or_mlp", ["resnet", "mlp"])
# # @pytest.mark.parametrize("transformed_target", [True, False])
# # def test_constant_predictor(resnet_or_mlp, transformed_target):
# #     # test that the prediction are replaced by the mean of the training set if the val loss
# #     # is infinite or too bad
# #     X, y = make_regression(n_samples=1000, n_features=20, n_informative=2, random_state=42)
# #
# #     # first lr 3 to get bad but finite val_loss
# #     model = create_model(True, resnet_or_mlp, random_state=42, lr=1, max_epochs=10, transformed_target=transformed_target)
# #     model.fit(X, y, val_idxs=np.arange(100))
# #     # check that val_loss is finite
# #     history = model.alg_interface_.sub_split_interfaces[0].model.history
# #     assert np.isfinite(history[:, 'valid_loss']).any()
# #     predictions = model.predict(X)
# #     assert np.allclose(predictions, np.mean(y[100:])), "Predictions should be the mean of the training set"
# #     # this should also correspond to model.alg_interface_.sub_split_interfaces[0].model.y_train_mean if transformed_target=False
# #     if not transformed_target:
# #         assert np.allclose(model.alg_interface_.sub_split_interfaces[0].model.y_train_mean, np.mean(y[100:]))
# #     assert model.alg_interface_.sub_split_interfaces[0].model.predict_mean == True
# #
# #     # now lr 1000 to get bad but infinite val_loss
# #     model = create_model(True, resnet_or_mlp, random_state=42, lr=10000, max_epochs=10, transformed_target=transformed_target)
# #     model.fit(X, y, val_idxs=np.arange(100))
# #     # check that val_loss is infinite
# #     history = model.alg_interface_.sub_split_interfaces[0].model.history
# #     assert ~np.isfinite(history[:, 'valid_loss']).all()
# #     predictions = model.predict(X)
# #     assert np.allclose(predictions, np.mean(y[100:])), "Predictions should be the mean of the training set"
# #     # this should also correspond to model.alg_interface_.sub_split_interfaces[0].model.y_train_mean if transformed_target=False
# #     if not transformed_target:
# #         assert np.allclose(model.alg_interface_.sub_split_interfaces[0].model.y_train_mean, np.mean(y[100:]))
# #     assert model.alg_interface_.sub_split_interfaces[0].model.predict_mean == True
# #
# #     # now lr=1e-5 to check that the predictions are not replaced by the mean of the training set
# #     model = create_model(True, resnet_or_mlp, random_state=42, lr=1e-5, max_epochs=10, transformed_target=transformed_target)
# #     model.fit(X, y, val_idxs=np.arange(100))
# #     # check that val_loss is finite
# #     history = model.alg_interface_.sub_split_interfaces[0].model.history
# #     assert np.isfinite(history[:, 'valid_loss']).any()
# #     predictions = model.predict(X)
# #     assert not np.allclose(predictions, np.mean(y[100:])), "Predictions should not be the mean of the training set"
# #     assert model.alg_interface_.sub_split_interfaces[0].model.predict_mean == False
