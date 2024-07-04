import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from pytabkit.models.sklearn.sklearn_interfaces import TabR_S_D_Classifier, TabR_S_D_Regressor
from sklearn.datasets import make_classification, make_regression
import pytest
import torch

# tests are currently not executed since TabR needs faiss which is not available via pip,
# therefore it cannot run via hatch test / in CI


# @pytest.mark.parametrize("n_classes", [2, 3])
# def test_numerical_data(n_classes):
#     # Generate synthetic data
#     X, y = make_classification(n_samples=1000, n_features=20, n_informative=3,n_classes=n_classes, random_state=42)
#     X = pd.DataFrame(X)
#     y = pd.Series(y)
#
#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # Train the classifier
#     clf = TabR_S_D_Classifier(n_epochs=5)
#     clf.fit(X_train, y_train, cat_features=[False] * 20)  # Assuming no categorical features
#
#     # Predict and evaluate
#     predictions = clf.predict(X_test)
#     accuracy = accuracy_score(y_test, predictions)
#     assert accuracy > 0.5, "Accuracy should be greater than 50%"
#
#
# @pytest.mark.parametrize("n_classes", [2, 3])
# def test_categorical_data(n_classes):
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
#     clf = TabR_S_D_Classifier(n_epochs=5)
#     clf.fit(X_train, y_train, cat_features=[False] * 20 + [True])
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
#
# @pytest.mark.parametrize("tranformed_target", [True, False])
# def test_regressor_numerical_categorical(tranformed_target):
#     # Generate synthetic data with a mix of numerical and categorical features
#     X, y = make_regression(n_samples=1000, n_features=5, n_informative=3, random_state=42)
#     cat_feature = np.random.choice([1, 2, 3], size=X.shape[0])
#     X = np.column_stack((X, cat_feature))
#
#     X = pd.DataFrame(X, columns=[f"num_{i}" for i in range(X.shape[1] - 1)] + ['cat'])
#     cat_features = [False]*5 + [True]
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # Train the regressor
#     regressor = TabR_S_D_Regressor(n_epochs=20, transformed_target=tranformed_target)
#     regressor.fit(X_train, y_train, cat_features=cat_features)
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
# @pytest.mark.parametrize("regression", [True, False])
# def test_determinist(regression):
#     # generate toy data
#     if regression:
#         X, y = make_regression(n_samples=300, n_features=20, n_informative=2, random_state=42)
#     else:
#         X, y = make_classification(n_samples=300, n_features=20, n_informative=2, random_state=42)
#
#     # add categorical feature
#     cat_feature = np.random.choice([1, 2, 3], size=X.shape[0])
#     X = np.column_stack((X, cat_feature))
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     random_states = [42, 42, 43]
#     res_list = []
#     for random_state in random_states:
#         if regression:
#             model = TabR_S_D_Regressor(random_state=random_state, n_epochs=5)
#         else:
#             model = TabR_S_D_Classifier(random_state=random_state, n_epochs=5)
#         model.fit(X_train, y_train, cat_features=[False]*20 + [True])
#         predictions = model.predict(X_test)
#         res_list.append(predictions)
#
#     assert np.allclose(res_list[0], res_list[1]), "Predictions should be the same with the same random_state"
#     assert not np.allclose(res_list[0], res_list[2]), "Predictions should be different with different random_state"
#
#
# @pytest.mark.parametrize("regression", [True, False])
# @pytest.mark.parametrize("n_classes", [2, 3])
# @pytest.mark.parametrize("cat_size", [2, 5])
# def test_all_categorical(regression, n_classes, cat_size):
#     X = np.random.randint(cat_size, size=(1000, 10))
#     if regression:
#         y = np.random.rand(1000)
#     else:
#         y = np.random.randint(n_classes, size=(1000,))
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
#     model = TabR_S_D_Regressor(n_epochs=5) if regression else TabR_S_D_Classifier(n_epochs=5)
#     model.fit(X_train, y_train, cat_features=[True] * 10)
#
#     model.predict(X_test)
#
#
# @pytest.mark.parametrize("seed", list(range(10)))
# def test_high_cardinality(seed):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#
#     x_df = pd.DataFrame({'cat_1': [270, 86, 154, 80, 56, 80, 80, 283, 199, 291]}).astype('category')
#     y = np.zeros(len(x_df))
#
#     reg = TabR_S_D_Regressor(n_epochs=5)
#     reg.fit(x_df, y, cat_features=[True])
#     reg.predict(x_df)
