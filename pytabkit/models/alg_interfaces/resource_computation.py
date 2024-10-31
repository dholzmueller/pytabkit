import time
from collections.abc import Callable
from typing import Dict, Union, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pytabkit.models.alg_interfaces.base import RequiredResources
from pytabkit.models import utils
from pytabkit.models.data.data import DictDataset, TensorInfo
from pytabkit.models.nn_models.models import PreprocessingFactory
from pytabkit.models.training.metrics import pinball_loss


# This file contains code to predict required resources (time and RAM) of a ML model on a dataset.
# There are two components:
# - Computing the predicted resources based on a linear model on raw and product features
# - Fitting the linear model coefficients based on evaluations on random parameters.


def get_resource_features(config: Dict, ds: DictDataset, n_cv: int, n_refit: int,
                          n_splits: int, **extra_params) -> Dict[str, float]:
    """
    Extracts features that can be used in a linear model for predicting resource usage.
    """
    # in hyperopt method also on number of steps (for time estimation)
    tensor_infos = ds.tensor_infos
    n_samples = ds.n_samples
    n_classes = tensor_infos['y'].get_cat_size_product()
    prep_factory = PreprocessingFactory(**config)
    onehot_factory = PreprocessingFactory(tfms=['one_hot'])
    fitter, out_tensor_infos = prep_factory.create_transform(tensor_infos)
    _, onehot_tensor_infos = onehot_factory.create_transform(tensor_infos)
    n_features = sum([ti.get_n_features() for key, ti in out_tensor_infos.items()
                      if key in ['x_cont', 'x_cat']])
    ds_prep = DictDataset(tensors=None, tensor_infos=out_tensor_infos, device=ds.device, n_samples=n_samples)
    ds_onehot = DictDataset(tensors=None, tensor_infos=onehot_tensor_infos, device=ds.device, n_samples=n_samples)
    cat_size_sum = 0 if 'x_cat' not in out_tensor_infos else out_tensor_infos['x_cat'].get_cat_sizes().sum().item()
    n_classes = ds.tensor_infos['y'].get_cat_size_product()
    n_cat = ds.tensor_infos['x_cat'].get_n_features()

    ds_size_gb = ds.get_size_gb()
    ds_prep_size_gb = ds_prep.get_size_gb()
    ds_onehot_size_gb = ds_onehot.get_size_gb()

    n_tree_repeats = 1 if n_classes <= 2 else n_classes

    features = dict()
    features['1/n_threads'] = 1 / config.get('n_threads', 1)
    features['ds_size_gb'] = ds_size_gb
    features['ds_prep_size_gb'] = ds_prep_size_gb
    features['ds_onehot_size_gb'] = ds_onehot_size_gb
    features['n_features'] = n_features
    features['n_samples'] = n_samples
    features['n_tree_repeats'] = n_tree_repeats
    features['n_cv_refit'] = n_cv + n_refit
    features['n_splits'] = n_splits
    max_depth = config.get('max_depth', 6)
    if isinstance(max_depth, int):
        features['2_power_maxdepth'] = 2 ** max_depth
    features['log_num_leaves'] = np.log(max(1, config.get('num_leaves', 31)))
    features['cat_size_sum'] = cat_size_sum
    features['n_classes'] = n_classes
    features['n_cat'] = n_cat

    return utils.join_dicts(config, features, extra_params)


def process_resource_features(raw_features: Dict[str, Any], feature_spec: List[str]):
    """
    Adds product features to raw features.
    :param raw_features: Raw feature values
    :param feature_spec: List of strings. Each string should be of the form 'feature_1*...*feature_n',
        using the names of the features whose products should be added
    :return: Returns a dictionary of the raw features along with the newly computed product features.
    """
    results = dict()
    for combination in feature_spec:
        # ignore empty factors
        factors = [factor for factor in combination.split('*') if factor != '']
        value = 1.0
        for factor in factors:
            value *= raw_features[factor]
        results[combination] = value
    return results


def eval_linear_product_model(raw_features: Dict[str, Any], params: Dict[str, float]):
    """
    Computes the "inner product" between the feature dictionaries
    (obtained from raw features and products according to the keys in params).
    :return:
    """
    result = 0.0
    for key, param in params.items():
        # ignore empty factors
        factors = [factor for factor in key.split('*') if factor != '']
        value = 1.0
        for factor in factors:
            value *= raw_features[factor]
        result += param * value
    return result


class FeatureSpec:
    """
    Allows to create a list of product feature names from product and powerset operations etc.
    """
    @staticmethod
    def _listify(spec: Union[List, str]):
        if isinstance(spec, list):
            return spec
        elif isinstance(spec, str):
            return [spec]
        else:
            raise ValueError(f'Unsupported spec type {type(spec)}')

    @staticmethod
    def _product_str(first: str, second: str) -> str:
        if len(first) == 0:
            if len(second) == 0:
                return ''
            else:
                return second
        else:
            if len(second) == 0:
                return first
            else:
                return f'{first}*{second}'

    @staticmethod
    def concat(*feature_specs):
        feature_specs = [FeatureSpec._listify(spec) for spec in feature_specs]
        flattened = [spec for lst in feature_specs for spec in lst]
        return flattened

    @staticmethod
    def product(*feature_specs):
        if len(feature_specs) <= 0:
            raise ValueError()
        elif len(feature_specs) == 1:
            return FeatureSpec._listify(feature_specs[0])
        else:
            first, rest = feature_specs[0], feature_specs[1:]
            first_list = FeatureSpec._listify(first)
            rest_product = FeatureSpec.product(*rest)
            return [FeatureSpec._product_str(first_spec, rest_spec)
                    for first_spec in first_list for rest_spec in rest_product]

    @staticmethod
    def powerset_products(*feature_specs):
        if len(feature_specs) == 0:
            return ['']
        elif len(feature_specs) == 1:
            return FeatureSpec.concat('', feature_specs[0])
        else:
            return FeatureSpec.product(FeatureSpec.concat('', feature_specs[0]),
                                       FeatureSpec.powerset_products(*feature_specs[1:]))


# some code for linear regression with different losses, to estimate coefficients for resource prediction

class NormalizedDataRegressor:
    def __init__(self, sub_regressor):
        self.sub_regressor = sub_regressor

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.x_norms_ = np.sqrt(np.mean(X ** 2, axis=0))
        self.y_norm_ = np.sqrt(np.mean(y ** 2))
        self.sub_regressor.fit(X / self.x_norms_[None, :], y / self.y_norm_)

    def get_coefs(self) -> np.ndarray:
        return self.sub_regressor.get_coefs() * self.y_norm_ / self.x_norms_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.sub_regressor.predict(X / self.x_norms_) * self.y_norm_


class LogLinearModule(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.params = nn.Parameter(torch.zeros(n_features, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ torch.exp(self.params)


class LogLinearRegressor:
    def __init__(self, pessimistic: bool):
        self.pessimistic = pessimistic

    def fit(self, X: np.ndarray, y: np.ndarray):
        x = torch.as_tensor(X, dtype=torch.float64)
        y = torch.as_tensor(y, dtype=torch.float64)
        y_log = torch.log(y + 1e-8)
        n_features = x.shape[1]
        self.model_ = LogLinearModule(n_features=n_features)
        opt = torch.optim.Adam(params=self.model_.parameters(), betas=(0.9, 0.95))

        n_it = 10000
        max_lr = 1e-1

        for i in range(n_it):
            for param_group in opt.param_groups:
                # linearly decaying lr schedule
                param_group['lr'] = (1 - i / n_it) * max_lr
            y_pred_log = torch.log(self.model_(x))
            if self.pessimistic:
                loss = pinball_loss(torch.exp(y_pred_log), y, quantile=0.99)
            else:
                loss = ((y_pred_log - y_log) ** 2).mean()
            if i % (n_it // 10) == 0:
                print(f'Loss: {loss.item():g}')
            loss.backward()
            opt.step()
            opt.zero_grad()

    def get_coefs(self) -> np.ndarray:
        return np.exp(self.model_.params.detach().numpy())


def fit_resource_factors(data: List[Tuple[Dict[str, float], float]], pessimistic: bool, coef_factor: float = 1.0):
    feature_names = list(data[0][0].keys())
    y = np.asarray([data[i][1] for i in range(len(data))])
    X = np.asarray([[data[i][0][feature_names[j]] for j in range(len(feature_names))] for i in range(len(data))])

    # transform data set to implicitly learn with relative mse
    # ((y_pred - y)/y)^2 = ((X/y)c - 1)^2
    # X = X / y[:, None]
    # y = np.ones_like(y)

    # coefs: np.ndarray = np.linalg.lstsq(X, y)[0]
    # always use pessimistic version
    reg = NormalizedDataRegressor(LogLinearRegressor(pessimistic=True))
    reg.fit(X, y)
    coefs = reg.get_coefs()
    coefs[coefs < 0.0] = 0.0

    if pessimistic:
        # rescale to a bit larger than the maximum on the training set
        y_pred = X @ coefs
        coefs *= coef_factor * np.max(y / y_pred)
    else:
        y_pred = X @ coefs
        coefs *= np.mean(y) / np.mean(y_pred)
        # # align their geometric means
        # coefs *= np.exp(np.mean(np.log(y)) - np.mean(np.log(y_pred)))
    return {name: coef for name, coef in zip(feature_names, coefs)}


class TimeWrapper:
    def __init__(self, f: Callable):
        self.f = f

    def __call__(self):
        start_time = time.time()
        self.f()
        end_time = time.time()
        return end_time - start_time


def create_ds(n_samples: int, n_cont: int, n_cat: int, cat_size: int, n_classes: int) -> DictDataset:
    torch.manual_seed(0)
    x_cont = torch.randn(n_samples, n_cont)
    x_cont_info = TensorInfo(feat_shape=[n_cont])
    x_cat = torch.randint(0, cat_size, size=(n_samples, n_cat))
    x_cat_info = TensorInfo(cat_sizes=[cat_size] * n_cat)
    if n_classes > 0:
        y = torch.randint(0, n_classes, size=(n_samples, 1))
        y_info = TensorInfo(cat_sizes=[n_classes])
    else:
        y = torch.randn(n_samples, 1)
        y_info = TensorInfo(feat_shape=[1])
    return DictDataset(tensors=dict(x_cont=x_cont, x_cat=x_cat, y=y),
                       tensor_infos=dict(x_cont=x_cont_info, x_cat=x_cat_info, y=y_info))


class Sampler:
    def sample(self) -> Union[int, float]:
        raise NotImplementedError()


class UniformSampler(Sampler):
    def __init__(self, low: Union[int, float], high: Union[int, float], log=False, is_int=False):
        self.low = low
        self.high = high
        self.log = log
        self.is_int = is_int

    def sample(self) -> Union[int, float]:
        low = self.low
        high = self.high + 1 if self.is_int else self.high  # in the integer case, make the upper bound inclusive
        if self.log:
            sample = np.exp(np.random.uniform(np.log(low), np.log(high)))
        else:
            sample = np.random.uniform(low, high)
        return int(sample) if self.is_int else sample


# class ChoiceSampler:
#     def __init__(self):


def ds_to_xy(ds: DictDataset) -> Tuple[pd.DataFrame, np.ndarray]:
    X = ds.without_labels().to_df()
    y = ds.tensors['y'].numpy()
    return X, y


class ResourcePredictor:
    """
    Predicts resource usages based on a linear model on raw and product features.
    """
    def __init__(self, config: Dict[str, Any], time_params: Dict[str, float], cpu_ram_params: Dict[str, float],
                 gpu_ram_params: Optional[Dict[str, float]] = None, n_gpus: int = 0, gpu_usage: float = 1.0):
        """
        :param config: Configuration parameters.
        :param time_params: Coefficients for the linear model for time prediction.
        :param cpu_ram_params: Coefficients for the linear model for CPU RAM prediction.
        :param gpu_ram_params: Coefficients for the linear model for GPU RAM prediction.
        :param n_gpus: Number of GPUs that should be used.
        :param gpu_usage: Usage level of each GPU (between 0 and 1).
        """
        self.config = config
        self.time_params = time_params
        self.cpu_ram_params = cpu_ram_params
        self.gpu_ram_params = gpu_ram_params
        self.n_gpus = n_gpus
        self.gpu_usage = gpu_usage

    def get_required_resources(self, ds: DictDataset, **extra_params) -> RequiredResources:
        """
        Function that provides an estimate of the required resources
        :param ds: Dataset (does not need to contain the tensors, just the n_samples and tensor_infos)
        :return: RequiredResources estimate.
        """
        # in hyperopt method also on number of steps
        # moreover it should depend on n_threads, and scaling law should be able to be configured
        # should allow n_threads to depend on the task_info  (based on certain thresholds and possibly scaling law)
        # include a time_factor depending on the method
        n_samples = ds.n_samples
        n_classes = ds.tensor_infos['y'].get_cat_sizes()[0].item()

        ds = DictDataset(tensors=None, tensor_infos=ds.tensor_infos, device='cpu', n_samples=ds.n_samples)
        raw_features_prelim = get_resource_features(self.config, ds, n_cv=1, n_refit=0, n_splits=1, **extra_params)
        n_features = raw_features_prelim['n_features']

        if 'n_threads' in self.config:
            n_threads = self.config['n_threads']
        else:
            # for dionis, it's roughly 100k * 60 * 355 = 2_130_000_000
            # for robert it's 10k * 7200 * 10 = 720_000_000
            # for indoor_loc_building it's roughly 20k * 520 * 3 = 31_200_000
            ds_complexity = n_samples * n_features * n_classes
            thresh = self.config.get('single_thread_complexity_threshold', 200_000_000)
            # n_threads = min(self.config.get('max_complexity_threads', 128), 1 + int(ds_complexity / thresh))
            n_threads = 1 + int(ds_complexity / thresh)

            config = utils.update_dict(self.config, dict(n_threads=n_threads))
            raw_features = get_resource_features(config, ds, n_cv=1, n_refit=0, n_splits=1, **extra_params)
            cpu_ram_gb = eval_linear_product_model(raw_features, self.cpu_ram_params)

            min_threads_per_gb = self.config.get('min_threads_per_gb', 0.3)
            n_threads = min(self.config.get('max_n_threads', 8), max(n_threads, int(min_threads_per_gb * cpu_ram_gb)))

        config = utils.update_dict(self.config, dict(n_threads=n_threads))
        raw_features = get_resource_features(config, ds, n_cv=1, n_refit=0, n_splits=1, **extra_params)
        time_s = eval_linear_product_model(raw_features, self.time_params)
        cpu_ram_gb = eval_linear_product_model(raw_features, self.cpu_ram_params)
        gpu_ram_gb = 0.0 if self.gpu_ram_params is None \
            else eval_linear_product_model(raw_features, self.gpu_ram_params)

        # todo: rough correction to prioritize dionis even if it's run with too many threads,
        #  should use better time estimation model
        time_s += 0.2 * n_threads * time_s

        return RequiredResources(time_s=time_s,
                                 n_threads=n_threads,
                                 cpu_ram_gb=cpu_ram_gb,
                                 gpu_ram_gb=gpu_ram_gb,
                                 n_gpus=self.n_gpus,
                                 gpu_usage=0.0 if self.n_gpus == 0 else self.gpu_usage)


# if __name__ == '__main__':
#     features = FeatureSpec.concat('', 'ds_size_gb',
#                                   FeatureSpec.product('n_cv_refit', 'n_splits',
#                                                       FeatureSpec.powerset_products('1/n_threads', 'n_features',
#                                                                                     'n_samples',
#                                                                                     'n_estimators', 'n_tree_repeats')))
#     print(features)
#     print(f'{len(features)=}')
