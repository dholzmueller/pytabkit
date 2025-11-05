import pathlib
from typing import Optional, Any, Union, List, Dict, Literal

import numpy as np

from pytabkit.models import utils
from pytabkit.models.sklearn.default_params import DefaultParams
from pytabkit.models.sklearn.sklearn_base import AlgInterfaceRegressor, AlgInterfaceClassifier
from pytabkit.models.alg_interfaces.sub_split_interfaces import SingleSplitWrapperAlgInterface
from pytabkit.models.alg_interfaces.alg_interfaces import AlgInterface
from pytabkit.models.alg_interfaces.nn_interfaces import NNAlgInterface, RandomParamsNNAlgInterface
from pytabkit.models.alg_interfaces.ensemble_interfaces import AlgorithmSelectionAlgInterface, \
    CaruanaEnsembleAlgInterface

# the list of methods can be auto-generated using scripts/get_sklearn_names.py
__all__ = ["CatBoost_D_Classifier", "CatBoost_D_Regressor", "CatBoost_HPO_Classifier",
           "CatBoost_HPO_Regressor", "CatBoost_HPO_TPE_Classifier", "CatBoost_HPO_TPE_Regressor",
           "CatBoost_TD_Classifier",
           "CatBoost_TD_Regressor", "Ensemble_TD_Classifier", "Ensemble_TD_Regressor", "Ensemble_HPO_Classifier",
           "Ensemble_HPO_Regressor", "FTT_D_Classifier",
           "FTT_D_Regressor",
           "FTT_HPO_Classifier", "FTT_HPO_Regressor", "LGBM_D_Classifier", "LGBM_D_Regressor", "LGBM_HPO_Classifier",
           "LGBM_HPO_Regressor",
           "LGBM_HPO_TPE_Classifier", "LGBM_HPO_TPE_Regressor", "LGBM_TD_Classifier", "LGBM_TD_Regressor",
           "MLP_PLR_D_Classifier",
           "MLP_PLR_D_Regressor", "MLP_PLR_HPO_Classifier", "MLP_PLR_HPO_Regressor", "MLP_RTDL_D_Classifier",
           "MLP_RTDL_D_Regressor",
           "MLP_RTDL_HPO_Classifier", "MLP_RTDL_HPO_Regressor", "MLP_SKL_D_Classifier", "MLP_SKL_D_Regressor",
           "RF_HPO_Classifier",
           "RF_HPO_Regressor", "RF_SKL_D_Classifier", "RF_SKL_D_Regressor", "RealMLP_HPO_Classifier",
           "RealMLP_HPO_Regressor",
           "RealMLP_TD_Classifier", "RealMLP_TD_Regressor", "RealMLP_TD_S_Classifier", "RealMLP_TD_S_Regressor",
           "RealTabR_D_Classifier",
           "RealTabR_D_Regressor", "Resnet_RTDL_D_Classifier", "Resnet_RTDL_D_Regressor", "Resnet_RTDL_HPO_Classifier",
           "Resnet_RTDL_HPO_Regressor", "TabR_HPO_Classifier", "TabR_HPO_Regressor", "TabR_S_D_Classifier",
           "TabR_S_D_Regressor",
           "TabM_D_Classifier", "TabM_D_Regressor", "TabM_HPO_Classifier", "TabM_HPO_Regressor",
           "XRFM_D_Classifier", "XRFM_D_Regressor", "XRFM_HPO_Classifier", "XRFM_HPO_Regressor",
           "XGB_D_Classifier", "XGB_D_Regressor", "XGB_HPO_Classifier", "XGB_HPO_Regressor", "XGB_HPO_TPE_Classifier",
           "XGB_HPO_TPE_Regressor", "XGB_PBB_D_Classifier", "XGB_TD_Classifier", "XGB_TD_Regressor"]


class RealMLPConstructorMixin:
    def __init__(self, device: Optional[str] = None, random_state: Optional[Union[int, np.random.RandomState]] = None,
                 n_cv: int = 1, n_refit: int = 0, n_repeats: int = 1, val_fraction: float = 0.2,
                 n_threads: Optional[int] = None,
                 tmp_folder: Optional[Union[str, pathlib.Path]] = None, verbosity: int = 0,
                 train_metric_name: Optional[str] = None, val_metric_name: Optional[str] = None,
                 n_epochs: Optional[int] = None,
                 batch_size: Optional[int] = None, predict_batch_size: Optional[int] = None,
                 hidden_sizes: Optional[Union[List[int], Literal['rectangular']]] = None,
                 n_hidden_layers: Optional[int] = None,
                 hidden_width: Optional[int] = None,
                 tfms: Optional[List[str]] = None,
                 num_emb_type: Optional[str] = None,
                 use_plr_embeddings: Optional[bool] = None, plr_sigma: Optional[float] = None,
                 plr_hidden_1: Optional[int] = None, plr_hidden_2: Optional[int] = None,
                 plr_act_name: Optional[str] = None, plr_use_densenet: Optional[bool] = None,
                 plr_use_cos_bias: Optional[bool] = None, plr_lr_factor: Optional[float] = None,
                 max_one_hot_cat_size: Optional[int] = None, embedding_size: Optional[int] = None,
                 act: Optional[str] = None,
                 use_parametric_act: Optional[bool] = None, act_lr_factor: Optional[float] = None,
                 weight_param: Optional[str] = None, weight_init_mode: Optional[str] = None,
                 weight_init_gain: Optional[str] = None,
                 weight_lr_factor: Optional[float] = None,
                 bias_init_mode: Optional[str] = None, bias_lr_factor: Optional[float] = None,
                 bias_wd_factor: Optional[float] = None,
                 add_front_scale: Optional[bool] = None,
                 scale_lr_factor: Optional[float] = None,
                 first_layer_lr_factor: Optional[float] = None,
                 block_str: Optional[str] = None,
                 first_layer_config: Optional[Dict[str, Any]] = None,
                 last_layer_config: Optional[Dict[str, Any]] = None,
                 middle_layer_config: Optional[Dict[str, Any]] = None,
                 p_drop: Optional[float] = None, p_drop_sched: Optional[str] = None,
                 wd: Optional[float] = None, wd_sched: Optional[str] = None,
                 opt: Optional[str] = None,
                 lr: Optional[Union[float, Dict[str, float]]] = None, lr_sched: Optional[str] = None,
                 mom: Optional[float] = None, mom_sched: Optional[str] = None,
                 sq_mom: Optional[float] = None, sq_mom_sched: Optional[str] = None,
                 opt_eps: Optional[float] = None, opt_eps_sched: Optional[str] = None,
                 normalize_output: Optional[bool] = None, clamp_output: Optional[bool] = None,
                 use_ls: Optional[bool] = None, ls_eps: Optional[float] = None, ls_eps_sched: Optional[str] = None,
                 use_early_stopping: Optional[bool] = None,
                 early_stopping_additive_patience: Optional[int] = None,
                 early_stopping_multiplicative_patience: Optional[float] = None,
                 calibration_method: Optional[str] = None,
                 sort_quantile_predictions: Optional[bool] = None,
                 stop_epoch: Optional[int] = None,
                 use_best_mean_epoch_for_cv: Optional[bool] = None,
                 n_ens: Optional[int] = None,
                 ens_av_before_softmax: Optional[int] = None,
                 ):
        """
        Constructor for RealMLP, using the default parameters from RealMLP-TD.
        For lists of default parameters, we refer to pytabkit.models.sklearn.default_params.DefaultParams.
        RealMLP-TD does automatic preprocessing,
        so no manual preprocessing is necessary except for imputing missing numerical values.

        Tips for modifications:

        * For faster training: For large datasets (say >50K samples), especially on GPUs, increase batch_size.
          It can also help to decrease n_epochs, set use_plr_embeddings=False (in case of many numerical features),
          increase max_one_hot_cat_size (in case of large-cardinality categories), or set use_parametric_act=False
        * For more accuracy: You can try increasing n_epochs or hidden_sizes while also decreasing lr.
        * For classification, if you care about metrics like cross-entropy or AUC instead of accuracy,
          we recommend setting val_metric_name='cross_entropy' and use_ls=False.

        :param device: PyTorch device name like 'cpu', 'cuda', 'cuda:0', 'mps' (default=None).
            If None, 'cuda' will be used if available, otherwise 'cpu'.
        :param random_state: Random state to use for random number generation
            (splitting, initialization, batch shuffling). If None, the behavior is not deterministic.
        :param n_cv: Number of cross-validation splits to use (default=1).
            If validation set indices or an explicit validation set are given in fit(),
            `n_cv` models will be fitted using different random seeds.
            Otherwise, `n_cv`-fold cross-validation will be used (stratified for classification).
            For n_cv=1, a single train-validation split will be used,
            where `val_fraction` controls the fraction of validation samples.
            If `n_refit=0` is set,
            the prediction will use the average of the models fitted during cross-validation.
            (Averaging is over probabilities for classification, and over outputs for regression.)
            Otherwise, refitted models will be used.
        :param n_refit: Number of models that should be refitted on the training+validation dataset (default=0).
            If zero, only the models from the cross-validation stage are used.
            If positive, `n_refit` models will be fitted on the training+validation dataset (all data given in fit())
            and their predictions will be averaged during predict().
        :param n_repeats: Number of times that the (cross-)validation split should be repeated (default=1).
            Values != 1 are only allowed when no custom validation split is provided.
            Larger number of repeats make things slower but reduce the potential for validation set overfitting,
            especially on smaller datasets.
        :param val_fraction: Fraction of samples used for validation (default=0.2). Has to be in [0, 1).
            Only used if `n_cv==1` and no validation split is provided in fit().
        :param n_threads: Number of threads that the method is allowed to use (default=number of physical cores).
        :param tmp_folder: Temporary folder in which data can be stored during fit().
            (Currently unused for RealMLP-TD and variants.) If None, methods generally try to not store intermediate data.
            Note that HPO and ensemble methods can use this to reduce RAM usage by storing fitted models,
            and will need this folder to be available whenever they are used.
        :param verbosity: Verbosity level (default=0, higher means more verbose).
            Set to 2 to see logs from intermediate epochs.
        :param train_metric_name: Name of the training metric
            (default='cross_entropy' for classification and 'mse' for regression).
            Currently most other metrics are not available for training.
        :param val_metric_name: Name of the validation metric (used for selecting the best epoch).
            Defaults are 'class_error' for classification and 'rmse' for regression.
            Main available classification metrics (all to be minimized): 'class_error', 'cross_entropy', '1-auc_ovo',
            '1-auc_ovr', '1-auc_mu', 'brier', '1-balanced_accuracy', '1-mcc', 'ece'.
            Main available regression metrics: 'rmse', 'mae', 'max_error',
            'pinball(0.95)' (also works with other quantiles specified directly in the string).
            For more metrics, we refer to `models.training.metrics.Metrics.apply()`.
        :param n_epochs: Number of epochs to train the model for (default=256)
        :param batch_size: Batch size to be used for fit(), default=256.
        :param predict_batch_size: Batch size to be used for predict(), default=1024.
        :param hidden_sizes: List of numbers of neurons for each hidden layer, default=[256, 256, 256].
            If this is set to 'rectangular', then [hidden_width] * n_hidden_layers will be used instead.
        :param n_hidden_layers: Number of hidden layers, default=3. Only used if hidden_sizes=='rectangular'.
        :param hidden_width: Width of each hidden layer, default=256. Only used if hidden_sizes=='rectangular'.
        :param tfms: List of preprocessing transformations,
            default=`['one_hot', 'median_center', 'robust_scale', 'smooth_clip', 'embedding']`.
            Other possible transformations include: 'median_center', 'l2_normalize', 'l1_normalize', 'quantile', 'kdi'.
        :param num_emb_type: Type of numerical embeddings used (default='pbld'). If not set to 'ignore',
            it overrides the parameters `use_plr_embeddings`, `plr_act_name`, `plr_use_densenet`, `plr_use_cos_bias`.
            Possible values: 'ignore', 'none' (no numerical embeddings), 'pl', 'plr', 'pbld', 'pblrd'.
        :param use_plr_embeddings: Whether PLR (or PL) numerical embeddings should be used (default=True).
        :param plr_sigma: Initialization standard deviation for first PLR embedding layer (default=0.1).
        :param plr_hidden_1: (Half of the) number of hidden neurons in the first PLR hidden layer (default=8).
            This number will be doubled since there are sin() and cos() versions for each hidden neuron.
        :param plr_hidden_2: Number of output neurons of the PLR hidden layer,
            excluding the optional densenet connection (default=7).
        :param plr_act_name: Name of PLR activation function (default='linear').
            Use 'relu' for the PLR version and 'linear' for the PL version.
        :param plr_use_densenet: Whether to append the original feature to the numerical embeddings (default=True).
        :param plr_use_cos_bias: Whether to use the cos(wx+b)
            version for the periodic embeddings instead of the (sin(wx), cos(wx)) version (default=True).
        :param plr_lr_factor: Learning rate factor for PLR embeddings (default=0.1).
            Gets multiplied with lr and with the value of the schedule.
        :param max_one_hot_cat_size: Maximum category size that one-hot encoding should be applied to,
            including the category for missing/unknown values (default=9).
        :param embedding_size: Number of output features of categorical embedding layers (default=8).
        :param act: Activation function (default='selu' for classification and 'mish' for regression).
            Can also be 'relu' or 'silu'.
        :param use_parametric_act: Whether to use a parametric activation as described in the paper (default=True).
        :param act_lr_factor: Learning rate factor for parametric activation (default=0.1).
        :param weight_param: Weight parametrization (default='ntk'). See models.nn.WeightFitter() for more options.
        :param weight_init_mode: Weight initialization mode (default='std').
            See models.nn.WeightFitter() for more options.
        :param weight_init_gain: Multiplier for the weight initialization standard deviation.
            (Does not apply to 'std' initialization mode.)
        :param weight_lr_factor: Learning rate factor for weights.
        :param bias_init_mode: Bias initialization mode (default='he+5'). See models.nn.BiasFitter() for more options.
        :param bias_lr_factor: Bias learning rate factor.
        :param bias_wd_factor: Bias weight decay factor.
        :param add_front_scale: Whether to add a scaling layer (diagonal weight matrix)
            before the linear layers (default=True). If set to true and a scaling layer is already configured
            in the block_str, this will create an additional scaling layer.
        :param scale_lr_factor: Scaling layer learning rate factor
            (default=1.0 but will be overridden by default for the first layer in first_layer_config).
        :param first_layer_lr_factor: First layer learning rate factor
            (default=1.0).
        :param block_str: String describing the default hidden layer components.
            The default is 'w-b-a-d' for weight, bias, activation, dropout.
            By default, the last layer config will override it with 'w-b'
            and the first layer config will override it with 's-w-b-a-d', where the 's' stands for the scaling layer.
        :param first_layer_config: Dictionary with more options
            that can override the other options for the construction of the first MLP layer specifically.
            The default is dict(block_str='s-w-b-a-d', scale_lr_factor=6.0),
            using a scaling layer at the beginning of the first layer with lr factor 6.0.
        :param last_layer_config: Dictionary with more options
            that can override the other options for the construction of the last MLP layer specifically.
            The default is an empty dict, in which case the block_str will still be overridden by 'w-b'.
        :param middle_layer_config: Dictionary with more options
            that can override the other options for the construction of the layers except first and last MLP layer.
            The default is an empty dict.
        :param p_drop: Dropout probability (default=0.15). Needs to be in [0, 1).
        :param p_drop_sched: Dropout schedule (default='flat_cos').
        :param wd: Weight decay implemented as in the PyTorch AdamW but works with all optimizers
            (default=0.0 for regression and 1e-2 for classification).
            Weight decay is implemented as
            param -= current_lr_value * current_wd_value * param
            where the current lr and wd values are determined using the base values (lr and wd),
            factors for the given parameter if available, and the respective schedule.
            Note that this is not identical to the original AdamW paper,
            where the lr base value is not included in the update equation.
        :param wd_sched: Weight decay schedule.
        :param opt: Optimizer (default='adam'). See optim.optimizers.get_opt_class().
        :param lr: Learning rate base value (default=0.04 for classification and 0.14 for regression).
        :param lr_sched: Learning rate schedule (default='coslog4'). See training.scheduling.get_schedule().
        :param mom: Momentum parameter, aka :math:`\\beta_1` for Adam (default=0.9).
        :param mom_sched: Momentum schedule (default='constant').
        :param sq_mom: Momentum of squared gradients, aka :math:`\\beta_2` for Adam (default=0.95).
        :param sq_mom_sched: Schedule for sq_mom (default='constant').
        :param opt_eps: Epsilon parameter of the optimizer (default=1e-8 for Adam).
        :param opt_eps_sched: Schedule for opt_eps (default='constant').
        :param normalize_output: Whether to standardize the target for regression (default=True for regression).
        :param clamp_output: Whether to clamp the output for predict() for regression
            to the min/max range seen during training (default=True for regression).
        :param use_ls: Whether to use label smoothing for classification (default=True for classification).
        :param ls_eps: Epsilon parameter for label smoothing (default=0.1 for classification)
        :param ls_eps_sched: Schedule for ls_eps (default='constant').
        :param use_early_stopping: Whether to use early stopping (default=False).
            Note that even without early stopping,
            the best epoch on the validation set is selected if there is a validation set.
            Training is stopped if the epoch exceeds
            early_stopping_multiplicative_patience * best_epoch + early_stopping_additive_patience.
        :param early_stopping_additive_patience: See use_early_stopping (default=20).
        :param early_stopping_multiplicative_patience: See use_early_stopping (default=2).
            We recommend to set it to 1 for monotone learning rate schedules
            but to keep it at 2 for the default schedule.
        :param calibration_method: Post-hoc calibration method (only for classification).
            We recommend 'ts-mix' for fast temperature scaling with Laplace smoothing.
            For other methods, see the get_calibrator method in https://github.com/dholzmueller/probmetrics.
        :param sort_quantile_predictions:
            If val_metric_name=='multi_pinball(...)', decides whether the predicted quantiles will be sorted
            to avoid quantile crossover. Default is True.
        :param stop_epoch:
            Epoch at which training should be stopped (for refitting).
            The total length of training used for the schedules will be determined by n_epochs,
            but the stopping epoch will be min(stop_epoch, n_epochs).
        :param use_best_mean_epoch_for_cv: If training an ensemble,
            whether they should all use a checkpoint from the same epoch with the best average loss,
            instead of using the best individual epochs (default=False).
        :param n_ens: Number of ensemble members that should be used per train-validation split (default=1).
            For best-epoch selection, the validation scores of averaged predictions will be used.
        :param ens_av_before_softmax: When using classifiction with n_ens>1, whether to average
            the ensemble predictions on each train-val split before taking the softmax (default=False). We recommend
            using False as it is representative of the averaging of models across train-val splits.
        """
        super().__init__()  # call the constructor of the other superclass for multiple inheritance
        self.device = device
        self.random_state = random_state
        self.n_cv = n_cv
        self.n_refit = n_refit
        self.n_repeats = n_repeats
        self.val_fraction = val_fraction
        self.n_threads = n_threads
        self.tmp_folder = tmp_folder
        self.verbosity = verbosity
        self.train_metric_name = train_metric_name
        self.val_metric_name = val_metric_name
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.predict_batch_size = predict_batch_size
        self.hidden_sizes = hidden_sizes
        self.n_hidden_layers = n_hidden_layers
        self.hidden_width = hidden_width
        self.tfms = tfms
        self.max_one_hot_cat_size = max_one_hot_cat_size
        self.embedding_size = embedding_size
        self.num_emb_type = num_emb_type
        self.use_plr_embeddings = use_plr_embeddings
        self.plr_sigma = plr_sigma
        self.plr_hidden_1 = plr_hidden_1
        self.plr_hidden_2 = plr_hidden_2
        self.plr_act_name = plr_act_name
        self.plr_use_densenet = plr_use_densenet
        self.plr_use_cos_bias = plr_use_cos_bias
        self.plr_lr_factor = plr_lr_factor
        self.act = act
        self.use_parametric_act = use_parametric_act
        self.act_lr_factor = act_lr_factor
        self.weight_param = weight_param
        self.weight_init_mode = weight_init_mode
        self.weight_init_gain = weight_init_gain
        self.weight_lr_factor = weight_lr_factor
        self.bias_init_mode = bias_init_mode
        self.bias_lr_factor = bias_lr_factor
        self.bias_wd_factor = bias_wd_factor
        self.add_front_scale = add_front_scale
        self.scale_lr_factor = scale_lr_factor
        self.first_layer_lr_factor = first_layer_lr_factor
        self.block_str = block_str
        self.first_layer_config = first_layer_config
        self.last_layer_config = last_layer_config
        self.middle_layer_config = middle_layer_config
        self.p_drop = p_drop
        self.p_drop_sched = p_drop_sched
        self.wd = wd
        self.wd_sched = wd_sched
        self.opt = opt
        self.lr = lr
        self.lr_sched = lr_sched
        self.mom = mom
        self.mom_sched = mom_sched
        self.sq_mom = sq_mom
        self.sq_mom_sched = sq_mom_sched
        self.opt_eps = opt_eps
        self.opt_eps_sched = opt_eps_sched
        self.normalize_output = normalize_output
        self.clamp_output = clamp_output
        self.use_ls = use_ls
        self.ls_eps = ls_eps
        self.ls_eps_sched = ls_eps_sched
        self.use_early_stopping = use_early_stopping
        self.early_stopping_additive_patience = early_stopping_additive_patience
        self.early_stopping_multiplicative_patience = early_stopping_multiplicative_patience
        self.calibration_method = calibration_method
        self.sort_quantile_predictions = sort_quantile_predictions
        self.stop_epoch = stop_epoch
        self.use_best_mean_epoch_for_cv = use_best_mean_epoch_for_cv
        self.n_ens = n_ens
        self.ens_av_before_softmax = ens_av_before_softmax


class RealMLP_TD_Classifier(RealMLPConstructorMixin, AlgInterfaceClassifier):
    """
    MLP-TD classifier. For constructor parameters, see `MLPConstructorMixin`.
    """

    def _get_default_params(self):
        return DefaultParams.RealMLP_TD_CLASS

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        return NNAlgInterface(**self.get_config())

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']


class RealMLP_TD_S_Classifier(RealMLPConstructorMixin, AlgInterfaceClassifier):
    """
    MLP-TD-S classifier. For constructor parameters, see `MLPConstructorMixin`.
    """

    def _get_default_params(self):
        return DefaultParams.RealMLP_TD_S_CLASS

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        return NNAlgInterface(**self.get_config())

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']


class RealMLP_TD_Regressor(RealMLPConstructorMixin, AlgInterfaceRegressor):
    """
    MLP-TD regressor. For constructor parameters, see `MLPConstructorMixin`.
    """

    def _get_default_params(self):
        return DefaultParams.RealMLP_TD_REG

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        return NNAlgInterface(**self.get_config())

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']


class RealMLP_TD_S_Regressor(RealMLPConstructorMixin, AlgInterfaceRegressor):
    """
    MLP-TD-S regressor. For constructor parameters, see `MLPConstructorMixin`.
    """

    def _get_default_params(self):
        return DefaultParams.RealMLP_TD_S_REG

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        return NNAlgInterface(**self.get_config())

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']


# --------------------------------- GBDTs -----------------------------------


class LGBMConstructorMixin:
    def __init__(self, device: Optional[str] = None, random_state: Optional[Union[int, np.random.RandomState]] = None,
                 n_cv: int = 1, n_refit: int = 0, n_repeats: int = 1, val_fraction: float = 0.2,
                 n_threads: Optional[int] = None,
                 tmp_folder: Optional[Union[str, pathlib.Path]] = None, verbosity: int = 0,
                 n_estimators: Optional[int] = None,
                 max_depth: Optional[int] = None,
                 num_leaves: Optional[int] = None,
                 lr: Optional[float] = None,
                 subsample: Optional[float] = None,
                 colsample_bytree: Optional[float] = None,
                 bagging_freq: Optional[float] = None,
                 min_data_in_leaf: Optional[int] = None,
                 min_sum_hessian_in_leaf: Optional[int] = None,
                 lambda_l1: Optional[float] = None,
                 lambda_l2: Optional[float] = None,
                 boosting: Optional[str] = None,
                 max_bin: Optional[int] = None,
                 cat_smooth: Optional[float] = None,
                 cat_l2: Optional[float] = None,
                 val_metric_name: Optional[str] = None,
                 calibration_method: Optional[str] = None,
                 ):
        self.device = device
        self.random_state = random_state
        self.n_cv = n_cv
        self.n_refit = n_refit
        self.n_repeats = n_repeats
        self.val_fraction = val_fraction
        self.n_threads = n_threads
        self.tmp_folder = tmp_folder
        self.verbosity = verbosity
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.lr = lr
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.bagging_freq = bagging_freq
        self.min_data_in_leaf = min_data_in_leaf
        self.min_sum_hessian_in_leaf = min_sum_hessian_in_leaf
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.boosting = boosting
        self.max_bin = max_bin
        self.cat_smooth = cat_smooth
        self.cat_l2 = cat_l2
        self.val_metric_name = val_metric_name
        self.calibration_method = calibration_method


class LGBM_TD_Classifier(LGBMConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self):
        return DefaultParams.LGBM_TD_CLASS

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.lightgbm_interfaces import LGBMSubSplitInterface
        return SingleSplitWrapperAlgInterface([LGBMSubSplitInterface(**self.get_config()) for i in range(n_cv)])


class LGBM_D_Classifier(LGBMConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self):
        return DefaultParams.LGBM_D

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.lightgbm_interfaces import LGBMSubSplitInterface
        return SingleSplitWrapperAlgInterface([LGBMSubSplitInterface(**self.get_config()) for i in range(n_cv)])


class LGBM_TD_Regressor(LGBMConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self):
        return DefaultParams.LGBM_TD_REG

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.lightgbm_interfaces import LGBMSubSplitInterface
        return SingleSplitWrapperAlgInterface([LGBMSubSplitInterface(**self.get_config()) for i in range(n_cv)])

    def _supports_multioutput(self) -> bool:
        return False


class LGBM_D_Regressor(LGBMConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self):
        return DefaultParams.LGBM_D

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.lightgbm_interfaces import LGBMSubSplitInterface
        return SingleSplitWrapperAlgInterface([LGBMSubSplitInterface(**self.get_config()) for i in range(n_cv)])

    def _supports_multioutput(self) -> bool:
        return False


class XGBConstructorMixin:
    def __init__(self, device: Optional[str] = None, random_state: Optional[Union[int, np.random.RandomState]] = None,
                 n_cv: int = 1, n_refit: int = 0, n_repeats: int = 1, val_fraction: float = 0.2,
                 n_threads: Optional[int] = None,
                 tmp_folder: Optional[Union[str, pathlib.Path]] = None, verbosity: int = 0,
                 train_metric_name: Optional[str] = None, val_metric_name: Optional[str] = None,
                 n_estimators: Optional[int] = None,
                 max_depth: Optional[int] = None,
                 lr: Optional[float] = None,
                 subsample: Optional[float] = None,
                 colsample_bytree: Optional[float] = None,
                 colsample_bylevel: Optional[float] = None,
                 colsample_bynode: Optional[float] = None,
                 min_child_weight: Optional[float] = None,
                 alpha: Optional[float] = None,
                 reg_lambda: Optional[float] = None,
                 gamma: Optional[float] = None,
                 tree_method: Optional[str] = None,
                 max_delta_step: Optional[float] = None,
                 max_cat_to_onehot: Optional[int] = None,
                 num_parallel_tree: Optional[int] = None,
                 max_bin: Optional[int] = None,
                 multi_strategy: Optional[str] = None,
                 calibration_method: Optional[str] = None,
                 ):
        self.device = device
        self.random_state = random_state
        self.n_cv = n_cv
        self.n_refit = n_refit
        self.n_repeats = n_repeats
        self.val_fraction = val_fraction
        self.n_threads = n_threads
        self.tmp_folder = tmp_folder
        self.verbosity = verbosity
        self.train_metric_name = train_metric_name
        self.val_metric_name = val_metric_name
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lr = lr
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.min_child_weight = min_child_weight
        self.alpha = alpha
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.tree_method = tree_method
        self.max_delta_step = max_delta_step
        self.max_cat_to_onehot = max_cat_to_onehot
        self.num_parallel_tree = num_parallel_tree
        self.max_bin = max_bin
        self.multi_strategy = multi_strategy
        self.calibration_method = calibration_method


class XGB_TD_Classifier(XGBConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self):
        return DefaultParams.XGB_TD_CLASS

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.xgboost_interfaces import XGBSubSplitInterface
        return SingleSplitWrapperAlgInterface([XGBSubSplitInterface(**self.get_config()) for i in range(n_cv)])

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda']


class XGB_D_Classifier(XGBConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self):
        return DefaultParams.XGB_D

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.xgboost_interfaces import XGBSubSplitInterface
        return SingleSplitWrapperAlgInterface([XGBSubSplitInterface(**self.get_config()) for i in range(n_cv)])

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda']


class XGB_PBB_D_Classifier(XGBConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self):
        return DefaultParams.XGB_PBB_CLASS

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.xgboost_interfaces import XGBSubSplitInterface
        return SingleSplitWrapperAlgInterface([XGBSubSplitInterface(**self.get_config()) for i in range(n_cv)])

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda']


class XGB_TD_Regressor(XGBConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self):
        return DefaultParams.XGB_TD_REG

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.xgboost_interfaces import XGBSubSplitInterface
        return SingleSplitWrapperAlgInterface([XGBSubSplitInterface(**self.get_config()) for i in range(n_cv)])

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda']

    def _supports_multioutput(self) -> bool:
        return False


class XGB_D_Regressor(XGBConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self):
        return DefaultParams.XGB_D

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.xgboost_interfaces import XGBSubSplitInterface
        return SingleSplitWrapperAlgInterface([XGBSubSplitInterface(**self.get_config()) for i in range(n_cv)])

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda']

    def _supports_multioutput(self) -> bool:
        return False


class CatBoostConstructorMixin:
    def __init__(self, device: Optional[str] = None, random_state: Optional[Union[int, np.random.RandomState]] = None,
                 n_cv: int = 1, n_refit: int = 0, n_repeats: int = 1, val_fraction: float = 0.2,
                 n_threads: Optional[int] = None,
                 tmp_folder: Optional[Union[str, pathlib.Path]] = None, verbosity: int = 0,
                 n_estimators: Optional[int] = None,
                 max_depth: Optional[int] = None,
                 lr: Optional[float] = None,
                 subsample: Optional[float] = None,
                 colsample_bylevel: Optional[float] = None,
                 random_strength: Optional[float] = None,
                 bagging_temperature: Optional[float] = None,
                 leaf_estimation_iterations: Optional[int] = None,
                 bootstrap_type: Optional[str] = None,
                 boosting_type: Optional[str] = None,
                 min_data_in_leaf: Optional[int] = None,
                 grow_policy: Optional[str] = None,
                 num_leaves: Optional[int] = None,
                 max_bin: Optional[int] = None,
                 # renamed from border_count since it is named max_bin in the default parameters
                 l2_leaf_reg: Optional[float] = None,
                 one_hot_max_size: Optional[int] = None,
                 val_metric_name: Optional[str] = None,
                 train_metric_name: Optional[str] = None,
                 calibration_method: Optional[str] = None,
                 ):
        self.device = device
        self.random_state = random_state
        self.n_cv = n_cv
        self.n_refit = n_refit
        self.n_repeats = n_repeats
        self.val_fraction = val_fraction
        self.n_threads = n_threads
        self.tmp_folder = tmp_folder
        self.verbosity = verbosity
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lr = lr
        self.subsample = subsample
        self.colsample_bylevel = colsample_bylevel
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
        self.leaf_estimation_iterations = leaf_estimation_iterations
        self.bootstrap_type = bootstrap_type
        self.boosting_type = boosting_type
        self.min_data_in_leaf = min_data_in_leaf
        self.grow_policy = grow_policy
        self.num_leaves = num_leaves
        self.max_bin = max_bin
        self.l2_leaf_reg = l2_leaf_reg
        self.one_hot_max_size = one_hot_max_size
        self.val_metric_name = val_metric_name
        self.train_metric_name = train_metric_name
        self.calibration_method = calibration_method


class CatBoost_TD_Classifier(CatBoostConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self):
        return DefaultParams.CB_TD_CLASS

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.catboost_interfaces import CatBoostSubSplitInterface
        return SingleSplitWrapperAlgInterface([CatBoostSubSplitInterface(**self.get_config()) for i in range(n_cv)])

    def _supports_single_class(self) -> bool:
        return False

    def _supports_single_sample(self) -> bool:
        return False

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda']


class CatBoost_D_Classifier(CatBoostConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self):
        return DefaultParams.CB_D

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.catboost_interfaces import CatBoostSubSplitInterface
        return SingleSplitWrapperAlgInterface([CatBoostSubSplitInterface(**self.get_config()) for i in range(n_cv)])

    def _supports_single_class(self) -> bool:
        return False

    def _supports_single_sample(self) -> bool:
        return False

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda']


class CatBoost_TD_Regressor(CatBoostConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self):
        return DefaultParams.CB_TD_REG

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.catboost_interfaces import CatBoostSubSplitInterface
        return SingleSplitWrapperAlgInterface([CatBoostSubSplitInterface(**self.get_config()) for i in range(n_cv)])

    def _supports_multioutput(self) -> bool:
        return False

    def _supports_single_sample(self) -> bool:
        return False

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda']


class CatBoost_D_Regressor(CatBoostConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self):
        return DefaultParams.CB_D

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.catboost_interfaces import CatBoostSubSplitInterface
        return SingleSplitWrapperAlgInterface([CatBoostSubSplitInterface(**self.get_config()) for i in range(n_cv)])

    def _supports_multioutput(self) -> bool:
        return False

    def _supports_single_sample(self) -> bool:
        return False

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda']


class RFConstructorMixin:
    def __init__(self, device: Optional[str] = None, random_state: Optional[Union[int, np.random.RandomState]] = None,
                 n_cv: int = 1, n_refit: int = 0, n_repeats: int = 1, val_fraction: float = 0.2,
                 n_threads: Optional[int] = None,
                 tmp_folder: Optional[Union[str, pathlib.Path]] = None, verbosity: int = 0,
                 n_estimators: Optional[int] = None,
                 calibration_method: Optional[str] = None,
                 ):
        """
        Validation set is not used.
        :param device:
        :param random_state:
        :param n_cv:
        :param n_refit:
        :param n_repeats:
        :param val_fraction:
        :param n_threads:
        :param tmp_folder:
        :param verbosity:
        :param n_estimators:
        :param calibration_method: Post-hoc calibration method (only for classification).
            We recommend 'ts-mix' for fast temperature scaling with Laplace smoothing.
            For other methods, see the get_calibrator method in https://github.com/dholzmueller/probmetrics.
        """
        self.device = device
        self.random_state = random_state
        self.n_cv = n_cv
        self.n_refit = n_refit
        self.n_repeats = n_repeats
        self.val_fraction = val_fraction
        self.n_threads = n_threads
        self.tmp_folder = tmp_folder
        self.verbosity = verbosity
        self.n_estimators = n_estimators
        self.calibration_method = calibration_method


class RF_SKL_D_Classifier(RFConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self):
        return DefaultParams.RF_SKL_D

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.other_interfaces import RFSubSplitInterface
        return SingleSplitWrapperAlgInterface([RFSubSplitInterface(**self.get_config()) for i in range(n_cv)])


class RF_SKL_D_Regressor(RFConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self):
        return DefaultParams.RF_SKL_D

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.other_interfaces import RFSubSplitInterface
        return SingleSplitWrapperAlgInterface([RFSubSplitInterface(**self.get_config()) for i in range(n_cv)])


class MLPSKLConstructorMixin:
    def __init__(self, device: Optional[str] = None, random_state: Optional[Union[int, np.random.RandomState]] = None,
                 n_cv: int = 1, n_refit: int = 0, n_repeats: int = 1, val_fraction: float = 0.2,
                 n_threads: Optional[int] = None,
                 tmp_folder: Optional[Union[str, pathlib.Path]] = None, verbosity: int = 0,
                 calibration_method: Optional[str] = None,
                 ):
        self.device = device
        self.random_state = random_state
        self.n_cv = n_cv
        self.n_refit = n_refit
        self.n_repeats = n_repeats
        self.val_fraction = val_fraction
        self.n_threads = n_threads
        self.tmp_folder = tmp_folder
        self.verbosity = verbosity
        self.calibration_method = calibration_method


class MLP_SKL_D_Classifier(MLPSKLConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self):
        return DefaultParams.MLP_SKL_D

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.other_interfaces import SklearnMLPSubSplitInterface
        return SingleSplitWrapperAlgInterface([SklearnMLPSubSplitInterface(**self.get_config()) for i in range(n_cv)])


class MLP_SKL_D_Regressor(MLPSKLConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self):
        return DefaultParams.MLP_SKL_D

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.other_interfaces import SklearnMLPSubSplitInterface
        return SingleSplitWrapperAlgInterface([SklearnMLPSubSplitInterface(**self.get_config()) for i in range(n_cv)])


# HPO methods

class GBDTHPOConstructorMixin:
    def __init__(self, device: Optional[str] = None, random_state: Optional[Union[int, np.random.RandomState]] = None,
                 n_cv: int = 1, n_refit: int = 0, n_repeats: int = 1, val_fraction: float = 0.2,
                 n_threads: Optional[int] = None,
                 tmp_folder: Optional[Union[str, pathlib.Path]] = None, verbosity: int = 0,
                 n_estimators: Optional[int] = None,
                 hpo_space_name: Optional[str] = None,
                 n_hyperopt_steps: Optional[int] = None,
                 calibration_method: Optional[str] = None,
                 use_caruana_ensembling: Optional[bool] = None,
                 time_limit_s: Optional[float] = None,
                 ):
        self.device = device
        self.random_state = random_state
        self.n_cv = n_cv
        self.n_refit = n_refit
        self.n_repeats = n_repeats
        self.val_fraction = val_fraction
        self.n_threads = n_threads
        self.tmp_folder = tmp_folder
        self.verbosity = verbosity
        self.n_estimators = n_estimators
        self.hpo_space_name = hpo_space_name
        self.n_hyperopt_steps = n_hyperopt_steps
        self.calibration_method = calibration_method
        self.use_caruana_ensembling = use_caruana_ensembling
        self.time_limit_s = time_limit_s


class XGB_HPO_Classifier(GBDTHPOConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self) -> Dict[str, Any]:
        return dict(n_hyperopt_steps=50)

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.xgboost_interfaces import RandomParamsXGBAlgInterface
        config = self.get_config()
        n_hyperopt_steps = config['n_hyperopt_steps']
        interface_type = CaruanaEnsembleAlgInterface if config.get('use_caruana_ensembling',
                                                                   False) else AlgorithmSelectionAlgInterface
        return interface_type(
            [SingleSplitWrapperAlgInterface([RandomParamsXGBAlgInterface(model_idx=i, **config) for j in range(n_cv)])
             for i in range(n_hyperopt_steps)], **config)

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda']


class XGB_HPO_TPE_Classifier(GBDTHPOConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self) -> Dict[str, Any]:
        return dict(n_estimators=1000, n_hyperopt_steps=50,
                    early_stopping_rounds=300,
                    tree_method='hist', hpo_space_name='grinsztajn')

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.xgboost_interfaces import XGBHyperoptAlgInterface
        return XGBHyperoptAlgInterface(**self.get_config())

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda']


class XGB_HPO_Regressor(GBDTHPOConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self) -> Dict[str, Any]:
        return dict(n_hyperopt_steps=50)

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda']

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.xgboost_interfaces import RandomParamsXGBAlgInterface
        config = self.get_config()
        n_hyperopt_steps = config['n_hyperopt_steps']
        interface_type = CaruanaEnsembleAlgInterface if config.get('use_caruana_ensembling',
                                                                   False) else AlgorithmSelectionAlgInterface
        return interface_type(
            [SingleSplitWrapperAlgInterface([RandomParamsXGBAlgInterface(model_idx=i, **config) for j in range(n_cv)])
             for i in range(n_hyperopt_steps)], **config)

    def _supports_multioutput(self) -> bool:
        return False


class XGB_HPO_TPE_Regressor(GBDTHPOConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self) -> Dict[str, Any]:
        return dict(n_estimators=1000, n_hyperopt_steps=50,
                    early_stopping_rounds=300,
                    tree_method='hist', hpo_space_name='grinsztajn')

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda']

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.xgboost_interfaces import XGBHyperoptAlgInterface
        return XGBHyperoptAlgInterface(**self.get_config())

    def _supports_multioutput(self) -> bool:
        return False


class LGBM_HPO_Classifier(GBDTHPOConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self) -> Dict[str, Any]:
        return dict(n_hyperopt_steps=50)

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.lightgbm_interfaces import RandomParamsLGBMAlgInterface
        config = self.get_config()
        n_hyperopt_steps = config['n_hyperopt_steps']
        interface_type = CaruanaEnsembleAlgInterface if config.get('use_caruana_ensembling',
                                                                   False) else AlgorithmSelectionAlgInterface
        return interface_type(
            [SingleSplitWrapperAlgInterface([RandomParamsLGBMAlgInterface(model_idx=i, **config) for j in range(n_cv)])
             for i in range(n_hyperopt_steps)], **config)


class LGBM_HPO_TPE_Classifier(GBDTHPOConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self) -> Dict[str, Any]:
        return dict(n_estimators=1000, n_hyperopt_steps=50,
                    early_stopping_rounds=300,
                    hpo_space_name='catboost_quality_benchmarks')

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.lightgbm_interfaces import LGBMHyperoptAlgInterface
        return LGBMHyperoptAlgInterface(**self.get_config())


class LGBM_HPO_Regressor(GBDTHPOConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self) -> Dict[str, Any]:
        return dict(n_hyperopt_steps=50)

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.lightgbm_interfaces import RandomParamsLGBMAlgInterface
        config = self.get_config()
        n_hyperopt_steps = config['n_hyperopt_steps']
        interface_type = CaruanaEnsembleAlgInterface if config.get('use_caruana_ensembling',
                                                                   False) else AlgorithmSelectionAlgInterface
        return interface_type(
            [SingleSplitWrapperAlgInterface([RandomParamsLGBMAlgInterface(model_idx=i, **config) for j in range(n_cv)])
             for i in range(n_hyperopt_steps)], **config)

    def _supports_multioutput(self) -> bool:
        return False


class LGBM_HPO_TPE_Regressor(GBDTHPOConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self) -> Dict[str, Any]:
        return dict(n_estimators=1000, n_hyperopt_steps=50,
                    early_stopping_rounds=300,
                    hpo_space_name='catboost_quality_benchmarks')

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.lightgbm_interfaces import LGBMHyperoptAlgInterface
        return LGBMHyperoptAlgInterface(**self.get_config())

    def _supports_multioutput(self) -> bool:
        return False


class CatBoost_HPO_Classifier(GBDTHPOConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self) -> Dict[str, Any]:
        return dict(n_hyperopt_steps=50)

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.catboost_interfaces import RandomParamsCatBoostAlgInterface
        config = self.get_config()
        n_hyperopt_steps = config['n_hyperopt_steps']
        interface_type = CaruanaEnsembleAlgInterface if config.get('use_caruana_ensembling',
                                                                   False) else AlgorithmSelectionAlgInterface
        return interface_type(
            [SingleSplitWrapperAlgInterface(
                [RandomParamsCatBoostAlgInterface(model_idx=i, **config) for j in range(n_cv)])
                for i in range(n_hyperopt_steps)], **config)

    def _supports_single_class(self) -> bool:
        return False

    def _supports_single_sample(self) -> bool:
        return False

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda']


class CatBoost_HPO_TPE_Classifier(GBDTHPOConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self) -> Dict[str, Any]:
        return dict(n_estimators=1000, n_hyperopt_steps=50,
                    early_stopping_rounds=300,
                    hpo_space_name='shwartz-ziv')

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.catboost_interfaces import CatBoostHyperoptAlgInterface
        return CatBoostHyperoptAlgInterface(**self.get_config())

    def _supports_single_class(self) -> bool:
        return False

    def _supports_single_sample(self) -> bool:
        return False

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda']


class CatBoost_HPO_Regressor(GBDTHPOConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self) -> Dict[str, Any]:
        return dict(n_hyperopt_steps=50)

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.catboost_interfaces import RandomParamsCatBoostAlgInterface
        config = self.get_config()
        n_hyperopt_steps = config['n_hyperopt_steps']
        interface_type = CaruanaEnsembleAlgInterface if config.get('use_caruana_ensembling',
                                                                   False) else AlgorithmSelectionAlgInterface
        return interface_type(
            [SingleSplitWrapperAlgInterface(
                [RandomParamsCatBoostAlgInterface(model_idx=i, **config) for j in range(n_cv)])
                for i in range(n_hyperopt_steps)], **config)

    def _supports_multioutput(self) -> bool:
        return False

    def _supports_single_sample(self) -> bool:
        return False

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda']


class CatBoost_HPO_TPE_Regressor(GBDTHPOConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self) -> Dict[str, Any]:
        return dict(n_estimators=1000, n_hyperopt_steps=50,
                    early_stopping_rounds=300,
                    hpo_space_name='shwartz-ziv')

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.catboost_interfaces import CatBoostHyperoptAlgInterface
        return CatBoostHyperoptAlgInterface(**self.get_config())

    def _supports_multioutput(self) -> bool:
        return False

    def _supports_single_sample(self) -> bool:
        return False

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda']


class RF_HPO_Classifier(GBDTHPOConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self) -> Dict[str, Any]:
        return dict(n_hyperopt_steps=50)

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.other_interfaces import RandomParamsRFAlgInterface
        config = self.get_config()
        n_hyperopt_steps = config['n_hyperopt_steps']
        interface_type = CaruanaEnsembleAlgInterface if config.get('use_caruana_ensembling',
                                                                   False) else AlgorithmSelectionAlgInterface
        return interface_type(
            [SingleSplitWrapperAlgInterface(
                [RandomParamsRFAlgInterface(model_idx=i, **config) for j in range(n_cv)])
                for i in range(n_hyperopt_steps)], **config)

    def _supports_single_class(self) -> bool:
        return False

    def _supports_single_sample(self) -> bool:
        return False


class RF_HPO_Regressor(GBDTHPOConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self) -> Dict[str, Any]:
        return dict(n_hyperopt_steps=50)

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.other_interfaces import RandomParamsRFAlgInterface
        config = self.get_config()
        n_hyperopt_steps = config['n_hyperopt_steps']
        interface_type = CaruanaEnsembleAlgInterface if config.get('use_caruana_ensembling',
                                                                   False) else AlgorithmSelectionAlgInterface
        return interface_type(
            [SingleSplitWrapperAlgInterface(
                [RandomParamsRFAlgInterface(model_idx=i, **config) for j in range(n_cv)])
                for i in range(n_hyperopt_steps)], **config)

    def _supports_multioutput(self) -> bool:
        return False

    def _supports_single_sample(self) -> bool:
        return False


class RealMLPHPOConstructorMixin:
    def __init__(self, device: Optional[str] = None, random_state: Optional[Union[int, np.random.RandomState]] = None,
                 n_cv: int = 1, n_refit: int = 0, n_repeats: int = 1, val_fraction: float = 0.2,
                 n_threads: Optional[int] = None,
                 tmp_folder: Optional[Union[str, pathlib.Path]] = None, verbosity: int = 0,
                 n_hyperopt_steps: Optional[int] = None, val_metric_name: Optional[str] = None,
                 calibration_method: Optional[str] = None, hpo_space_name: Optional[str] = None,
                 n_caruana_steps: Optional[int] = None, n_epochs: Optional[int] = None,
                 use_caruana_ensembling: Optional[bool] = None, train_metric_name: Optional[str] = None,
                 time_limit_s: Optional[float] = None,
                 ):
        """

        :param device: PyTorch device name like 'cpu', 'cuda', 'cuda:0', 'mps' (default=None).
            If None, 'cuda' will be used if available, otherwise 'cpu'.
        :param random_state: Random state to use for random number generation
            (splitting, initialization, batch shuffling). If None, the behavior is not deterministic.
        :param n_cv: Number of cross-validation splits to use (default=1).
            If validation set indices or an explicit validation set are given in fit(),
            `n_cv` models will be fitted using different random seeds.
            Otherwise, `n_cv`-fold cross-validation will be used (stratified for classification).
            For n_cv=1, a single train-validation split will be used,
            where `val_fraction` controls the fraction of validation samples.
            If `n_refit=0` is set,
            the prediction will use the average of the models fitted during cross-validation.
            (Averaging is over probabilities for classification, and over outputs for regression.)
            Otherwise, refitted models will be used.
        :param n_refit: Number of models that should be refitted on the training+validation dataset (default=0).
            If zero, only the models from the cross-validation stage are used.
            If positive, `n_refit` models will be fitted on the training+validation dataset (all data given in fit())
            and their predictions will be averaged during predict().
        :param n_repeats: Number of times that the (cross-)validation split should be repeated (default=1).
            Values != 1 are only allowed when no custom validation split is provided.
            Larger number of repeats make things slower but reduce the potential for validation set overfitting,
            especially on smaller datasets.
        :param val_fraction: Fraction of samples used for validation (default=0.2). Has to be in [0, 1).
            Only used if `n_cv==1` and no validation split is provided in fit().
        :param n_threads: Number of threads that the method is allowed to use (default=number of physical cores).
        :param tmp_folder: Folder in which models can be stored.
            Setting this allows reducing RAM/VRAM usage by not having all models in RAM at the same time.
            In this case, the folder needs to be preserved as long as the model exists
            (including when the model is pickled to disk).
        :param verbosity: Verbosity level (default=0, higher means more verbose).
            Set to 2 to see logs from intermediate epochs.
        :param n_hyperopt_steps: Number of random hyperparameter configs
            that should be used to train models (default=50).
        :param val_metric_name: Name of the validation metric (used for selecting the best epoch).
            Not used for all models but at least for RealMLP and probably TabM.
            Defaults are 'class_error' for classification and 'rmse' for regression.
            Main available classification metrics (all to be minimized): 'class_error', 'cross_entropy', '1-auc_ovo',
            '1-auc_ovr', '1-auc_mu', 'brier', '1-balanced_accuracy', '1-mcc', 'ece'.
            Main available regression metrics: 'rmse', 'mae', 'max_error',
            'pinball(0.95)' (also works with other quantiles specified directly in the string).
            For more metrics, we refer to `models.training.metrics.Metrics.apply()`.
        :param calibration_method: Post-hoc calibration method (only for classification) (default=None).
            We recommend 'ts-mix' for fast temperature scaling with Laplace smoothing.
            For other methods, see the get_calibrator method in https://github.com/dholzmueller/probmetrics.
        :param hpo_space_name: Name of the HPO space (default='default').
            The search space used in the paper for RealMLP is 'default'. However, we recommend using 'tabarena' for the best results.
        :param n_caruana_steps: Number of weight update iterations for Caruana et al. weighted ensembling (default=40).
            This parameter is only used when use_caruana_ensembling=True.
        :param n_epochs: Number of epochs to train for each NN (default=None).
            If set, it will override the values from the search space. (Might be ignored for non-RealMLP methods.)
        :param use_caruana_ensembling: Whether to use the algorithm by Caruana et al. (2004)
            to select a weighted ensemble of models instead of only selecting the best model (default=False).
        :param train_metric_name: Name of the training metric
            (default is cross_entropy for classification and mse for regression).
            For regression, pinball/multi_pinball can be used instead. (Might be ignored for non-RealMLP methods.)
        :param time_limit_s: Time limit in seconds (default=None).
        """
        self.device = device
        self.random_state = random_state
        self.n_cv = n_cv
        self.n_refit = n_refit
        self.n_repeats = n_repeats
        self.val_fraction = val_fraction
        self.n_threads = n_threads
        self.tmp_folder = tmp_folder
        self.verbosity = verbosity
        self.n_hyperopt_steps = n_hyperopt_steps
        self.val_metric_name = val_metric_name
        self.calibration_method = calibration_method
        self.hpo_space_name = hpo_space_name
        self.n_caruana_steps = n_caruana_steps
        self.n_epochs = n_epochs
        self.use_caruana_ensembling = use_caruana_ensembling
        self.train_metric_name = train_metric_name
        self.time_limit_s = time_limit_s


class RealMLP_HPO_Classifier(RealMLPHPOConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self):
        return dict(n_hyperopt_steps=50)

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        config = self.get_config()
        n_hyperopt_steps = config['n_hyperopt_steps']
        interface_type = CaruanaEnsembleAlgInterface if config.get('use_caruana_ensembling',
                                                                   False) else AlgorithmSelectionAlgInterface
        return interface_type([RandomParamsNNAlgInterface(model_idx=i, **config)
                               for i in range(n_hyperopt_steps)], **config)

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']


class RealMLP_HPO_Regressor(RealMLPHPOConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self):
        return dict(n_hyperopt_steps=50)

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        config = self.get_config()
        n_hyperopt_steps = config['n_hyperopt_steps']
        interface_type = CaruanaEnsembleAlgInterface if config.get('use_caruana_ensembling',
                                                                   False) else AlgorithmSelectionAlgInterface
        return interface_type([RandomParamsNNAlgInterface(model_idx=i, **config)
                               for i in range(n_hyperopt_steps)], **config)

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']


class ResnetConstructorMixin:
    def __init__(self,
                 module_d_embedding: Optional[int] = None,
                 module_d: Optional[int] = None,
                 module_d_hidden_factor: Optional[float] = None,
                 module_n_layers: Optional[int] = None,
                 module_activation: Optional[str] = None,
                 module_normalization: Optional[str] = None,
                 module_hidden_dropout: Optional[float] = None,
                 module_residual_dropout: Optional[float] = None,

                 verbose: Optional[int] = None,
                 max_epochs: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 optimizer: Optional[str] = None,
                 es_patience: Optional[int] = None,
                 lr: Optional[float] = None,
                 lr_scheduler: Optional[bool] = None,
                 lr_patience: Optional[int] = None,
                 optimizer_weight_decay: Optional[float] = None,
                 use_checkpoints: Optional[bool] = None,
                 transformed_target: Optional[bool] = None,
                 tfms: Optional[List[str]] = None,
                 quantile_output_distribution: Optional[str] = None,
                 val_metric_name: Optional[str] = None,
                 device: Optional[str] = None, random_state: Optional[Union[int, np.random.RandomState]] = None,
                 n_cv: int = 1, n_refit: int = 0, n_repeats: int = 1, val_fraction: float = 0.2,
                 n_threads: Optional[int] = None,
                 tmp_folder: Optional[Union[str, pathlib.Path]] = None, verbosity: int = 0,
                 calibration_method: Optional[str] = None,
                 ):
        self.module_d_embedding = module_d_embedding
        self.module_d = module_d
        self.module_d_hidden_factor = module_d_hidden_factor
        self.module_n_layers = module_n_layers
        self.module_activation = module_activation
        self.module_normalization = module_normalization
        self.module_hidden_dropout = module_hidden_dropout
        self.module_residual_dropout = module_residual_dropout
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.es_patience = es_patience
        self.lr_scheduler = lr_scheduler
        self.lr_patience = lr_patience
        self.lr = lr
        self.optimizer_weight_decay = optimizer_weight_decay
        self.use_checkpoints = use_checkpoints
        self.transformed_target = transformed_target
        self.tfms = tfms
        self.quantile_output_distribution = quantile_output_distribution
        self.val_metric_name = val_metric_name
        self.device = device
        self.random_state = random_state
        self.n_cv = n_cv
        self.n_refit = n_refit
        self.n_repeats = n_repeats
        self.val_fraction = val_fraction
        self.n_threads = n_threads
        self.tmp_folder = tmp_folder
        self.verbosity = verbosity
        self.calibration_method = calibration_method


class Resnet_RTDL_D_Classifier(ResnetConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self):
        return DefaultParams.RESNET_RTDL_D_CLASS_TabZilla

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.rtdl_interfaces import ResnetSubSplitInterface
        return SingleSplitWrapperAlgInterface([ResnetSubSplitInterface(**self.get_config()) for i in range(n_cv)])

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']

    def _supports_single_class(self) -> bool:
        return False

    def _supports_single_sample(self) -> bool:
        return False

    def _non_deterministic_tag(self) -> bool:
        # set non-deterministic
        # since this class can otherwise fail the check_methods_subset_invariance test due to low precision (?)
        # only on windows, only recently?? probably a skorch problem?
        return True


class Resnet_RTDL_D_Regressor(ResnetConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self):
        return DefaultParams.RESNET_RTDL_D_REG_TabZilla

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.rtdl_interfaces import ResnetSubSplitInterface
        return SingleSplitWrapperAlgInterface([ResnetSubSplitInterface(**self.get_config()) for i in range(n_cv)])

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']

    def _supports_single_sample(self) -> bool:
        return False

    def _supports_multioutput(self) -> bool:
        return False

    def _non_deterministic_tag(self) -> bool:
        # set non-deterministic
        # since this class can otherwise fail the check_methods_subset_invariance test due to low precision (?)
        return True


class FTTransformerConstructorMixin:
    def __init__(self,
                 module_d_token: Optional[int] = None,
                 module_d_ffn_factor: Optional[float] = None,
                 module_n_layers: Optional[int] = None,
                 module_n_heads: Optional[int] = None,
                 module_token_bias: Optional[bool] = None,
                 module_attention_dropout: Optional[float] = None,
                 module_ffn_dropout: Optional[float] = None,
                 module_residual_dropout: Optional[float] = None,
                 module_activation: Optional[str] = None,
                 module_prenormalization: Optional[bool] = None,
                 module_initialization: Optional[str] = None,
                 module_kv_compression: Optional[str] = None,
                 module_kv_compression_sharing: Optional[str] = None,
                 verbose: Optional[int] = None,
                 max_epochs: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 optimizer: Optional[str] = None,
                 es_patience: Optional[int] = None,
                 lr: Optional[float] = None,
                 lr_scheduler: Optional[bool] = None,
                 lr_patience: Optional[int] = None,
                 optimizer_weight_decay: Optional[float] = None,
                 use_checkpoints: Optional[bool] = None,
                 transformed_target: Optional[bool] = None,
                 tfms: Optional[List[str]] = None,
                 quantile_output_distribution: Optional[str] = None,
                 val_metric_name: Optional[str] = None,
                 device: Optional[str] = None, random_state: Optional[Union[int, np.random.RandomState]] = None,
                 n_cv: int = 1, n_refit: int = 0, n_repeats: int = 1, val_fraction: float = 0.2,
                 n_threads: Optional[int] = None,
                 tmp_folder: Optional[Union[str, pathlib.Path]] = None, verbosity: int = 0,
                 calibration_method: Optional[str] = None,
                 ):
        self.module_d_token = module_d_token
        self.module_d_ffn_factor = module_d_ffn_factor
        self.module_n_layers = module_n_layers
        self.module_n_heads = module_n_heads
        self.module_token_bias = module_token_bias
        self.module_attention_dropout = module_attention_dropout
        self.module_ffn_dropout = module_ffn_dropout
        self.module_residual_dropout = module_residual_dropout
        self.module_activation = module_activation
        self.module_prenormalization = module_prenormalization
        self.module_initialization = module_initialization
        self.module_kv_compression = module_kv_compression
        self.module_kv_compression_sharing = module_kv_compression_sharing
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.es_patience = es_patience
        self.lr_scheduler = lr_scheduler
        self.lr_patience = lr_patience
        self.lr = lr
        self.optimizer_weight_decay = optimizer_weight_decay
        self.use_checkpoints = use_checkpoints
        self.transformed_target = transformed_target
        self.tfms = tfms
        self.quantile_output_distribution = quantile_output_distribution
        self.val_metric_name = val_metric_name
        self.device = device
        self.random_state = random_state
        self.n_cv = n_cv
        self.n_refit = n_refit
        self.n_repeats = n_repeats
        self.val_fraction = val_fraction
        self.n_threads = n_threads
        self.tmp_folder = tmp_folder
        self.verbosity = verbosity
        self.calibration_method = calibration_method


class FTT_D_Classifier(FTTransformerConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self):
        return DefaultParams.FTT_D_CLASS

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.rtdl_interfaces import FTTransformerSubSplitInterface
        return SingleSplitWrapperAlgInterface(
            [FTTransformerSubSplitInterface(**self.get_config()) for i in range(n_cv)])

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']

    def _supports_single_class(self) -> bool:
        return False

    def _supports_single_sample(self) -> bool:
        return False

    def _non_deterministic_tag(self) -> bool:
        # set non-deterministic
        # since this class can otherwise fail the check_methods_subset_invariance test due to low precision (?)
        # only on windows, only recently?? probably a skorch problem?
        return True


class FTT_D_Regressor(FTTransformerConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self):
        return DefaultParams.FTT_D_REG

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.rtdl_interfaces import FTTransformerSubSplitInterface
        return SingleSplitWrapperAlgInterface(
            [FTTransformerSubSplitInterface(**self.get_config()) for i in range(n_cv)])

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']

    def _supports_single_sample(self) -> bool:
        return False

    def _supports_multioutput(self) -> bool:
        return False

    def _non_deterministic_tag(self) -> bool:
        # set non-deterministic
        # since this class can otherwise fail the check_methods_subset_invariance test due to low precision (?)
        return True


class RTDL_MLPConstructorMixin:
    def __init__(self,
                 module_d_embedding: Optional[int] = None,
                 module_d_layers: Optional[int] = None,
                 module_d_first_layer: Optional[int] = None,
                 module_d_last_layer: Optional[int] = None,
                 module_n_layers: Optional[int] = None,
                 module_dropout: Optional[float] = None,
                 verbose: Optional[int] = None,
                 max_epochs: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 optimizer: Optional[str] = None,
                 es_patience: Optional[int] = None,
                 lr: Optional[float] = None,
                 lr_scheduler: Optional[bool] = None,
                 lr_patience: Optional[int] = None,
                 optimizer_weight_decay: Optional[float] = None,
                 use_checkpoints: Optional[bool] = None,
                 transformed_target: Optional[bool] = None,
                 tfms: Optional[List[str]] = None,
                 quantile_output_distribution: Optional[str] = None,
                 val_metric_name: Optional[str] = None,
                 module_num_emb_type: Optional[str] = None,
                 module_num_emb_dim: Optional[int] = None,
                 module_num_emb_hidden_dim: Optional[int] = None,
                 module_num_emb_sigma: Optional[float] = None,
                 module_num_emb_lite: Optional[bool] = None,
                 device: Optional[str] = None, random_state: Optional[Union[int, np.random.RandomState]] = None,
                 n_cv: int = 1, n_refit: int = 0, n_repeats: int = 1, val_fraction: float = 0.2,
                 n_threads: Optional[int] = None,
                 tmp_folder: Optional[Union[str, pathlib.Path]] = None, verbosity: int = 0,
                 calibration_method: Optional[str] = None,
                 ):
        self.module_d_embedding = module_d_embedding
        self.module_d_layers = module_d_layers
        self.module_d_first_layer = module_d_first_layer
        self.module_d_last_layer = module_d_last_layer
        self.module_n_layers = module_n_layers
        self.module_dropout = module_dropout
        self.verbose = verbose
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.es_patience = es_patience
        self.lr_scheduler = lr_scheduler
        self.lr_patience = lr_patience
        self.lr = lr
        self.optimizer_weight_decay = optimizer_weight_decay
        self.use_checkpoints = use_checkpoints
        self.transformed_target = transformed_target
        self.tfms = tfms
        self.quantile_output_distribution = quantile_output_distribution
        self.module_num_emb_type = module_num_emb_type
        self.module_num_emb_dim = module_num_emb_dim
        self.module_num_emb_hidden_dim = module_num_emb_hidden_dim
        self.module_num_emb_sigma = module_num_emb_sigma
        self.module_num_emb_lite = module_num_emb_lite
        self.val_metric_name = val_metric_name
        self.device = device
        self.random_state = random_state
        self.n_cv = n_cv
        self.n_refit = n_refit
        self.n_repeats = n_repeats
        self.val_fraction = val_fraction
        self.n_threads = n_threads
        self.tmp_folder = tmp_folder
        self.verbosity = verbosity
        self.calibration_method = calibration_method


class MLP_RTDL_D_Classifier(RTDL_MLPConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self):
        return DefaultParams.MLP_RTDL_D_CLASS_TabZilla

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.rtdl_interfaces import RTDL_MLPSubSplitInterface
        return SingleSplitWrapperAlgInterface([RTDL_MLPSubSplitInterface(**self.get_config()) for i in range(n_cv)])

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']

    def _supports_single_class(self) -> bool:
        return False

    def _supports_single_sample(self) -> bool:
        return False

    def _non_deterministic_tag(self) -> bool:
        # set non-deterministic
        # since this class can otherwise fail the check_methods_subset_invariance test due to low precision (?)
        # only on windows, only recently?? probably a skorch problem?
        return True


class MLP_RTDL_D_Regressor(RTDL_MLPConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self):
        return DefaultParams.MLP_RTDL_D_REG_TabZilla

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.rtdl_interfaces import RTDL_MLPSubSplitInterface
        return SingleSplitWrapperAlgInterface([RTDL_MLPSubSplitInterface(**self.get_config()) for i in range(n_cv)])

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']

    def _supports_single_sample(self) -> bool:
        return False

    def _supports_multioutput(self) -> bool:
        return False

    def _non_deterministic_tag(self) -> bool:
        # set non-deterministic
        # since this class can otherwise fail the check_methods_subset_invariance test due to low precision (?)
        return True


class MLP_PLR_D_Classifier(RTDL_MLPConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self):
        return DefaultParams.MLP_PLR_D_CLASS

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.rtdl_interfaces import RTDL_MLPSubSplitInterface
        return SingleSplitWrapperAlgInterface([RTDL_MLPSubSplitInterface(**self.get_config()) for i in range(n_cv)])

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']

    def _supports_single_class(self) -> bool:
        return False

    def _supports_single_sample(self) -> bool:
        return False

    def _non_deterministic_tag(self) -> bool:
        # set non-deterministic
        # since this class can otherwise fail the check_methods_subset_invariance test due to low precision (?)
        # only on windows, only recently?? probably a skorch problem?
        return True


class MLP_PLR_D_Regressor(RTDL_MLPConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self):
        return DefaultParams.MLP_PLR_D_REG

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.rtdl_interfaces import RTDL_MLPSubSplitInterface
        return SingleSplitWrapperAlgInterface([RTDL_MLPSubSplitInterface(**self.get_config()) for i in range(n_cv)])

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']

    def _supports_single_sample(self) -> bool:
        return False

    def _supports_multioutput(self) -> bool:
        return False

    def _non_deterministic_tag(self) -> bool:
        # set non-deterministic
        # since this class can otherwise fail the check_methods_subset_invariance test due to low precision (?)
        return True


class TabrConstructorMixin:
    def __init__(self,
                 num_embeddings: Optional[int] = None,
                 d_main: Optional[int] = None,
                 d_multiplier: Optional[int] = None,
                 encoder_n_blocks: Optional[int] = None,
                 predictor_n_blocks: Optional[int] = None,
                 mixer_normalization: Optional[Union[bool, Literal['auto']]] = None,
                 context_dropout: Optional[float] = None,
                 dropout0: Optional[float] = None,
                 dropout1: Optional[float] = None,
                 normalization: Optional[str] = None,
                 activation: Optional[str] = None,
                 memory_efficient: Optional[bool] = None,
                 candidate_encoding_batch_size: Optional[int] = None,
                 n_epochs: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 eval_batch_size: Optional[int] = None,
                 context_size: Optional[int] = None,
                 freeze_contexts_after_n_epochs: Optional[int] = None,
                 optimizer: Optional[Dict] = None,
                 patience: Optional[int] = None,
                 transformed_target: Optional[bool] = None,
                 tfms: Optional[List[str]] = None,
                 quantile_output_distribution: Optional[str] = None,
                 val_metric_name: Optional[str] = None,
                 add_scaling_layer: Optional[bool] = None,
                 scale_lr_factor: Optional[float] = None,
                 use_ntp_linear: Optional[bool] = None,
                 linear_init_type: Optional[str] = None,  # only relevant if use_ntp_linear=True
                 use_ntp_encoder: Optional[bool] = None,
                 ls_eps: Optional[float] = None,
                 device: Optional[str] = None,
                 random_state: Optional[Union[int, np.random.RandomState]] = None,
                 n_cv: int = 1, n_refit: int = 0, n_repeats: int = 1, val_fraction: float = 0.2,
                 n_threads: Optional[int] = None,
                 tmp_folder: Optional[Union[str, pathlib.Path]] = None, verbosity: int = 0,
                 calibration_method: Optional[str] = None,
                 ):
        self.num_embeddings = num_embeddings
        self.d_main = d_main
        self.d_multiplier = d_multiplier
        self.encoder_n_blocks = encoder_n_blocks
        self.predictor_n_blocks = predictor_n_blocks
        self.mixer_normalization = mixer_normalization
        self.context_dropout = context_dropout
        self.dropout0 = dropout0
        self.dropout1 = dropout1
        self.normalization = normalization
        self.activation = activation
        self.memory_efficient = memory_efficient
        self.candidate_encoding_batch_size = candidate_encoding_batch_size
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.context_size = context_size
        self.freeze_contexts_after_n_epochs = freeze_contexts_after_n_epochs
        self.optimizer = optimizer
        self.patience = patience
        self.transformed_target = transformed_target
        self.tfms = tfms
        self.quantile_output_distribution = quantile_output_distribution
        self.val_metric_name = val_metric_name
        self.add_scaling_layer = add_scaling_layer
        self.scale_lr_factor = scale_lr_factor
        self.use_ntp_linear = use_ntp_linear
        self.linear_init_type = linear_init_type
        self.use_ntp_encoder = use_ntp_encoder
        self.ls_eps = ls_eps

        self.device = device
        self.random_state = random_state
        self.n_cv = n_cv
        self.n_refit = n_refit
        self.n_repeats = n_repeats
        self.val_fraction = val_fraction
        self.n_threads = n_threads
        self.tmp_folder = tmp_folder
        self.verbosity = verbosity
        self.calibration_method = calibration_method


class TabR_S_D_Classifier(TabrConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self):
        return DefaultParams.TABR_S_D_CLASS

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.tabr_interface import TabRSubSplitInterface
        return SingleSplitWrapperAlgInterface([TabRSubSplitInterface(**self.get_config()) for i in range(n_cv)])

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']


class TabR_S_D_Regressor(TabrConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self):
        return DefaultParams.TABR_S_D_REG

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.tabr_interface import TabRSubSplitInterface
        return SingleSplitWrapperAlgInterface([TabRSubSplitInterface(**self.get_config()) for i in range(n_cv)])

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']


class RealTabR_D_Classifier(TabrConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self):
        return DefaultParams.RealTABR_D_CLASS

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.tabr_interface import TabRSubSplitInterface
        return SingleSplitWrapperAlgInterface([TabRSubSplitInterface(**self.get_config()) for i in range(n_cv)])

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']


class RealTabR_D_Regressor(TabrConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self):
        return DefaultParams.RealTABR_D_REG

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.tabr_interface import TabRSubSplitInterface
        return SingleSplitWrapperAlgInterface([TabRSubSplitInterface(**self.get_config()) for i in range(n_cv)])

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']


class TabMConstructorMixin:
    def __init__(self, device: Optional[str] = None, random_state: Optional[Union[int, np.random.RandomState]] = None,
                 n_cv: int = 1, n_refit: int = 0, n_repeats: int = 1, val_fraction: float = 0.2,
                 n_threads: Optional[int] = None,
                 tmp_folder: Optional[Union[str, pathlib.Path]] = None, verbosity: int = 0,
                 arch_type: Optional[str] = None,
                 tabm_k: Optional[int] = None,
                 num_emb_type: Optional[str] = None,
                 num_emb_n_bins: Optional[int] = None,
                 batch_size: Optional[int] = None,
                 lr: Optional[float] = None,
                 weight_decay: Optional[float] = None,
                 n_epochs: Optional[int] = None,
                 patience: Optional[int] = None,
                 d_embedding: Optional[int] = None,
                 d_block: Optional[int] = None,
                 n_blocks: Optional[Union[str, int]] = None,
                 dropout: Optional[float] = None,
                 compile_model: Optional[bool] = None,
                 allow_amp: Optional[bool] = None,
                 tfms: Optional[List[str]] = None,
                 gradient_clipping_norm: Optional[Union[float, Literal['none']]] = None,
                 calibration_method: Optional[str] = None,
                 share_training_batches: Optional[bool] = None,
                 val_metric_name: Optional[str] = None,
                 train_metric_name: Optional[str] = None,
                 ):
        """

        :param device: PyTorch device name like 'cpu', 'cuda', 'cuda:0', 'mps' (default=None).
            If None, 'cuda' will be used if available, otherwise 'cpu'.
        :param random_state: Random state to use for random number generation
            (splitting, initialization, batch shuffling). If None, the behavior is not deterministic.
        :param n_cv: Number of cross-validation splits to use (default=1).
            If validation set indices are given in fit(), `n_cv` models will be fitted using different random seeds.
            Otherwise, `n_cv`-fold cross-validation will be used (stratified for classification). If `n_refit=0` is set,
            the prediction will use the average of the models fitted during cross-validation.
            (Averaging is over probabilities for classification, and over outputs for regression.)
            Otherwise, refitted models will be used.
        :param n_refit: Number of models that should be refitted on the training+validation dataset (default=0).
            If zero, only the models from the cross-validation stage are used.
            If positive, `n_refit` models will be fitted on the training+validation dataset (all data given in fit())
            and their predictions will be averaged during predict().
        :param n_repeats: Number of times that the (cross-)validation split should be repeated (default=1).
            Values != 1 are only allowed when no custom validation split is provided.
            Larger number of repeats make things slower but reduce the potential for validation set overfitting,
            especially on smaller datasets.
        :param val_fraction: Fraction of samples used for validation (default=0.2). Has to be in [0, 1).
            Only used if `n_cv==1` and no validation split is provided in fit().
        :param n_threads: Number of threads that the method is allowed to use (default=number of physical cores).
        :param tmp_folder: Temporary folder in which data can be stored during fit().
            (Currently unused for TabM and variants.) If None, methods generally try to not store intermediate data.
        :param verbosity: Verbosity level (default=0, higher means more verbose).
            Set to 2 to see logs from intermediate epochs.
        :param arch_type: Architecture type for TabM, one of ['tabm', 'tabm-mini', 'tabm-normal', 'tabm-mini-normal', 'plain'].
        :param tabm_k: Value of $k$ (number of memory-efficient ensemble members). Default is 32.
        :param num_emb_type: Type of numerical embedding, one of ['none', 'pwl']. Default is 'none'.
            'pwl' stands for piecewise linear embeddings.
        :param num_emb_n_bins: Number of bins for piecewise linear embeddings (default=48).
        Only used when piecewise linear numerical embeddings are used.
        Must be at most the number of training samples, but >1.
        :param batch_size: Batch size, default is 256.
        :param lr: Learning rate, default is 2e-3.
        :param weight_decay: Weight decay, default is 0.
        :param n_epochs: Maximum number of epochs (if early stopping doesn't apply). Default is 1 billion.
        :param patience: Patience for early stopping. Default is 16
        :param d_embedding: Embedding dimension for numerical embeddings.
        :param d_block: Hidden layer size.
        :param n_blocks: Number of linear layers, or 'auto'. Default is 'auto', which will use
            3 when num_emb_type=='none' and 2 otherwise.
        :param dropout: Dropout probability. Default is 0.1.
        :param compile_model: Whether torch.compile should be applied to the model (default=False).
        :param allow_amp: Whether automatic mixed precision should be used if the device is a GPU (default=False).
        :param tfms: Preprocessing transformations, see models.nn_models.models.PreprocessingFactory.
            Default is ['quantile_tabr']. Categorical values will be one-hot encoded by the model.
            Note that in the original experiments, it seems that when cat_policy='ordinal',
            the ordinal-encoded categorical values will later be one-hot encoded by the model.
        :param gradient_clipping_norm: Norm for gradient clipping.
            Default is None from the example code (no gradient clipping), but the experiments from the paper use 1.0.
        :param calibration_method: Post-hoc calibration method (only for classification).
            We recommend 'ts-mix' for fast temperature scaling with Laplace smoothing.
            For other methods, see the get_calibrator method in https://github.com/dholzmueller/probmetrics.
        :param share_training_batches: New in v1.4.1: Whether TabM should use the same training samples
            for each model in the batch (default=False).
            We adopt the default value False from the newer version of TabM,
            while the old code (prior to 1.4.1) was equivalent to share_training_batches=True,
            except that the new code also excludes certain parameters from weight decay.
        :param val_metric_name: Name of the validation metric used for early stopping.
            For classification, the default is 'class_error' but could be 'cross_entropy', 'brier', '1-auc_ovr' etc.
            For regression, the default is 'rmse' but could be 'mae'.
        :param train_metric_name: Name of the metric (loss) used for training.
            For classification, the default is 'cross_entropy'.
            For regression, it is 'mse' but could be set to something like 'multi_pinball(0.05,0.95)'.
        """
        self.device = device
        self.random_state = random_state
        self.n_cv = n_cv
        self.n_refit = n_refit
        self.n_repeats = n_repeats
        self.val_fraction = val_fraction
        self.n_threads = n_threads
        self.tmp_folder = tmp_folder
        self.verbosity = verbosity

        self.arch_type = arch_type
        self.num_emb_type = num_emb_type
        self.num_emb_n_bins = num_emb_n_bins
        self.n_epochs = n_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.compile_model = compile_model
        self.lr = lr
        self.weight_decay = weight_decay
        self.d_embedding = d_embedding
        self.d_block = d_block
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.tabm_k = tabm_k
        self.allow_amp = allow_amp
        self.tfms = tfms
        self.gradient_clipping_norm = gradient_clipping_norm
        self.calibration_method = calibration_method
        self.share_training_batches = share_training_batches
        self.val_metric_name = val_metric_name
        self.train_metric_name = train_metric_name


class TabM_D_Classifier(TabMConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self):
        return DefaultParams.TABM_D_CLASS

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.tabm_interface import TabMSubSplitInterface
        return SingleSplitWrapperAlgInterface([TabMSubSplitInterface(**self.get_config()) for i in range(n_cv)])

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']

    def _supports_single_class(self) -> bool:
        return False

    def _supports_single_sample(self) -> bool:
        return False


class TabM_D_Regressor(TabMConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self):
        return DefaultParams.TABM_D_REG

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.tabm_interface import TabMSubSplitInterface
        return SingleSplitWrapperAlgInterface([TabMSubSplitInterface(**self.get_config()) for i in range(n_cv)])

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']

    def _supports_multioutput(self) -> bool:
        return False

    def _supports_single_sample(self) -> bool:
        return False


class TabM_HPO_Classifier(RealMLPHPOConstructorMixin, AlgInterfaceClassifier):
    """
    HPO spaces ('default', 'tabarena') use TabM-mini with numerical embeddings
    """
    def _get_default_params(self):
        return dict(n_hyperopt_steps=50)

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.tabm_interface import RandomParamsTabMAlgInterface
        config = self.get_config()
        n_hyperopt_steps = config['n_hyperopt_steps']
        interface_type = CaruanaEnsembleAlgInterface if config.get('use_caruana_ensembling',
                                                                   False) else AlgorithmSelectionAlgInterface
        return interface_type([RandomParamsTabMAlgInterface(model_idx=i, **config)
                               for i in range(n_hyperopt_steps)], **config)

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']


class TabM_HPO_Regressor(RealMLPHPOConstructorMixin, AlgInterfaceRegressor):
    """
    HPO spaces ('default', 'tabarena') use TabM-mini with numerical embeddings
    """
    def _get_default_params(self):
        return dict(n_hyperopt_steps=50)

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.tabm_interface import RandomParamsTabMAlgInterface
        config = self.get_config()
        n_hyperopt_steps = config['n_hyperopt_steps']
        interface_type = CaruanaEnsembleAlgInterface if config.get('use_caruana_ensembling',
                                                                   False) else AlgorithmSelectionAlgInterface
        return interface_type([RandomParamsTabMAlgInterface(model_idx=i, **config)
                               for i in range(n_hyperopt_steps)], **config)

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']


# ------------------------------

class XRFMConstructorMixin:
    def __init__(self, device: Optional[str] = None, random_state: Optional[Union[int, np.random.RandomState]] = None,
                 n_cv: int = 1, n_refit: int = 0, n_repeats: int = 1, val_fraction: float = 0.2,
                 n_threads: Optional[int] = None,
                 tmp_folder: Optional[Union[str, pathlib.Path]] = None, verbosity: int = 0,
                 bandwidth: Optional[float] = None,
                 p_interp: Optional[float] = None,
                 exponent: Optional[float] = None,
                 reg: Optional[float] = None,
                 iters: Optional[int] = None,
                 diag: Optional[bool] = None,
                 bandwidth_mode: Optional[str] = None,
                 kernel_type: Optional[str] = None,
                 max_leaf_samples: Optional[int] = None,
                 val_metric_name: Optional[str] = None,
                 early_stop_rfm: Optional[bool] = None,
                 early_stop_multiplier: Optional[float] = None,
                 classification_mode: Optional[str] = None,
                 calibration_method: Optional[str] = None,
                 time_limit_s: Optional[float] = None,
                 M_batch_size: Optional[int] = None,
                 ):
        """
        xRFM. In case of out-of-memory, try reducing M_batch_size and/or max_leaf_samples.
        Some parameters generally benefit a lot from tuning, such as the regularization (reg).

        :param device: PyTorch device name like 'cpu', 'cuda', 'cuda:0', 'mps' (default=None).
            If None, 'cuda' will be used if available, otherwise 'cpu'.
        :param random_state: Random state to use for random number generation
            (splitting, initialization, batch shuffling). If None, the behavior is not deterministic.
        :param n_cv: Number of cross-validation splits to use (default=1).
            If validation set indices are given in fit(), `n_cv` models will be fitted using different random seeds.
            Otherwise, `n_cv`-fold cross-validation will be used (stratified for classification). If `n_refit=0` is set,
            the prediction will use the average of the models fitted during cross-validation.
            (Averaging is over probabilities for classification, and over outputs for regression.)
            Otherwise, refitted models will be used.
        :param n_refit: Number of models that should be refitted on the training+validation dataset (default=0).
            If zero, only the models from the cross-validation stage are used.
            If positive, `n_refit` models will be fitted on the training+validation dataset (all data given in fit())
            and their predictions will be averaged during predict().
        :param n_repeats: Number of times that the (cross-)validation split should be repeated (default=1).
            Values != 1 are only allowed when no custom validation split is provided.
            Larger number of repeats make things slower but reduce the potential for validation set overfitting,
            especially on smaller datasets.
        :param val_fraction: Fraction of samples used for validation (default=0.2). Has to be in [0, 1).
            Only used if `n_cv==1` and no validation split is provided in fit().
        :param n_threads: Number of threads that the method is allowed to use (default=number of physical cores).
        :param tmp_folder: Temporary folder in which data can be stored during fit().
            (Currently unused for xRFM and variants.) If None, methods generally try to not store intermediate data.
        :param verbosity: Verbosity level (default=0, higher means more verbose).
        :param bandwidth: Bandwidth of the kernel, i.e., how wide the kernel is (default=10).
        :param p_interp: For kernel_type='lpq', this parameter controls the parameter p of the L_p norm
            in the exponent of the kernel. Specifically, we set p = 2 * p_interp + exponent * (1 - p_interp).
            Should be in [0, 1].
        :param exponent: Exponent of the norm inside the kernel (default=1). Should be in (0, 2].
            Recommended values are in [0.7, 1.4].
        :param reg: Regularization parameter lambda in the kernel ridge regression (default=1e-3).
        :param iters: How many iterations (fitting the regressor, updating the AGOP matrix) should be done (default=5).
            The default should be good for most cases.
        :param diag: Whether to only fit a diagonal AGOP matrix (default=True).
        :param bandwidth_mode: How to set the bandwidth (default='constant').
            For 'constant', the specified bandwidth will be used directly.
            For 'adaptive', it will be scaled relative to the median distance between samples.
            We recommend 'constant' for smaller datasets (< max_leaf_samples) where only a single RFM is fit.
            For larger datasets, 'adaptive' may be more suited since it can adapt the bandwidth to the data in the leaf.
        :param kernel_type: Type of kernel (default='l2').
            For 'l2', the L_2-norm will be used in the generalized Laplace kernel exp(-||x - x'||_2^q),
            where q is the exponent. This is the fastest kernel and a good default.
            For 'lpq', the slower exp(-||x - x'||_p^q) will be used, where p is determined from q and p_interp.
            It will use the kermac implementation if kermac is installed.
        :param max_leaf_samples: Maximum number of samples in a leaf of xRFM (default=60_000).
            For datasets with more than max_leaf_samples samples, the memory usage is O(max_leaf_samples**2)
            and the time complexity is roughly O(n_samples * max_leaf_samples**2).
            The default is around 60000, which is optimized for GPUs with ~40 GB of VRAM.
            Reduce this number to reduce the RAM usage.
            On GPUs with less VRAM, this number can be automatically lowered to avoid exceeding the maximum RAM.
        :param val_metric_name: Name of the validation metric (used for selecting the best iteration).
            Defaults are 'class_error' for classification and 'rmse' for regression.
            Available classification metrics (all to be minimized): 'class_error', 'cross_entropy', '1-auroc-ovr',
            'brier'.
            Available regression metrics: 'rmse'.
        :param early_stop_rfm: Whether to stop the iterations early if the error stops decreasing (default=False).
        :param early_stop_multiplier: Tolerance for early stopping, should be larger than one (default=1.1).
            Larger values will early-stop less aggressively.
        :param classification_mode: How to convert classification problems to regression problems internally
            (default='zero_one'). 'zero_one' uses a one-hot encoding,
            while 'prevalence' uses a simplex encoding with zero corresponding to the marginal class ratio.
        :param calibration_method: Post-hoc calibration method (only for classification) (default=None).
            We recommend 'ts-mix' for fast temperature scaling with Laplace smoothing.
            For other methods, see the get_calibrator method in https://github.com/dholzmueller/probmetrics.
        :param time_limit_s:
            Time limit in seconds (default=None).
        :param M_batch_size:
            Batch size used to construct the AGOP matrix M (default=8000).
            Higher values can speed up the computation but may lead to out-of-memory (esp. for the 'lpq' kernel).
        """

        self.device = device
        self.random_state = random_state
        self.n_cv = n_cv
        self.n_refit = n_refit
        self.n_repeats = n_repeats
        self.val_fraction = val_fraction
        self.n_threads = n_threads
        self.tmp_folder = tmp_folder
        self.verbosity = verbosity

        self.bandwidth = bandwidth
        self.p_interp = p_interp
        self.exponent = exponent
        self.reg = reg
        self.iters = iters
        self.diag = diag
        self.bandwidth_mode = bandwidth_mode
        self.kernel_type = kernel_type
        self.max_leaf_samples = max_leaf_samples
        self.val_metric_name = val_metric_name
        self.early_stop_rfm = early_stop_rfm
        self.early_stop_multiplier = early_stop_multiplier
        self.classification_mode = classification_mode
        self.calibration_method = calibration_method
        self.time_limit_s = time_limit_s
        self.M_batch_size = M_batch_size


class XRFM_D_Classifier(XRFMConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self):
        return DefaultParams.XRFM_D_CLASS

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.xrfm_interfaces import xRFMSubSplitInterface
        return SingleSplitWrapperAlgInterface([xRFMSubSplitInterface(**self.get_config()) for i in range(n_cv)])

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']

    def _non_deterministic_tag(self) -> bool:
        # set non-deterministic
        # since this class can otherwise fail the check_methods_subset_invariance test due to low precision (?)
        return True

    def _supports_single_sample(self) -> bool:
        return False

    def _supports_multioutput(self) -> bool:
        return False



class XRFM_D_Regressor(XRFMConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self):
        return DefaultParams.XRFM_D_REG

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.xrfm_interfaces import xRFMSubSplitInterface
        return SingleSplitWrapperAlgInterface([xRFMSubSplitInterface(**self.get_config()) for i in range(n_cv)])

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']

    def _non_deterministic_tag(self) -> bool:
        # set non-deterministic
        # since this class can otherwise fail the check_methods_subset_invariance test due to low precision (?)
        return True

    def _supports_single_sample(self) -> bool:
        return False

    def _supports_multioutput(self) -> bool:
        return False


class XRFMHPOConstructorMixin:
    def __init__(self, device: Optional[str] = None, random_state: Optional[Union[int, np.random.RandomState]] = None,
                 n_cv: int = 1, n_refit: int = 0, n_repeats: int = 1, val_fraction: float = 0.2,
                 n_threads: Optional[int] = None,
                 tmp_folder: Optional[Union[str, pathlib.Path]] = None, verbosity: int = 0,
                 n_hyperopt_steps: Optional[int] = None, val_metric_name: Optional[str] = None,
                 max_leaf_samples: Optional[int] = None,
                 M_batch_size: Optional[int] = None,
                 bandwidth_mode: Optional[str] = None,
                 calibration_method: Optional[str] = None, hpo_space_name: Optional[str] = None,
                 n_caruana_steps: Optional[int] = None,
                 use_caruana_ensembling: Optional[bool] = None,
                 time_limit_s: Optional[float] = None,
                 ):
        """

        :param device: PyTorch device name like 'cpu', 'cuda', 'cuda:0', 'mps' (default=None).
            If None, 'cuda' will be used if available, otherwise 'cpu'.
        :param random_state: Random state to use for random number generation
            (splitting, initialization, batch shuffling). If None, the behavior is not deterministic.
        :param n_cv: Number of cross-validation splits to use (default=1).
            If validation set indices or an explicit validation set are given in fit(),
            `n_cv` models will be fitted using different random seeds.
            Otherwise, `n_cv`-fold cross-validation will be used (stratified for classification).
            For n_cv=1, a single train-validation split will be used,
            where `val_fraction` controls the fraction of validation samples.
            If `n_refit=0` is set,
            the prediction will use the average of the models fitted during cross-validation.
            (Averaging is over probabilities for classification, and over outputs for regression.)
            Otherwise, refitted models will be used.
        :param n_refit: Number of models that should be refitted on the training+validation dataset (default=0).
            If zero, only the models from the cross-validation stage are used.
            If positive, `n_refit` models will be fitted on the training+validation dataset (all data given in fit())
            and their predictions will be averaged during predict().
        :param n_repeats: Number of times that the (cross-)validation split should be repeated (default=1).
            Values != 1 are only allowed when no custom validation split is provided.
            Larger number of repeats make things slower but reduce the potential for validation set overfitting,
            especially on smaller datasets.
        :param val_fraction: Fraction of samples used for validation (default=0.2). Has to be in [0, 1).
            Only used if `n_cv==1` and no validation split is provided in fit().
        :param n_threads: Number of threads that the method is allowed to use (default=number of physical cores).
        :param tmp_folder: Folder in which models can be stored.
            Setting this allows reducing RAM/VRAM usage by not having all models in RAM at the same time.
            In this case, the folder needs to be preserved as long as the model exists
            (including when the model is pickled to disk).
        :param verbosity: Verbosity level (default=0, higher means more verbose).
            Set to 2 to see logs from intermediate epochs.
        :param n_hyperopt_steps: Number of random hyperparameter configs
            that should be used to train models (default=50).
        :param val_metric_name: Name of the validation metric (used for selecting the best epoch).
            Defaults are 'class_error' for classification and 'rmse' for regression.
            Main available classification metrics (all to be minimized): 'class_error', 'cross_entropy', '1-auc_ovo',
            '1-auc_ovr', '1-auc_mu', 'brier', '1-balanced_accuracy', '1-mcc', 'ece'.
            Main available regression metrics: 'rmse', 'mae', 'max_error',
            'pinball(0.95)' (also works with other quantiles specified directly in the string).
            For more metrics, we refer to `models.training.metrics.Metrics.apply()`.
        :param max_leaf_samples: Maximum number of samples in a leaf of xRFM.
            For datasets with more than max_leaf_samples samples, the memory usage is O(max_leaf_samples**2)
            and the time complexity is roughly O(n_samples * max_leaf_samples**2).
            The default is around 60000, which is optimized for GPUs with ~40 GB of VRAM.
            Reduce this number to reduce the RAM usage.
        :param M_batch_size:
            Batch size used to construct the AGOP matrix M (default=8000).
            Higher values can speed up the computation but may lead to out-of-memory (esp. for the 'lpq' kernel).
        :param bandwidth_mode: How to set the bandwidth (default='constant').
            For 'constant', the specified bandwidth will be used directly.
            For 'adaptive', it will be scaled relative to the median distance between samples.
            We recommend 'constant' for smaller datasets (< max_leaf_samples) where only a single RFM is fit.
            For larger datasets, 'adaptive' may be more suited since it can adapt the bandwidth to the data in the leaf.
        :param calibration_method: Post-hoc calibration method (only for classification) (default=None).
            We recommend 'ts-mix' for fast temperature scaling with Laplace smoothing.
            For other methods, see the get_calibrator method in https://github.com/dholzmueller/probmetrics.
        :param hpo_space_name: Name of the HPO space.
            We recommend using 'tabarena' (the default) for the best results.
        :param n_caruana_steps: Number of weight update iterations for Caruana et al. weighted ensembling (default=40).
            This parameter is only used when use_caruana_ensembling=True.
        :param use_caruana_ensembling: Whether to use the algorithm by Caruana et al. (2004)
            to select a weighted ensemble of models instead of only selecting the best model (default=False).
        :param time_limit_s: Time limit in seconds (default=None).
        """
        self.device = device
        self.random_state = random_state
        self.n_cv = n_cv
        self.n_refit = n_refit
        self.n_repeats = n_repeats
        self.val_fraction = val_fraction
        self.n_threads = n_threads
        self.tmp_folder = tmp_folder
        self.verbosity = verbosity
        self.n_hyperopt_steps = n_hyperopt_steps
        self.val_metric_name = val_metric_name
        self.max_leaf_samples = max_leaf_samples
        self.M_batch_size = M_batch_size
        self.bandwidth_mode = bandwidth_mode
        self.calibration_method = calibration_method
        self.hpo_space_name = hpo_space_name
        self.n_caruana_steps = n_caruana_steps
        self.use_caruana_ensembling = use_caruana_ensembling
        self.time_limit_s = time_limit_s



class XRFM_HPO_Classifier(XRFMHPOConstructorMixin, AlgInterfaceClassifier):
    """
    HPO spaces ('default') use xRFM
    """
    def _get_default_params(self):
        return dict(n_hyperopt_steps=50)

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.xrfm_interfaces import RandomParamsxRFMAlgInterface
        config = self.get_config()
        n_hyperopt_steps = config['n_hyperopt_steps']
        interface_type = CaruanaEnsembleAlgInterface if config.get('use_caruana_ensembling',
                                                                   False) else AlgorithmSelectionAlgInterface
        return interface_type([RandomParamsxRFMAlgInterface(model_idx=i, **config)
                               for i in range(n_hyperopt_steps)], **config)

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']

    def _supports_single_sample(self) -> bool:
        return False

    def _supports_multioutput(self) -> bool:
        return False


class XRFM_HPO_Regressor(XRFMHPOConstructorMixin, AlgInterfaceRegressor):
    """
    HPO spaces ('default', 'tabarena') use TabM-mini with numerical embeddings
    """
    def _get_default_params(self):
        return dict(n_hyperopt_steps=50)

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.xrfm_interfaces import RandomParamsxRFMAlgInterface
        config = self.get_config()
        n_hyperopt_steps = config['n_hyperopt_steps']
        interface_type = CaruanaEnsembleAlgInterface if config.get('use_caruana_ensembling',
                                                                   False) else AlgorithmSelectionAlgInterface
        return interface_type([RandomParamsxRFMAlgInterface(model_idx=i, **config)
                               for i in range(n_hyperopt_steps)], **config)

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']

    def _supports_single_sample(self) -> bool:
        return False

    def _supports_multioutput(self) -> bool:
        return False


# ------------------------------


class MLP_RTDL_HPO_Classifier(RealMLPHPOConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self):
        return dict(n_hyperopt_steps=50)

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.rtdl_interfaces import RandomParamsRTDLMLPAlgInterface
        config = self.get_config()
        n_hyperopt_steps = config['n_hyperopt_steps']
        interface_type = CaruanaEnsembleAlgInterface if config.get('use_caruana_ensembling',
                                                                   False) else AlgorithmSelectionAlgInterface
        return interface_type([RandomParamsRTDLMLPAlgInterface(model_idx=i, **config)
                               for i in range(n_hyperopt_steps)], **config)

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']


class MLP_RTDL_HPO_Regressor(RealMLPHPOConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self):
        return dict(n_hyperopt_steps=50)

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.rtdl_interfaces import RandomParamsRTDLMLPAlgInterface
        config = self.get_config()
        n_hyperopt_steps = config['n_hyperopt_steps']
        interface_type = CaruanaEnsembleAlgInterface if config.get('use_caruana_ensembling',
                                                                   False) else AlgorithmSelectionAlgInterface
        return interface_type([RandomParamsRTDLMLPAlgInterface(model_idx=i, **config)
                               for i in range(n_hyperopt_steps)], **config)

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']


class MLP_PLR_HPO_Classifier(RealMLPHPOConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self):
        return dict(n_hyperopt_steps=50)

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.rtdl_interfaces import RandomParamsRTDLMLPAlgInterface
        config = self.get_config()
        n_hyperopt_steps = config['n_hyperopt_steps']
        interface_type = CaruanaEnsembleAlgInterface if config.get('use_caruana_ensembling',
                                                                   False) else AlgorithmSelectionAlgInterface
        return interface_type([RandomParamsRTDLMLPAlgInterface(model_idx=i, num_emb_type='plr',
                                                               **config)
                               for i in range(n_hyperopt_steps)], **config)

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']


class MLP_PLR_HPO_Regressor(RealMLPHPOConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self):
        return dict(n_hyperopt_steps=50)

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.rtdl_interfaces import RandomParamsRTDLMLPAlgInterface
        config = self.get_config()
        n_hyperopt_steps = config['n_hyperopt_steps']
        interface_type = CaruanaEnsembleAlgInterface if config.get('use_caruana_ensembling',
                                                                   False) else AlgorithmSelectionAlgInterface
        return interface_type(
            [RandomParamsRTDLMLPAlgInterface(model_idx=i, num_emb_type='plr', **config)
             for i in range(n_hyperopt_steps)], **config)

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']


class Resnet_RTDL_HPO_Classifier(RealMLPHPOConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self):
        return dict(n_hyperopt_steps=50)

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.rtdl_interfaces import RandomParamsResnetAlgInterface
        config = self.get_config()
        n_hyperopt_steps = config['n_hyperopt_steps']
        interface_type = CaruanaEnsembleAlgInterface if config.get('use_caruana_ensembling',
                                                                   False) else AlgorithmSelectionAlgInterface
        return interface_type([RandomParamsResnetAlgInterface(model_idx=i, **config)
                               for i in range(n_hyperopt_steps)], **config)

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']


class Resnet_RTDL_HPO_Regressor(RealMLPHPOConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self):
        return dict(n_hyperopt_steps=50)

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.rtdl_interfaces import RandomParamsResnetAlgInterface
        config = self.get_config()
        n_hyperopt_steps = config['n_hyperopt_steps']
        interface_type = CaruanaEnsembleAlgInterface if config.get('use_caruana_ensembling',
                                                                   False) else AlgorithmSelectionAlgInterface
        return interface_type([RandomParamsResnetAlgInterface(model_idx=i, **config)
                               for i in range(n_hyperopt_steps)], **config)

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']


class FTT_HPO_Classifier(RealMLPHPOConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self):
        return dict(n_hyperopt_steps=50)

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.rtdl_interfaces import RandomParamsFTTransformerAlgInterface
        config = self.get_config()
        n_hyperopt_steps = config['n_hyperopt_steps']
        interface_type = CaruanaEnsembleAlgInterface if config.get('use_caruana_ensembling',
                                                                   False) else AlgorithmSelectionAlgInterface
        return interface_type([RandomParamsFTTransformerAlgInterface(model_idx=i, **config)
                               for i in range(n_hyperopt_steps)], **config)

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']


class FTT_HPO_Regressor(RealMLPHPOConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self):
        return dict(n_hyperopt_steps=50)

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.rtdl_interfaces import RandomParamsFTTransformerAlgInterface
        config = self.get_config()
        n_hyperopt_steps = config['n_hyperopt_steps']
        interface_type = CaruanaEnsembleAlgInterface if config.get('use_caruana_ensembling',
                                                                   False) else AlgorithmSelectionAlgInterface
        return interface_type([RandomParamsFTTransformerAlgInterface(model_idx=i, **config)
                               for i in range(n_hyperopt_steps)], **config)

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']


class TabR_HPO_Classifier(RealMLPHPOConstructorMixin, AlgInterfaceClassifier):
    def _get_default_params(self):
        return dict(n_hyperopt_steps=50)

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.tabr_interface import RandomParamsTabRAlgInterface
        config = self.get_config()
        n_hyperopt_steps = config['n_hyperopt_steps']
        interface_type = CaruanaEnsembleAlgInterface if config.get('use_caruana_ensembling',
                                                                   False) else AlgorithmSelectionAlgInterface
        return interface_type([RandomParamsTabRAlgInterface(model_idx=i, **config)
                               for i in range(n_hyperopt_steps)], **config)

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']


class TabR_HPO_Regressor(RealMLPHPOConstructorMixin, AlgInterfaceRegressor):
    def _get_default_params(self):
        return dict(n_hyperopt_steps=50)

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.tabr_interface import RandomParamsTabRAlgInterface
        config = self.get_config()
        n_hyperopt_steps = config['n_hyperopt_steps']
        interface_type = CaruanaEnsembleAlgInterface if config.get('use_caruana_ensembling',
                                                                   False) else AlgorithmSelectionAlgInterface
        return interface_type([RandomParamsTabRAlgInterface(model_idx=i, **config)
                               for i in range(n_hyperopt_steps)], **config)

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']


# Ensemble-TD

class Ensemble_TD_Classifier(AlgInterfaceClassifier):
    def __init__(self, device: Optional[str] = None, random_state: Optional[Union[int, np.random.RandomState]] = None,
                 n_cv: int = 1, n_refit: int = 0, n_repeats: int = 1, val_fraction: float = 0.2,
                 n_threads: Optional[int] = None,
                 tmp_folder: Optional[Union[str, pathlib.Path]] = None, verbosity: int = 0,
                 val_metric_name: Optional[str] = None, use_ls: Optional[bool] = None,
                 calibration_method: Optional[str] = None):
        self.device = device
        self.random_state = random_state
        self.n_cv = n_cv
        self.n_refit = n_refit
        self.n_repeats = n_repeats
        self.val_fraction = val_fraction
        self.n_threads = n_threads
        self.tmp_folder = tmp_folder
        self.verbosity = verbosity
        self.val_metric_name = val_metric_name
        self.use_ls = use_ls
        self.calibration_method = calibration_method

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.catboost_interfaces import CatBoostSubSplitInterface
        from pytabkit.models.alg_interfaces.lightgbm_interfaces import LGBMSubSplitInterface
        from pytabkit.models.alg_interfaces.xgboost_interfaces import XGBSubSplitInterface

        extra_params = dict()
        if self.val_metric_name is not None:
            extra_params['val_metric_name'] = self.val_metric_name
        if self.use_ls is not None:
            extra_params['use_ls'] = self.use_ls
        if self.calibration_method is not None:
            extra_params['calibration_method'] = self.calibration_method
        td_interfaces = [
            SingleSplitWrapperAlgInterface(
                [LGBMSubSplitInterface(**DefaultParams.LGBM_TD_CLASS, **extra_params, allow_gpu=False) for i in
                 range(n_cv)]),
            SingleSplitWrapperAlgInterface(
                [XGBSubSplitInterface(**DefaultParams.XGB_TD_CLASS, **extra_params, allow_gpu=False) for i in
                 range(n_cv)]),
            SingleSplitWrapperAlgInterface(
                [CatBoostSubSplitInterface(**DefaultParams.CB_TD_CLASS, **extra_params, allow_gpu=False) for i in
                 range(n_cv)]),
            NNAlgInterface(**utils.join_dicts(DefaultParams.RealMLP_TD_CLASS, extra_params)),
        ]
        return CaruanaEnsembleAlgInterface(td_interfaces, **extra_params)

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']


class Ensemble_TD_Regressor(AlgInterfaceRegressor):
    def __init__(self, device: Optional[str] = None, random_state: Optional[Union[int, np.random.RandomState]] = None,
                 n_cv: int = 1, n_refit: int = 0, n_repeats: int = 1, val_fraction: float = 0.2,
                 n_threads: Optional[int] = None,
                 tmp_folder: Optional[Union[str, pathlib.Path]] = None, verbosity: int = 0,
                 val_metric_name: Optional[str] = None):
        self.device = device
        self.random_state = random_state
        self.n_cv = n_cv
        self.n_refit = n_refit
        self.n_repeats = n_repeats
        self.val_fraction = val_fraction
        self.n_threads = n_threads
        self.tmp_folder = tmp_folder
        self.verbosity = verbosity
        self.val_metric_name = val_metric_name

    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.catboost_interfaces import CatBoostSubSplitInterface
        from pytabkit.models.alg_interfaces.lightgbm_interfaces import LGBMSubSplitInterface
        from pytabkit.models.alg_interfaces.xgboost_interfaces import XGBSubSplitInterface

        extra_params = dict()
        if self.val_metric_name is not None:
            extra_params['val_metric_name'] = self.val_metric_name
        td_interfaces = [
            SingleSplitWrapperAlgInterface(
                [LGBMSubSplitInterface(**DefaultParams.LGBM_TD_REG, **extra_params, allow_gpu=False) for i in
                 range(n_cv)]),
            SingleSplitWrapperAlgInterface(
                [XGBSubSplitInterface(**DefaultParams.XGB_TD_REG, **extra_params, allow_gpu=False) for i in
                 range(n_cv)]),
            SingleSplitWrapperAlgInterface(
                [CatBoostSubSplitInterface(**DefaultParams.CB_TD_REG, **extra_params, allow_gpu=False) for i in
                 range(n_cv)]),
            NNAlgInterface(**DefaultParams.RealMLP_TD_REG, **extra_params),
        ]
        return CaruanaEnsembleAlgInterface(td_interfaces, **extra_params)

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']


class EnsembleHPOConstructorMixin:
    def __init__(self, device: Optional[str] = None, random_state: Optional[Union[int, np.random.RandomState]] = None,
                 n_cv: int = 1, n_refit: int = 0, n_repeats: int = 1, val_fraction: float = 0.2,
                 n_threads: Optional[int] = None,
                 tmp_folder: Optional[Union[str, pathlib.Path]] = None, verbosity: int = 0,
                 val_metric_name: Optional[str] = None,
                 n_hpo_steps: int = 50,
                 calibration_method: Optional[str] = None,
                 use_full_caruana_ensembling: bool = False,
                 n_caruana_steps: int = 40,
                 use_tabarena_spaces: bool = False,
                 time_limit_s: Optional[float] = None,
                 ):
        """
        :param device:
        :param random_state:
        :param n_cv:
        :param n_refit:
        :param n_repeats:
        :param val_fraction:
        :param n_threads:
        :param tmp_folder:
        :param verbosity:
        :param val_metric_name:
        :param n_hpo_steps: Number of HPO configs per method.
        :param calibration_method: Calibration method (only for classification).
        :param use_full_caruana_ensembling: Whether to also ensemble
            different hyperparameter configs of the same method (default=False).
            False corresponds to the method used in the paper,
            True should give better results (with larger inference time).
        :param n_caruana_steps: How many iterations to use for Caruana et al. (2004) weighted ensembling.
        :param use_tabarena_spaces: Whether to use search spaces from TabArena instead of from the RealMLP paper.
        :param time_limit_s: Time limit in seconds (default=None).
        """
        self.device = device
        self.random_state = random_state
        self.n_cv = n_cv
        self.n_refit = n_refit
        self.n_repeats = n_repeats
        self.val_fraction = val_fraction
        self.n_threads = n_threads
        self.tmp_folder = tmp_folder
        self.verbosity = verbosity
        self.val_metric_name = val_metric_name
        self.n_hpo_steps = n_hpo_steps
        self.calibration_method = calibration_method
        self.use_full_caruana_ensembling = use_full_caruana_ensembling
        self.n_caruana_steps = n_caruana_steps
        self.use_tabarena_spaces = use_tabarena_spaces
        self.time_limit_s = time_limit_s


class Ensemble_HPO_Classifier(EnsembleHPOConstructorMixin, AlgInterfaceClassifier):
    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.catboost_interfaces import RandomParamsCatBoostAlgInterface
        from pytabkit.models.alg_interfaces.lightgbm_interfaces import RandomParamsLGBMAlgInterface
        from pytabkit.models.alg_interfaces.xgboost_interfaces import RandomParamsXGBAlgInterface

        extra_params = dict()
        if self.val_metric_name is not None:
            extra_params['val_metric_name'] = self.val_metric_name
        if self.calibration_method is not None:
            extra_params['calibration_method'] = self.calibration_method
        if self.use_tabarena_spaces:
            extra_params['hpo_space_name'] = 'tabarena'
        extra_params['n_caruana_steps'] = self.n_caruana_steps
        extra_params['time_limit_s'] = self.time_limit_s
        n_hpo_steps = self.n_hpo_steps or 50
        hpo_configs = [
            [RandomParamsLGBMAlgInterface(model_idx=i, **extra_params, allow_gpu=False) for i in
                 range(n_hpo_steps)],
            [RandomParamsXGBAlgInterface(model_idx=i, **extra_params, allow_gpu=False) for i in
                 range(n_hpo_steps)],
            [RandomParamsCatBoostAlgInterface(model_idx=i, **extra_params, allow_gpu=False) for i in
                 range(n_hpo_steps)],
            [RandomParamsNNAlgInterface(model_idx=i, **extra_params) for i in range(n_hpo_steps)],
        ]
        if self.use_full_caruana_ensembling:
            hpo_interfaces = sum(hpo_configs, [])
        else:
            hpo_interfaces = [AlgorithmSelectionAlgInterface(lst, **extra_params) for lst in hpo_configs]

        return CaruanaEnsembleAlgInterface(hpo_interfaces, **extra_params)

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']


class Ensemble_HPO_Regressor(EnsembleHPOConstructorMixin, AlgInterfaceRegressor):
    def _create_alg_interface(self, n_cv: int) -> AlgInterface:
        from pytabkit.models.alg_interfaces.catboost_interfaces import RandomParamsCatBoostAlgInterface
        from pytabkit.models.alg_interfaces.lightgbm_interfaces import RandomParamsLGBMAlgInterface
        from pytabkit.models.alg_interfaces.xgboost_interfaces import RandomParamsXGBAlgInterface

        extra_params = dict()
        if self.val_metric_name is not None:
            extra_params['val_metric_name'] = self.val_metric_name
        if self.use_tabarena_spaces:
            extra_params['hpo_space_name'] = 'tabarena'
        extra_params['n_caruana_steps'] = self.n_caruana_steps
        extra_params['time_limit_s'] = self.time_limit_s
        n_hpo_steps = self.n_hpo_steps or 50
        hpo_configs = [
            [RandomParamsLGBMAlgInterface(model_idx=i, **extra_params, allow_gpu=False) for i in
             range(n_hpo_steps)],
            [RandomParamsXGBAlgInterface(model_idx=i, **extra_params, allow_gpu=False) for i in
             range(n_hpo_steps)],
            [RandomParamsCatBoostAlgInterface(model_idx=i, **extra_params, allow_gpu=False) for i in
             range(n_hpo_steps)],
            [RandomParamsNNAlgInterface(model_idx=i, **extra_params) for i in range(n_hpo_steps)],
        ]
        if self.use_full_caruana_ensembling:
            hpo_interfaces = sum(hpo_configs, [])
        else:
            hpo_interfaces = [AlgorithmSelectionAlgInterface(lst, **extra_params) for lst in hpo_configs]
        return CaruanaEnsembleAlgInterface(hpo_interfaces, **extra_params)

    def _allowed_device_names(self) -> List[str]:
        return ['cpu', 'cuda', 'mps']
