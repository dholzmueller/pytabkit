import shutil
from pathlib import Path
from typing import Callable, List, Optional

import torch

from pytabkit.bench.data.paths import Paths
from pytabkit.models import utils
from pytabkit.models.alg_interfaces.autogluon_model_interfaces import AutoGluonModelAlgInterface
from pytabkit.models.alg_interfaces.catboost_interfaces import CatBoostSubSplitInterface, CatBoostHyperoptAlgInterface, \
    CatBoostSklearnSubSplitInterface, RandomParamsCatBoostAlgInterface
from pytabkit.models.alg_interfaces.ensemble_interfaces import PrecomputedPredictionsAlgInterface, \
    CaruanaEnsembleAlgInterface, AlgorithmSelectionAlgInterface
from pytabkit.models.alg_interfaces.lightgbm_interfaces import LGBMSubSplitInterface, LGBMHyperoptAlgInterface, \
    LGBMSklearnSubSplitInterface, RandomParamsLGBMAlgInterface
from pytabkit.bench.alg_wrappers.general import AlgWrapper
from pytabkit.bench.data.tasks import TaskPackage, TaskInfo
from pytabkit.bench.run.results import ResultManager
from pytabkit.models.alg_interfaces.other_interfaces import RFSubSplitInterface, SklearnMLPSubSplitInterface, \
    KANSubSplitInterface, GrandeSubSplitInterface, GBTSubSplitInterface, RandomParamsRFAlgInterface
from pytabkit.bench.scheduling.resources import NodeResources
from pytabkit.models.alg_interfaces.alg_interfaces import AlgInterface, MultiSplitWrapperAlgInterface
from pytabkit.models.alg_interfaces.base import SplitIdxs, RequiredResources
from pytabkit.models.alg_interfaces.rtdl_interfaces import RTDL_MLPSubSplitInterface, ResnetSubSplitInterface, \
    FTTransformerSubSplitInterface, RandomParamsResnetAlgInterface, RandomParamsRTDLMLPAlgInterface, \
    RandomParamsFTTransformerAlgInterface
from pytabkit.models.alg_interfaces.sub_split_interfaces import SingleSplitWrapperAlgInterface
from pytabkit.models.alg_interfaces.tabm_interface import TabMSubSplitInterface
from pytabkit.models.alg_interfaces.tabr_interface import TabRSubSplitInterface, \
    RandomParamsTabRAlgInterface
from pytabkit.models.alg_interfaces.nn_interfaces import NNAlgInterface, RandomParamsNNAlgInterface, NNHyperoptAlgInterface
from pytabkit.models.alg_interfaces.xgboost_interfaces import XGBSubSplitInterface, XGBHyperoptAlgInterface, \
    XGBSklearnSubSplitInterface, RandomParamsXGBAlgInterface
from pytabkit.models.data.data import TaskType, DictDataset
from pytabkit.models.nn_models.models import PreprocessingFactory
from pytabkit.models.training.logging import Logger
from pytabkit.models.training.metrics import Metrics


# what is the value of wrappers around AlgInterface?
#  - it has a create-function that can create multiple instances,
#    and can wrap with MultiSplitAlgInterface and SingleSplitAlgInterface
#  - there is some wrapping code in run(), but this could be moved to where the wrapper is used
#  - it provides get_max_n_vectorized()

# perhaps we should generalize TreeResourceComputation to also work for NNs?
# But this would require extra functionality for backprop, GPU RAM, etc.

def get_prep_factory(**config):
    return config.get('factory', None) or PreprocessingFactory(**config)


class AlgInterfaceWrapper(AlgWrapper):
    """
    Base class for wrapping AlgInterface classes for benchmarking.
    """
    def __init__(self, create_alg_interface_fn: Optional[Callable[[...], AlgInterface]], **config):
        """
        Constructor.

        :param create_alg_interface_fn: Function to create an AlgInterface via create_alg_interface_fn(**config).
        :param config: Configuration parameters.
        """
        super().__init__(**config)
        self.create_alg_interface_fn = create_alg_interface_fn

    # def _create_alg_interface_impl(self, n_cv: int, n_splits: int, task_type: TaskType) -> AlgInterface:
    def _create_alg_interface_impl(self, task_package: TaskPackage) -> AlgInterface:
        """
        Factory method to create an AlgInterface.
        Should be overridden unless ``create_alg_interface_fn`` has been provided in the constructor.
        This method should not be used directly, instead create_alg_interface() should be used.

        :param task_package: Task information.
        :return: An AlgInterface corresponding to an ML method.
        """
        if self.create_alg_interface_fn is not None:
            return self.create_alg_interface_fn(**self.config)
        else:
            raise NotImplementedError()

    def create_alg_interface(self, task_package: TaskPackage) -> AlgInterface:
        """
        Method to create an AlgInterface.

        :param task_package: Task information.
        :return: An AlgInterface corresponding to an ML method.
        """
        alg_interface = self._create_alg_interface_impl(task_package)
        if 'calibration_method' in self.config:
            try:
                from pytabkit.models.alg_interfaces.calibration import PostHocCalibrationAlgInterface
                alg_interface = PostHocCalibrationAlgInterface(alg_interface, **self.config)
            except ImportError:
                raise ValueError('Calibration methods are not implemented')
        if 'quantile_calib_alpha' in self.config:
            try:
                from pytabkit.models.alg_interfaces.custom_interfaces import QuantileCalibrationAlgInterface
                alg_interface = QuantileCalibrationAlgInterface(alg_interface, **self.config)
            except ImportError:
                raise ValueError('Quantile Calibration methods are not implemented')

        return alg_interface

    def run(self, task_package: TaskPackage, logger: Logger, assigned_resources: NodeResources,
            tmp_folders: List[Path]) -> List[ResultManager]:
        task = task_package.task_info.load_task(task_package.paths)
        task_desc = task_package.task_info.task_desc
        n_cv = task_package.n_cv
        n_refit = task_package.n_refit
        n_splits = len(task_package.split_infos)

        interface_resources = assigned_resources.get_interface_resources()

        old_torch_n_threads = torch.get_num_threads()
        torch.set_num_threads(interface_resources.n_threads)

        ds = task.ds
        name = 'alg ' + task_package.alg_name + ' on task ' + str(task_desc)

        # return_preds = self.config.get(f'save_y_pred', False)
        return_preds = task_package.save_y_pred
        metrics = Metrics.defaults(ds.tensor_infos['y'].cat_sizes,
                                   val_metric_name=self.config.get('val_metric_name', None))

        cv_idxs_list = []
        refit_idxs_list = []

        rms = [ResultManager() for split_info in task_package.split_infos]

        if len(rms) == 1:
            logger.log(1,
                       f'Running on split {task_package.split_infos[0].id} of task {task_package.task_info.task_desc}')
        else:
            logger.log(1, f'Running on {len(rms)} splits of task {task_package.task_info.task_desc}')

        for split_id, (rm, split_info) in enumerate(zip(rms, task_package.split_infos)):
            # this will usually be called with len(task_package.split_infos) == 1, but do a loop for safety
            test_split = split_info.splitter.split_ds(task.ds)
            trainval_idxs, test_idxs = test_split.idxs[0], test_split.idxs[1]
            trainval_ds = test_split.get_sub_ds(0)
            cv_sub_splits = split_info.get_sub_splits(trainval_ds, n_splits=n_cv, is_cv=True)
            cv_train_idxs = []
            cv_val_idxs = []
            for sub_idx, sub_split in enumerate(cv_sub_splits):
                cv_train_idxs.append(trainval_idxs[sub_split.idxs[0]])
                cv_val_idxs.append(trainval_idxs[sub_split.idxs[1]])
            cv_train_idxs = torch.stack(cv_train_idxs, dim=0)
            cv_val_idxs = torch.stack(cv_val_idxs, dim=0)
            cv_alg_seeds = [split_info.get_sub_seed(split_idx, is_cv=True) for split_idx in range(n_cv)]
            cv_idxs_list.append(SplitIdxs(cv_train_idxs, cv_val_idxs, test_idxs, split_seed=split_info.alg_seed,
                                          sub_split_seeds=cv_alg_seeds, split_id=split_id))

            if n_refit > 0:
                refit_train_idxs = torch.stack([trainval_idxs] * n_refit, dim=0)
                refit_alg_seeds = [split_info.get_sub_seed(split_idx, is_cv=False) for split_idx in range(n_refit)]
                refit_idxs_list.append(SplitIdxs(refit_train_idxs, None, test_idxs, split_seed=split_info.alg_seed,
                                                 sub_split_seeds=refit_alg_seeds, split_id=split_id))

        if task_package.rerun:
            for tmp_folder in tmp_folders:
                if utils.existsDir(tmp_folder):
                    # delete the folder such that the method doesn't load old results from the tmp folder
                    shutil.rmtree(tmp_folder)

        cv_tmp_folders = [tmp_folder / 'cv' for tmp_folder in tmp_folders]
        refit_tmp_folders = [tmp_folder / 'refit' for tmp_folder in tmp_folders]

        cv_alg_interface = self.create_alg_interface(task_package)
        cv_results_list = cv_alg_interface.fit_and_eval(ds, cv_idxs_list, interface_resources, logger, cv_tmp_folders,
                                                        name,
                                                        metrics, return_preds)
        for rm, cv_results in zip(rms, cv_results_list):
            rm.add_results(is_cv=True, results_dict=cv_results.get_dict())

        if n_refit > 0:
            refit_alg_interface = cv_alg_interface.get_refit_interface(n_refit)
            refit_results_list = refit_alg_interface.fit_and_eval(ds, refit_idxs_list, interface_resources, logger,
                                                                  refit_tmp_folders, name, metrics, return_preds)
            for rm, refit_results in zip(rms, refit_results_list):
                rm.add_results(is_cv=False, results_dict=refit_results.get_dict())

        torch.set_num_threads(old_torch_n_threads)

        return rms

    def get_required_resources(self, task_package: TaskPackage) -> RequiredResources:
        ds = DictDataset(tensors=None, tensor_infos=task_package.task_info.tensor_infos,
                         device='cpu', n_samples=task_package.task_info.n_samples)
        alg_interface = self.create_alg_interface(task_package)
        n_train, n_val = task_package.split_infos[0].get_train_and_val_size(n_samples=task_package.task_info.n_samples,
                                                                            n_splits=len(task_package.split_infos),
                                                                            is_cv=True)
        # n_train = split_info.get_sub_splits(trainval_ds, n_splits=n_cv, is_cv=True)
        return alg_interface.get_required_resources(ds=ds, n_cv=task_package.n_cv, n_refit=task_package.n_refit,
                                                    n_splits=len(task_package.split_infos),
                                                    split_seeds=[si.alg_seed for si in task_package.split_infos],
                                                    n_train=n_train)


class LoadResultsWrapper(AlgInterfaceWrapper):
    def __init__(self, alg_name: str, **config):
        super().__init__(create_alg_interface_fn=None, **config)
        self.alg_name = alg_name

    def _create_alg_interface_impl(self, task_package: TaskPackage) -> AlgInterface:
        assert len(task_package.split_infos) == 1  # only support single-split

        paths = self.config.get('paths', Paths.from_env_variables())
        task_info = task_package.task_info
        split_info = task_package.split_infos[0]
        split_id = split_info.id
        results_path = paths.results_alg_task_split(task_desc=task_info.task_desc, alg_name=self.alg_name,
                                                    n_cv=task_package.n_cv, split_type=split_info.split_type,
                                                    split_id=split_id)
        rm = ResultManager.load(results_path)
        y_preds_cv = torch.as_tensor(rm.other_dict['cv']['y_preds'], dtype=torch.float32)
        y_preds_refit = None if 'refit' not in rm.other_dict else torch.as_tensor(
            rm.other_dict['refit']['y_preds'], dtype=torch.float32)
        fit_params_cv = rm.other_dict['cv']['fit_params']
        fit_params_refit = None if 'refit' not in rm.other_dict else rm.other_dict['refit']['fit_params']
        return PrecomputedPredictionsAlgInterface(y_preds_cv=y_preds_cv, y_preds_refit=y_preds_refit,
                                                  fit_params_cv=fit_params_cv, fit_params_refit=fit_params_refit)

    def get_required_resources(self, task_package: TaskPackage) -> RequiredResources:
        # do this here such that we don't have to load the results for computing the required resources
        return RequiredResources(time_s=1e-5 * task_package.task_info.n_samples, cpu_ram_gb=1.5, n_threads=1)


class CaruanaEnsembleWrapper(AlgInterfaceWrapper):
    def __init__(self, sub_wrappers: List[AlgInterfaceWrapper], **config):
        super().__init__(create_alg_interface_fn=None, **config)
        self.sub_wrappers = sub_wrappers

    def _create_alg_interface_impl(self, task_package: TaskPackage) -> AlgInterface:
        single_split_alg_interfaces = []
        for split_info in task_package.split_infos:
            single_alg_interfaces = []
            for sub_wrapper in self.sub_wrappers:
                sub_tp = TaskPackage(task_info=task_package.task_info, split_infos=[split_info], n_cv=task_package.n_cv,
                                     n_refit=task_package.n_refit, paths=task_package.paths, rerun=task_package.rerun,
                                     alg_name=task_package.alg_name, save_y_pred=task_package.save_y_pred)
                single_alg_interfaces.append(sub_wrapper.create_alg_interface(sub_tp))
            single_split_alg_interfaces.append(CaruanaEnsembleAlgInterface(single_alg_interfaces, **self.config))
        return MultiSplitWrapperAlgInterface(single_split_alg_interfaces)

    def get_required_resources(self, task_package: TaskPackage) -> RequiredResources:
        single_resources = [sub_wrapper.get_required_resources(task_package)
                            for sub_wrapper in self.sub_wrappers]
        return RequiredResources.combine_sequential(single_resources)


class AlgorithmSelectionWrapper(AlgInterfaceWrapper):
    def __init__(self, sub_wrappers: List[AlgInterfaceWrapper], **config):
        super().__init__(create_alg_interface_fn=None, **config)
        self.sub_wrappers = sub_wrappers

    def _create_alg_interface_impl(self, task_package: TaskPackage) -> AlgInterface:
        single_split_alg_interfaces = []
        for split_info in task_package.split_infos:
            single_alg_interfaces = []
            for sub_wrapper in self.sub_wrappers:
                sub_tp = TaskPackage(task_info=task_package.task_info, split_infos=[split_info], n_cv=task_package.n_cv,
                                     n_refit=task_package.n_refit, paths=task_package.paths, rerun=task_package.rerun,
                                     alg_name=task_package.alg_name, save_y_pred=task_package.save_y_pred)
                single_alg_interfaces.append(sub_wrapper.create_alg_interface(sub_tp))
            single_split_alg_interfaces.append(AlgorithmSelectionAlgInterface(single_alg_interfaces, **self.config))
        return MultiSplitWrapperAlgInterface(single_split_alg_interfaces)

    def get_required_resources(self, task_package: TaskPackage) -> RequiredResources:
        # too pessimistic for refit...
        single_resources = [sub_wrapper.get_required_resources(task_package)
                            for sub_wrapper in self.sub_wrappers]
        return RequiredResources.combine_sequential(single_resources)


class MultiSplitAlgInterfaceWrapper(AlgInterfaceWrapper):
    def __init__(self, **config):
        super().__init__(create_alg_interface_fn=None, **config)

    def create_single_alg_interface(self, n_cv: int, task_type: TaskType) \
            -> AlgInterface:
        raise NotImplementedError()

    def _create_alg_interface_impl(self, task_package: TaskPackage) -> AlgInterface:
        n_cv = task_package.n_cv
        task_type = task_package.task_info.task_type
        n_splits = len(task_package.split_infos)
        return MultiSplitWrapperAlgInterface(
            single_split_interfaces=[self.create_single_alg_interface(n_cv, task_type)
                                     for i in range(n_splits)])


class SubSplitInterfaceWrapper(MultiSplitAlgInterfaceWrapper):
    def __init__(self, create_sub_split_learner_fn: Optional[Callable[[...], AlgInterface]] = None, **config):
        super().__init__(**config)
        self.create_sub_split_learner_fn = create_sub_split_learner_fn

    def create_sub_split_interface(self, task_type: TaskType) -> AlgInterface:
        if self.create_sub_split_learner_fn is not None:
            return self.create_sub_split_learner_fn(**self.config)
        raise NotImplementedError()

    def create_single_alg_interface(self, n_cv: int, task_type: TaskType) \
            -> AlgInterface:
        return SingleSplitWrapperAlgInterface([self.create_sub_split_interface(task_type)
                                               for i in range(n_cv)])


class NNInterfaceWrapper(AlgInterfaceWrapper):
    def __init__(self, **config):
        super().__init__(NNAlgInterface, **config)

    def get_max_n_vectorized(self, task_info: TaskInfo) -> int:
        ds = DictDataset(tensors=None, tensor_infos=task_info.tensor_infos, device='cpu',
                         n_samples=task_info.n_samples)
        max_ram_gb = 8.0
        max_n_vectorized = self.config.get('max_n_vectorized', 50)
        alg_interface = NNAlgInterface(**self.config)
        while max_n_vectorized > 1:
            required_resources = alg_interface.get_required_resources(ds, n_cv=1, n_refit=0, n_splits=max_n_vectorized,
                                                                      split_seeds=[0] * max_n_vectorized,
                                                                      n_train=task_info.n_samples)
            if required_resources.gpu_ram_gb <= max_ram_gb and required_resources.cpu_ram_gb <= max_ram_gb:
                return max_n_vectorized
            max_n_vectorized -= 1

        return 1


class NNHyperoptInterfaceWrapper(AlgInterfaceWrapper):
    def __init__(self, **config):
        super().__init__(NNHyperoptAlgInterface, **config)

    def get_max_n_vectorized(self, task_info: TaskInfo) -> int:
        ds = DictDataset(tensors=None, tensor_infos=task_info.tensor_infos, device='cpu',
                         n_samples=task_info.n_samples)
        max_ram_gb = 8.0
        max_n_vectorized = self.config.get('max_n_vectorized', 50)
        alg_interface = NNHyperoptAlgInterface(**self.config)
        while max_n_vectorized > 1:
            required_resources = alg_interface.get_required_resources(ds, n_cv=1, n_refit=0, n_splits=max_n_vectorized,
                                                                      split_seeds=[0] * max_n_vectorized,
                                                                      n_train=task_info.n_samples)
            if required_resources.gpu_ram_gb <= max_ram_gb and required_resources.cpu_ram_gb <= max_ram_gb:
                return max_n_vectorized
            max_n_vectorized -= 1

        return 1


class RandomParamsNNInterfaceWrapper(AlgInterfaceWrapper):
    def __init__(self, model_idx: int, **config):
        # model_idx should be the random search iteration (i.e. start from zero)
        super().__init__(RandomParamsNNAlgInterface, model_idx=model_idx, **config)


class LGBMSklearnInterfaceWrapper(SubSplitInterfaceWrapper):
    def create_sub_split_interface(self, task_type: TaskType):
        return LGBMSklearnSubSplitInterface(**self.config)


class LGBMInterfaceWrapper(SubSplitInterfaceWrapper):
    def create_sub_split_interface(self, task_type: TaskType) -> AlgInterface:
        return LGBMSubSplitInterface(**self.config)


class LGBMHyperoptInterfaceWrapper(MultiSplitAlgInterfaceWrapper):
    def create_single_alg_interface(self, n_cv: int, task_type: TaskType) \
            -> AlgInterface:
        return LGBMHyperoptAlgInterface(**self.config)


class RandomParamsLGBMInterfaceWrapper(MultiSplitAlgInterfaceWrapper):
    def create_single_alg_interface(self, n_cv: int, task_type: TaskType) \
            -> AlgInterface:
        return RandomParamsLGBMAlgInterface(**self.config)


class XGBSklearnInterfaceWrapper(SubSplitInterfaceWrapper):
    def create_sub_split_interface(self, task_type: TaskType) -> AlgInterface:
        return XGBSklearnSubSplitInterface(**self.config)


class XGBInterfaceWrapper(SubSplitInterfaceWrapper):
    def create_sub_split_interface(self, task_type: TaskType) -> AlgInterface:
        return XGBSubSplitInterface(**self.config)


class RandomParamsXGBInterfaceWrapper(MultiSplitAlgInterfaceWrapper):
    def create_single_alg_interface(self, n_cv: int, task_type: TaskType) \
            -> AlgInterface:
        return RandomParamsXGBAlgInterface(**self.config)


class XGBHyperoptInterfaceWrapper(MultiSplitAlgInterfaceWrapper):
    def create_single_alg_interface(self, n_cv: int, task_type: TaskType) \
            -> AlgInterface:
        return XGBHyperoptAlgInterface(**self.config)


class CatBoostSklearnInterfaceWrapper(SubSplitInterfaceWrapper):
    def create_sub_split_interface(self, task_type: TaskType) -> AlgInterface:
        return CatBoostSklearnSubSplitInterface(**self.config)


class CatBoostInterfaceWrapper(SubSplitInterfaceWrapper):
    def create_sub_split_interface(self, task_type: TaskType) -> AlgInterface:
        return CatBoostSubSplitInterface(**self.config)


class CatBoostHyperoptInterfaceWrapper(MultiSplitAlgInterfaceWrapper):
    def create_single_alg_interface(self, n_cv: int, task_type: TaskType) \
            -> AlgInterface:
        return CatBoostHyperoptAlgInterface(**self.config)


class RandomParamsCatBoostInterfaceWrapper(MultiSplitAlgInterfaceWrapper):
    def create_single_alg_interface(self, n_cv: int, task_type: TaskType) \
            -> AlgInterface:
        return RandomParamsCatBoostAlgInterface(**self.config)


class RFInterfaceWrapper(SubSplitInterfaceWrapper):
    def create_sub_split_interface(self, task_type: TaskType) -> AlgInterface:
        return RFSubSplitInterface(**self.config)


class GBTInterfaceWrapper(SubSplitInterfaceWrapper):
    def create_sub_split_interface(self, task_type: TaskType) -> AlgInterface:
        return GBTSubSplitInterface(**self.config)


class SklearnMLPInterfaceWrapper(SubSplitInterfaceWrapper):
    def create_sub_split_interface(self, task_type: TaskType) -> AlgInterface:
        return SklearnMLPSubSplitInterface(**self.config)


class KANInterfaceWrapper(SubSplitInterfaceWrapper):
    def create_sub_split_interface(self, task_type: TaskType) -> AlgInterface:
        return KANSubSplitInterface(**self.config)


class GrandeInterfaceWrapper(SubSplitInterfaceWrapper):
    def create_sub_split_interface(self, task_type: TaskType) -> AlgInterface:
        return GrandeSubSplitInterface(**self.config)


class MLPRTDLInterfaceWrapper(SubSplitInterfaceWrapper):
    def create_sub_split_interface(self, task_type: TaskType) -> AlgInterface:
        return RTDL_MLPSubSplitInterface(**self.config)


class ResNetRTDLInterfaceWrapper(SubSplitInterfaceWrapper):
    def create_sub_split_interface(self, task_type: TaskType) -> AlgInterface:
        return ResnetSubSplitInterface(**self.config)


class FTTransformerInterfaceWrapper(SubSplitInterfaceWrapper):
    def create_sub_split_interface(self, task_type: TaskType) -> AlgInterface:
        return FTTransformerSubSplitInterface(**self.config)


class TabRInterfaceWrapper(SubSplitInterfaceWrapper):
    def create_sub_split_interface(self, task_type: TaskType) -> AlgInterface:
        return TabRSubSplitInterface(**self.config)


class TabMInterfaceWrapper(SubSplitInterfaceWrapper):
    def create_sub_split_interface(self, task_type: TaskType) -> AlgInterface:
        return TabMSubSplitInterface(**self.config)


class RandomParamsResnetInterfaceWrapper(AlgInterfaceWrapper):
    def __init__(self, model_idx: int, **config):
        # model_idx should be the random search iteration (i.e. start from zero)
        super().__init__(RandomParamsResnetAlgInterface, model_idx=model_idx, **config)


class RandomParamsRTDLMLPInterfaceWrapper(AlgInterfaceWrapper):
    def __init__(self, model_idx: int, **config):
        # model_idx should be the random search iteration (i.e. start from zero)
        super().__init__(RandomParamsRTDLMLPAlgInterface, model_idx=model_idx, **config)


class RandomParamsFTTransformerInterfaceWrapper(AlgInterfaceWrapper):
    def __init__(self, model_idx: int, **config):
        # model_idx should be the random search iteration (i.e. start from zero)
        super().__init__(RandomParamsFTTransformerAlgInterface, model_idx=model_idx, **config)


class AutoGluonModelInterfaceWrapper(AlgInterfaceWrapper):
    def __init__(self, **config):
        # model_idx should be the random search iteration (i.e. start from zero)
        super().__init__(AutoGluonModelAlgInterface, **config)


class RandomParamsTabRInterfaceWrapper(SubSplitInterfaceWrapper):
    def create_single_alg_interface(self, n_cv: int, task_type: TaskType) \
            -> AlgInterface:
        return RandomParamsTabRAlgInterface(**self.config)


class RandomParamsRFInterfaceWrapper(AlgInterfaceWrapper):
    def __init__(self, model_idx: int, **config):
        # model_idx should be the random search iteration (i.e. start from zero)
        super().__init__(RandomParamsRFAlgInterface, model_idx=model_idx, **config)
