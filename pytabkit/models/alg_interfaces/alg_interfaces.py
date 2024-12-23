import functools
from pathlib import Path
from typing import List, Tuple, Any, Optional, Dict

import torch

from pytabkit.models.alg_interfaces.base import SplitIdxs, InterfaceResources, RequiredResources
from pytabkit.models.data.nested_dict import NestedDict
from pytabkit.models.hyper_opt.hyper_optimizers import HyperOptimizer

from pytabkit.models import utils
from pytabkit.models.data.data import DictDataset, TaskType
from pytabkit.models.torch_utils import cat_if_necessary
from pytabkit.models.training.logging import Logger
from pytabkit.models.training.metrics import Metrics


class AlgInterface:
    """
    AlgInterface is an abstract base class for tabular ML methods
    with an interfaces that offers more possibilities than a standard scikit-learn interface.

    In particular, it allows for parallelized fitting of multiple models, bagging, and refitting.
    The idea is as follows:

    - The dataset can be split into a test set and the remaining data. (We call this a trainval-test split.)
        The fit() method allows to specify multiple such splits,
        and some AlgInterface implementations (NNAlgInterface) allow to vectorize computations across these splits.
        However, for vectorization, we may require that the test set sizes are identical in all splits.
    - The remaining data can further be split into training and validation data. (We call this a train-val split.)
        AlgInterface allows to fit with one or multiple train-val splits, which can also be vectorized in NNAlgInterface.
        Optionally, the function `get_refit_interface()` allows to extract an AlgInterface that can be used for
        fitting the model on training+validation set
        with the best settings found on the validation set in the cross-validation stage (represented by self.fit_params).
        These "best settings" could be an early stopping epoch or number of trees,
        or best hyperparameters found by hyperparameter optimization.
        We call this refitting.

    Another feature of AlgInterface is that it provides methods to get (an estimate of) required resources
    and to evaluate metrics on training, validation, and test set.
    """

    def __init__(self, fit_params: Optional[List[Dict[str, Any]]] = None, **config):
        """
        :param fit_params: This parameter can be used to store the best hyperparameters
            found during fit() in (cross-)validation mode. These can then be used for fit() in refitting mode.
            If fit_params is not None, it should be a list with one dictionary per trainval-test split.
            The dictionaries then contain the obtained hyperparameters for each of the trainval-test splits.
            Normally, there are no best parameters per train-val split
            as we might not have the same number of refitted models as train-val splits.
        :param config: Other parameters.
        """
        self.config = config
        self.fit_params = fit_params

    def fit(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources,
            logger: Logger, tmp_folders: List[Optional[Path]], name: str) -> Optional[
        List[List[List[Tuple[Dict, float]]]]]:
        """
        Fit the models on the given data and splits.
        Should be overridden by subclasses unless fit_and_eval() is overloaded.
        In the latter case, this method will by default use fit_and_eval() and discard the evaluation.

        :param ds: DictDataset representing the dataset. Should be on the CPU.
        :param idxs_list: List containing one SplitIdxs object per trainval-test split. Indices should be on the CPU.
        :param interface_resources: Resources assigned to fit().
        :param logger: Logger that can be used for logging.
        :param tmp_folders: List of paths that can be used for storing intermediate data.
            The paths can be None, in which case methods will try not to save intermediate results.
            There should be one folder per trainval-test-split (i.e. only one per k-fold CV).
        :param name: Name of the algorithm (for logging).
        :return: May return information about different possible fit_params settings that can be used.
            Say a variable `results` is returned that is not None.
            Then, results[tt_split_idx][tv_split_idx] should be a list of tuples (params, loss).
            This is useful for k-fold cross-validation,
            where the params with the best average loss (averaged over tv_split_idx) can be selected for fit_params.
        """
        if self.__class__.fit_and_eval == AlgInterface.fit_and_eval:
            raise NotImplementedError()  # avoid infinite recursion
        else:
            self.fit_and_eval(ds, idxs_list, interface_resources, logger, tmp_folders, name, metrics=None,
                              return_preds=False)
        return None

    def fit_and_eval(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources,
                     logger: Logger, tmp_folders: List[Optional[Path]], name: str, metrics: Optional[Metrics],
                     return_preds: bool) -> List[NestedDict]:
        """
        Run fit() with the given parameters and then return the result of eval() with the given metrics.
        This method can be overridden instead of fit() if it is more convenient.
        The idea is that for hyperparameter optimization,
        one has to evaluate each hyperparameter combination anyway after training it,
        so it is more efficient to implement fit_and_eval() and return the evaluation of the best method at the end.
        See the documentation of fit() and eval() for the meaning of the parameters and returned values.
        """
        if self.__class__.fit == AlgInterface.fit:
            raise NotImplementedError()  # avoid infinite recursion
        self.fit(ds=ds, idxs_list=idxs_list, interface_resources=interface_resources,
                 logger=logger, tmp_folders=tmp_folders, name=name)
        return self.eval(ds=ds, idxs_list=idxs_list, metrics=metrics, return_preds=return_preds)

    def eval(self, ds: DictDataset, idxs_list: List[SplitIdxs], metrics: Optional[Metrics],
             return_preds: bool) -> List[NestedDict]:
        """
        Evaluates the (already fitted) method using various metrics on training, validation, and test sets.
        The results will also contain the found fit_params and optionally the predictions on the dataset.
        This method should normally not be overridden in subclasses.

        :param ds: Dataset.
        :param idxs_list: List of indices for the training-validation-test splits,
            one per trainval-test split as in fit().
        :param metrics: Metrics object that defines which metrics should be evaluated.
            If metrics is None, an empty list will be returned
            (which might avoid unnecessary computation when implementing fit() through fit_and_eval()).
        :param return_preds: Whether the predictions on the dataset should be included in the returned results.
        :return: Returns a list with one NestedDict for every trainval-test split.
            Denote by `results` such a NestedDict object. Then, `results` will contain the following contents:
            results['metrics', 'train'/'val'/'test', str(n_models), str(start_idx), metric_name] = metric_value
            Here, an ensemble of the predictions of models [start_idx:start_idx+n_models] will be used.
            results['y_preds'] = a list (converted from a tensor) with predictions on the whole dataset,
            included only if return_preds==True.
            results['fit_params'] = self.fit_params
        """

        if metrics is None:
            results = []
            # for idxs in idxs_list:
            #     result = NestedDict()
            #     for split_name in ['train', 'val', 'test']:
            #         result['metrics'][split_name]['1']['0'] = dict()
            #     if return_preds:
            #         pass
            #     results.append(dict(metrics))
            return results
        X, y = ds.split_xy()
        y = y.tensors['y']
        y_pred_full = self.predict(X).detach().cpu()
        # print(f'{y=}')
        # print(f'{y_pred_full=}')
        # print(f'{y.shape=}')
        # print(f'{y_pred_full.shape=}')
        idx = 0
        results_list = []
        for split_idx, idxs in enumerate(idxs_list):
            results = NestedDict()

            y_preds = y_pred_full[idx:idx + idxs.n_trainval_splits]
            if return_preds:
                results['y_preds'] = y_preds.numpy().tolist()
            idx += idxs.n_trainval_splits

            if idxs.test_idxs is not None:
                # print(f'{y_preds.shape=}')
                # print(f'{y.shape=}')
                results['metrics', 'test'] = metrics.compute_metrics_dict(
                    y_preds=[y_preds[i, idxs.test_idxs] for i in range(y_preds.shape[0])],
                    y=y[idxs.test_idxs],
                    use_ens=True)
            train_metrics = NestedDict()
            val_metrics = NestedDict()
            for i in range(idxs.n_trainval_splits):
                train_dict = metrics.compute_metrics_dict([y_preds[i, idxs.train_idxs[i]]], y[idxs.train_idxs[i]],
                                                          use_ens=False)
                train_metrics['1', str(i)] = train_dict['1', '0']

                if idxs.val_idxs is not None and idxs.val_idxs.shape[-1] > 0:
                    val_dict = metrics.compute_metrics_dict([y_preds[i, idxs.val_idxs[i]]], y[idxs.val_idxs[i]],
                                                            use_ens=False)
                    val_metrics['1', str(i)] = val_dict['1', '0']

            results['metrics', 'train'] = train_metrics
            if idxs.val_idxs is not None:
                results['metrics', 'val'] = val_metrics
            if self.fit_params is not None:
                results['fit_params'] = self.fit_params[split_idx]
            results_list.append(results)

        return results_list

    def predict(self, ds: DictDataset) -> torch.Tensor:
        """
        Method to predict labels on the given dataset. Override in subclasses.

        :param ds: Dataset on which to predict labels
        :return: Returns a tensor of shape [n_trainval_splits * n_splits, ds.n_samples, output_shape]
            In the classification case, output_shape will be the number of classes (even in the binary case)
            and the outputs will be logits (i.e., softmax should be applied to get probabilities)
            In the regression case, output_shape will be the target dimension (often 1).
        """
        raise NotImplementedError()

    def get_refit_interface(self, n_refit: int, fit_params: Optional[List[Dict]] = None) -> 'AlgInterface':
        """
        Returns another AlgInterface that is configured for refitting on the training and validation data.
        Override in subclasses.

        :param n_refit: Number of models that should be refitted (with different seeds) per trainval-test split.
        :param fit_params: Fit parameters (see the constructor) that should be used for refitting.
            If fit_params is None, self.fit_params will be used instead.
        :return: Returns the AlgInterface object for refitting.
        """
        raise NotImplementedError()

    def get_fit_params(self) -> Optional[List[Dict]]:
        """
        :return: Return self.fit_params.
        """
        return self.fit_params

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        """
        Estimate the required resources for fit().

        :param ds: Dataset. Does not have to contain tensors.
        :param n_cv: Number of train-val splits per trainval-test split.
        :param n_refit: Number of refitted models per trainval-test split.
        :param n_splits: Number of trainval-test splits.
        :param split_seeds: Seeds for every trainval-test split.
        :return: Returns estimated required resources.
        """
        raise NotImplementedError()


class MultiSplitWrapperAlgInterface(AlgInterface):
    # todo: do we need the option to run this with a "split batch size" > 1 for the NNInterface?
    def __init__(self, single_split_interfaces: List[AlgInterface]):
        super().__init__(single_split_interfaces=single_split_interfaces)
        # todo: could allow parallel evaluation, but not for now
        self.single_split_interfaces = single_split_interfaces

    def get_refit_interface(self, n_refit: int, fit_params: Optional[List[Dict]] = None) -> 'AlgInterface':
        # return interface with the hyperparameters found by cross-validation for refitting
        # this can only be called if some fit method has been called before with validation data
        fit_params = fit_params or self.fit_params
        if fit_params is not None:
            assert len(fit_params) == len(self.single_split_interfaces)
            fit_params_list = [[p] for p in fit_params]
        else:
            fit_params_list = [None] * len(self.single_split_interfaces)
        return MultiSplitWrapperAlgInterface([s.get_refit_interface(n_refit, p)
                                              for p, s in zip(fit_params_list, self.single_split_interfaces)])

    def fit(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources,
            logger: Logger, tmp_folders: List[Optional[Path]], name: str) -> Optional[
        List[List[List[Tuple[Dict, float]]]]]:
        assert len(self.single_split_interfaces) == len(idxs_list)
        assert len(idxs_list) == len(tmp_folders)

        for split_idx in range(len(idxs_list)):
            self.single_split_interfaces[split_idx].fit(ds, [idxs_list[split_idx]], interface_resources, logger,
                                                        [tmp_folders[split_idx]], name)
        self.fit_params = [ssi.fit_params[0] for ssi in self.single_split_interfaces]
        return None

    def fit_and_eval(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources,
                     logger: Logger, tmp_folders: List[Optional[Path]], name: str, metrics: Optional[Metrics],
                     return_preds: bool) -> List[NestedDict]:
        assert len(self.single_split_interfaces) == len(idxs_list)
        assert len(idxs_list) == len(tmp_folders)

        results_list = []

        for split_idx in range(len(idxs_list)):
            results_list.extend(self.single_split_interfaces[split_idx].fit_and_eval(
                ds, [idxs_list[split_idx]], interface_resources, logger,
                [tmp_folders[split_idx]], name, metrics, return_preds))

        return results_list

    def predict(self, ds: DictDataset) -> torch.Tensor:
        return cat_if_necessary([s.predict(ds) for s in self.single_split_interfaces], dim=0)

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        single_resources = [
            ssi.get_required_resources(ds, n_cv, n_refit, n_splits=1, split_seeds=[split_seeds[i]], n_train=n_train)
            for i, ssi in enumerate(self.single_split_interfaces)]
        return RequiredResources.combine_sequential(single_resources)


class SingleSplitAlgInterface(AlgInterface):
    pass  # this class is just to document that the fit() and fit_and_eval() functions can only take one split


class OptAlgInterface(SingleSplitAlgInterface):
    def __init__(self, hyper_optimizer: HyperOptimizer, max_resource_config: Dict, **config):
        super().__init__(**config)
        # self.create_alg_interface = create_alg_interface
        self.hyper_optimizer = hyper_optimizer

        # a configuration that can be passed to self.create_alg_interface()
        # which should be used for resource estimation.
        # E.g. for tree-based methods this should involve the maximum depth and maximum n_estimators
        # that can be used during HPO.
        self.max_resource_config = max_resource_config

        # self.fit_params['hyper_fit_params'] will contain the optimized parameters,
        # self.fit_params['sub_fit_params'] will contain the fit_params of the best fitted alg_interface
        self.best_alg_interface = None
        self.opt_step = 0

        # list where all results from all optimization steps can be stored (except y_preds, to save memory)
        # this list will then be included into the final results, such that one can retrospectively simulate
        # what would have happened if the optimization had been terminated earlier
        self.results_list = []

    def create_alg_interface(self, n_sub_splits: int, **config) -> AlgInterface:
        raise NotImplementedError()

    def get_refit_interface(self, n_refit: int, fit_params: Optional[List[Dict]] = None) -> 'AlgInterface':
        if fit_params is not None:
            assert len(fit_params) == 1  # single split
        else:
            assert self.fit_params is not None
            fit_params = self.fit_params
        alg_interface = self.create_alg_interface(n_refit,
                                                  **utils.join_dicts(self.config, fit_params[0]['hyper_fit_params']))
        # the alg_interface itself may have other hypers that have been fit
        return alg_interface.get_refit_interface(n_refit, fit_params[0]['sub_fit_params'])

    def objective(self, params, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources,
                  logger: Logger, tmp_folder: Optional[Path], name: str, metrics: Optional[Metrics],
                  return_preds: bool) -> Tuple[float, Tuple[List[NestedDict], AlgInterface]]:
        self.opt_step += 1
        tmp_folder = tmp_folder / f'step_{self.opt_step}' if tmp_folder is not None else None

        could_load = False

        # try to load results
        if tmp_folder is not None and utils.existsFile(tmp_folder / 'DONE'):
            # should be able to load the results
            alg_interface = utils.deserialize(tmp_folder / 'alg_interface.pkl', compressed=True)
            results = utils.deserialize(tmp_folder / 'results.pkl')
            sub_fit_params = utils.deserialize(tmp_folder / 'fit_params.pkl')
            loaded_params = utils.deserialize(tmp_folder / 'params.pkl')

            if loaded_params != params:
                print('Got different params than the saved ones, '
                      'hyperparameter optimizer might be non-deterministic')
                print(f'{params=}')
                print(f'{loaded_params=}', flush=True)
                # logger.log(1, 'Got different params than the saved ones, '
                #               'hyperparameter optimizer might be non-deterministic')
                # don't set could_load to true, recompute
                utils.delete_file(tmp_folder / 'DONE')
            else:
                could_load = True

        if not could_load:
            # compute results
            tmp_folders = [tmp_folder / 'alg_interface' if tmp_folder is not None else None]
            alg_interface = self.create_alg_interface(idxs_list[0].n_trainval_splits,
                                                      **utils.join_dicts(self.config, params))
            results = alg_interface.fit_and_eval(ds=ds, idxs_list=idxs_list, interface_resources=interface_resources,
                                                 logger=logger, tmp_folders=tmp_folders, name=name, metrics=metrics,
                                                 return_preds=return_preds)
            sub_fit_params = alg_interface.get_fit_params()

            # save results
            if tmp_folder is not None:
                utils.serialize(tmp_folder / 'alg_interface.pkl', alg_interface, compressed=True)
                utils.serialize(tmp_folder / 'results.pkl', results)
                # serialize fit_params separately in case the alg_interface cannot be loaded
                utils.serialize(tmp_folder / 'fit_params.pkl', sub_fit_params)
                utils.serialize(tmp_folder / 'params.pkl', params)

                # save the "DONE" file last to indicate that all other files have been completely written
                utils.writeToFile(tmp_folder / 'DONE', '')

        # todo: could do sub_fit_params[0] instead since it's only one split anyway?
        results[0]['fit_params'] = {'hyper_fit_params': params, 'sub_fit_params': sub_fit_params}

        # store all parameters and results (metrics) without predictions
        self.results_list.append(utils.update_dict(results[0].get_dict(), remove_keys=['y_preds']))

        val_loss = metrics.compute_val_score(results[0]['metrics']['val'])
        return val_loss, (results, alg_interface)

    def fit_and_eval(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources,
                     logger: Logger, tmp_folders: List[Optional[Path]], name: str, metrics: Optional[Metrics],
                     return_preds: bool) -> List[NestedDict]:
        assert len(idxs_list) == 1  # this is a SingleSplitAlgInterface
        assert len(tmp_folders) == 1  # this is a SingleSplitAlgInterface
        split_idxs = idxs_list[0]
        tmp_folder = tmp_folders[0]
        opt_desc = f'split {split_idxs.split_id} of {name}'

        if metrics is None:
            # create metrics because we need to have a validation score
            task_type = TaskType.CLASSIFICATION if ds.tensor_infos['y'].is_cat() else TaskType.REGRESSION
            val_metric_name = self.config.get('val_metric_name', Metrics.default_val_metric_name(task_type))
            metrics = Metrics(metric_names=[val_metric_name], val_metric_name=val_metric_name, task_type=task_type)

        self.opt_step = 0
        f = functools.partial(self.objective, ds=ds, idxs_list=idxs_list, interface_resources=interface_resources,
                              logger=logger, tmp_folder=tmp_folder, name=name, metrics=metrics,
                              return_preds=return_preds)
        hyper_fit_params, (results, best_alg_interface) = self.hyper_optimizer.optimize(
            f=f, seed=split_idxs.sub_split_seeds[0], opt_desc=opt_desc, logger=logger)
        self.best_alg_interface = best_alg_interface
        self.fit_params = [results[0]['fit_params']]
        results[0]['opt_step_results'] = self.results_list
        return results

    def predict(self, ds: DictDataset) -> torch.Tensor:
        return self.best_alg_interface.predict(ds)

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        ref_alg_interface = self.create_alg_interface(n_sub_splits=1, **self.max_resource_config)
        single_resources = ref_alg_interface.get_required_resources(ds, n_cv=1, n_refit=0, n_splits=1,
                                                                    split_seeds=split_seeds, n_train=n_train)
        single_resources.time_s *= (self.hyper_optimizer.get_n_hyperopt_steps() * n_cv + n_refit) * n_splits
        return single_resources


class RandomParamsAlgInterface(SingleSplitAlgInterface):
    def __init__(self, model_idx: int, fit_params: Optional[List[Dict[str, Any]]] = None, **config):
        """
        :param model_idx: used for seeding along with the seed given in fit(), so we can do random search HPO
            by combining multiple RandomParamsNNAlgInterface objects with different model_idx values-
        :param fit_params: Fit parameters (stopping epoch for refitting).
        :param config: Configuration parameters.
        """
        super().__init__(fit_params=fit_params, **config)
        self.model_idx = model_idx
        self.alg_interface = None

    def _sample_params(self, is_classification: bool, seed: int, n_train: int):
        raise NotImplementedError()  # override in subclass

    def _create_interface_from_config(self, n_tv_splits: int, **config):
        raise NotImplementedError()

    def get_refit_interface(self, n_refit: int, fit_params: Optional[List[Dict]] = None) -> 'AlgInterface':
        raise NotImplementedError('Refit is not fully implemented...')
        # return RandomParamsNNAlgInterface(model_idx=self.model_idx, fit_params=fit_params or self.fit_params,
        #                                   **self.config)

    def _create_sub_interface(self, ds: DictDataset, seed: int, n_train: int, n_tv_splits: int):
        # this is also set in get_required_resources, but okay
        if self.fit_params is None:
            hparam_seed = utils.combine_seeds(seed, self.model_idx)
            is_classification = not ds.tensor_infos['y'].is_cont()
            self.fit_params = [self._sample_params(is_classification, hparam_seed, n_train)]
        # todo: need epoch for refit
        return self._create_interface_from_config(n_tv_splits=n_tv_splits, fit_params=None,
                                                  **utils.update_dict(self.config, self.fit_params[0]))

    def fit(self, ds: DictDataset, idxs_list: List[SplitIdxs], interface_resources: InterfaceResources,
            logger: Logger, tmp_folders: List[Optional[Path]], name: str) -> None:
        assert len(idxs_list) == 1
        n_tv_splits = idxs_list[0].n_trainval_splits
        self.alg_interface = self._create_sub_interface(ds, idxs_list[0].split_seed, n_train=idxs_list[0].n_train,
                                                        n_tv_splits=n_tv_splits)
        self.alg_interface.fit(ds, idxs_list, interface_resources, logger, tmp_folders, name)

    def predict(self, ds: DictDataset) -> torch.Tensor:
        return self.alg_interface.predict(ds)

    def get_required_resources(self, ds: DictDataset, n_cv: int, n_refit: int, n_splits: int,
                               split_seeds: List[int], n_train: int) -> RequiredResources:
        assert len(split_seeds) == 1
        alg_interface = self._create_sub_interface(ds, split_seeds[0], n_train=n_train, n_tv_splits=n_cv, )
        return alg_interface.get_required_resources(ds, n_cv, n_refit, n_splits, split_seeds, n_train=n_train)
