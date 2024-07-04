from pathlib import Path
from typing import List

from pytabkit.bench.data.tasks import TaskPackage, TaskInfo
from pytabkit.bench.run.results import ResultManager
from pytabkit.models.training.logging import Logger

from pytabkit.bench.scheduling.resources import NodeResources
from pytabkit.models.alg_interfaces.base import RequiredResources


class AlgWrapper:
    """
    Base class for ML methods that can be run in the benchmarking code.
    """
    def __init__(self, **config):
        """
        Constructor.

        :param config: Configuration parameters.
        """
        self.config = config

    def run(self, task_package: TaskPackage, logger: Logger, assigned_resources: NodeResources,
            tmp_folders: List[Path]) -> List[ResultManager]:
        """
        Run the ML method on the given task. Should be overridden in subclasses.

        :param task_package: Information about the task to be run.
        :param logger: Logger.
        :param assigned_resources: Assigned resources (e.g. number of threads).
        :param tmp_folders: Temporary folders, one for each train/test split, to save temporary data to.
        :return: A List of ResultManager objects, one for each train/test split, that contain the results of the run.
        """
        raise NotImplementedError()

    def get_required_resources(self, task_package: TaskPackage) -> RequiredResources:
        """
        Should be overridden in subclasses.

        :param task_package: Information about the task that should be executed.
        :return: Information about the estimated required resources that will be needed to run this task.
        """
        raise NotImplementedError()

    def get_max_n_vectorized(self, task_info: TaskInfo) -> int:
        """
        Returns 1 by default, should be overridden in subclasses if they benefit from vectorization.

        :param task_info: Information about the task that this method should run on.
        :return: Maximum number of train/test splits that this method can be run on at once.
        """
        return 1


# want to have:
# - more general / easy ResourceComputation
# - generic thread-allocation parameters for such a ResourceComputation
#   that allow to allocate more threads for larger workloads
# - better NodeResources class that supports mps or perhaps a new class that summarizes the allocated resources
# - should the resource estimation be moved to AlgInterface?
#   Then, we would need to instantiate an AlgInterface in the wrapper to do the estimation
# - maybe a code that estimates RAM (and time) constants? With fake data sets?

# better ResourceComputation:
# have identical components for CPU and GPU, and maybe also for RAM and time
# components:
# - dataset size
# - factory (model) size
# - RAM for forward (and backward) pass
# - generic calculation (constant, per-tree, per-class, per-sample),
# for the NN we might also need to include the batch size, number of epochs, etc.
# what about the number of threads etc.?
# want to have one per device?

# better NodeResources:
# maybe just have a dict with the devices that are being referred to by the array?




