from typing import Optional, List

import numpy as np
import copy

from pytabkit.models.alg_interfaces.base import InterfaceResources, RequiredResources


# already add fixed GPU RAM in assigned resources?  (problem: does try_assign know these fixed resources?)
# or have fixed_resources: NodeResources that are added each time?
# problem: fixed resources only need to be added to those GPUs that are actually assigned
# or maybe a method add_fixed_resources that takes in the fixed GPU RAM assignments


class NodeResources:
    """
    Represents available/used/free resources on a compute node.
    """
    def __init__(self, node_id: int, n_threads: float, cpu_ram_gb: float, gpu_usages: np.ndarray,
                 gpu_rams_gb: np.ndarray, physical_core_usages: np.ndarray):
        self.node_id = node_id
        self.n_gpus = len(gpu_usages)
        self.data: np.ndarray = np.array(np.concatenate(
            [[n_threads, cpu_ram_gb], gpu_usages, gpu_rams_gb, physical_core_usages]))
        self.data.setflags(write=True)

    def get_n_threads(self) -> int:
        return round(self.data[0])

    def set_n_threads(self, n_threads: int):
        # somehow necessary because self.data can get non-writeable after transmitting it from another ray process
        self.data = np.copy(self.data)
        self.data[0] = n_threads

    def get_cpu_ram_gb(self) -> float:
        return self.data[1]

    def set_cpu_ram_gb(self, cpu_ram_gb: float) -> None:
        # somehow necessary because self.data can get non-writeable after transmitting it from another ray process
        self.data = np.copy(self.data)
        self.data[1] = cpu_ram_gb

    def get_gpu_usages(self) -> np.ndarray:
        return self.data[2:2+self.n_gpus]

    def get_gpu_rams_gb(self) -> np.ndarray:
        return self.data[2+self.n_gpus:2+2*self.n_gpus]

    def get_physical_core_usages(self) -> np.ndarray:
        return self.data[2+2*self.n_gpus:]

    def get_n_physical_cores(self) -> int:
        return len(self.data) - (2+2*self.n_gpus)

    def get_total_gpu_ram_gb(self) -> float:
        return np.sum(self.get_gpu_rams_gb())

    def get_total_gpu_usage(self) -> float:
        return np.sum(self.get_gpu_usages())

    def get_used_gpu_ids(self) -> np.ndarray:  # todo: naming
        return np.argwhere(self.get_gpu_usages() > 1e-8)[:, 0]

    def get_used_physical_cores(self) -> np.ndarray:
        return np.argwhere(self.get_physical_core_usages() > 1e-8)[:, 0]

    def get_resource_vector(self) -> np.ndarray:
        return np.asarray([self.get_n_threads(), self.get_cpu_ram_gb(),
                           self.get_total_gpu_usage(), self.get_total_gpu_ram_gb()])

    def get_interface_resources(self) -> InterfaceResources:
        return InterfaceResources(n_threads=self.get_n_threads(),
                                  gpu_devices=[f'cuda:{i}' for i in self.get_used_gpu_ids()])

    def __iadd__(self, other: 'NodeResources') -> 'NodeResources':  # operator +=
        self.data += other.data  # todo: some compatibility checks?
        return self

    def __isub__(self, other: 'NodeResources') -> 'NodeResources':
        self.data -= other.data
        return self

    def __imul__(self, other: 'NodeResources') -> 'NodeResources':
        self.data *= other.data
        return self

    def __itruediv__(self, other: 'NodeResources') -> 'NodeResources':
        self.data /= other.data
        return self

    def __add__(self, other: 'NodeResources') -> 'NodeResources':
        result = copy.deepcopy(self)
        result += other
        return result

    def __sub__(self, other: 'NodeResources') -> 'NodeResources':
        result = copy.deepcopy(self)
        result -= other
        return result

    def __mul__(self, other: 'NodeResources') -> 'NodeResources':
        result = copy.deepcopy(self)
        result *= other
        return result

    def __truediv__(self, other: 'NodeResources') -> 'NodeResources':
        result = copy.deepcopy(self)
        result /= other
        return result

    def try_assign(self, required_resources: RequiredResources,
                   fixed_resources: 'SystemResources') -> Optional['NodeResources']:
        rr = required_resources
        fr = fixed_resources.resources[self.node_id]
        if not rr.should_add_fixed_resources():
            fr = NodeResources.zeros_like(fr)
        # todo: distribution across GPUs is potentially suboptimal
        # CPU stuff
        n_threads = fr.get_n_threads() + rr.n_threads
        if self.get_n_threads() < n_threads:
            return None

        cpu_ram_gb = fr.get_cpu_ram_gb() + rr.cpu_ram_gb
        if self.get_cpu_ram_gb() < cpu_ram_gb:
            return None

        n_cores = rr.n_explicit_physical_cores
        physical_core_usages = np.zeros(self.get_n_physical_cores())
        if n_cores > 0:
            free_pcu = self.get_physical_core_usages()
            free_in_sequence = np.convolve(free_pcu, np.ones(n_cores),'valid')
            idx = np.argmax(free_in_sequence >= n_cores - 0.5)
            if free_in_sequence[idx] >= n_cores - 0.5:
                physical_core_usages[idx:idx+n_cores] = 1.0
            else:
                return None

        # GPU stuff
        gpu_usages = np.zeros(self.n_gpus)
        gpu_rams_gb = np.zeros(self.n_gpus)
        gpu_usages_all = fr.get_gpu_usages() + rr.gpu_usage
        gpu_rams_gb_all = fr.get_gpu_rams_gb() + rr.gpu_ram_gb
        gpu_availability = np.logical_and(gpu_usages_all <= self.get_gpu_usages() + 1e-8,
                                          gpu_rams_gb_all <= self.get_gpu_rams_gb())
        available_gpus = np.argwhere(gpu_availability)[:, 0]  # squeeze second dimension
        # sort available gpus by usage
        available_gpu_usages = self.get_gpu_usages()[available_gpus]
        # pick gpus with most free resources first
        available_gpus = available_gpus[np.argsort(available_gpu_usages)[::-1]]
        # print('gpu selection:', gpu_availability, available_gpu_usages, available_gpus)
        if len(available_gpus) < rr.n_gpus:
            return None
        else:
            gpu_ids = available_gpus[:rr.n_gpus]
            for i in gpu_ids:
                gpu_usages[i] = gpu_usages_all[i]
                gpu_rams_gb[i] = gpu_rams_gb_all[i]

        return NodeResources(node_id=self.node_id, n_threads=n_threads, cpu_ram_gb=cpu_ram_gb,
                             gpu_usages=gpu_usages, gpu_rams_gb=gpu_rams_gb,
                             physical_core_usages=physical_core_usages)

    # todo: maybe a __str__ or __repr__ method for printing?
    @staticmethod
    def zeros_like(node_resources: 'NodeResources') -> 'NodeResources':
        result = copy.deepcopy(node_resources)
        result.data *= 0
        return result


class SystemResources:
    """
    System resources, consisting of NodeResources for each node.
    """
    def __init__(self, resources: List[NodeResources]):
        self.resources = resources

    def __getitem__(self, index: int):
        return self.resources[index]

    def __len__(self):
        return len(self.resources)

    def __iadd__(self, other):
        for i in range(len(self.resources)):
            self.resources[i] += other.resources[i]
        return self

    def __isub__(self, other):
        for i in range(len(self.resources)):
            self.resources[i] -= other.resources[i]
        return self

    def __imul__(self, other):
        for i in range(len(self.resources)):
            self.resources[i] *= other.resources[i]
        return self

    def __itruediv__(self, other):
        for i in range(len(self.resources)):
            self.resources[i] /= other.resources[i]
        return self

    def __add__(self, other):
        result = copy.deepcopy(self)
        result += other
        return result

    def __sub__(self, other):
        result = copy.deepcopy(self)
        result -= other
        return result

    def __mul__(self, other):
        result = copy.deepcopy(self)
        result *= other
        return result

    def __truediv__(self, other):
        result = copy.deepcopy(self)
        result /= other
        return result

    def get_n_threads(self):
        return sum([r.get_n_threads() for r in self.resources])

    def get_cpu_ram_gb(self):
        return sum([r.get_cpu_ram_gb() for r in self.resources])

    def get_gpu_usage(self):
        return sum([r.get_total_gpu_usage() for r in self.resources])

    def get_gpu_ram_gb(self):
        return sum([r.get_total_gpu_ram_gb() for r in self.resources])

    def get_num_gpus(self):
        return sum([r.n_gpus for r in self.resources])

    def get_resource_vector(self):
        return sum([r.get_resource_vector() for r in self.resources])

    # todo: maybe a __str__ or __repr__ method for printing?


