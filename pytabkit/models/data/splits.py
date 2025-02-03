import math
from typing import Tuple, List, Optional

import torch

from pytabkit.models import utils
from pytabkit.models.data.data import DictDataset
from pytabkit.models.torch_utils import seeded_randperm


# splits should not reference tasks, since tasks should only be loaded in the respective processes in the DevicePool,
# while splits are loaded earlier

class Split:
    def __init__(self, ds: DictDataset, idxs: Tuple[torch.Tensor, torch.Tensor]):
        """
        :param ds: The dataset that is split into parts
        :param idxs: Tuple of Tensors containing indices of the different parts of ds
        """
        self.ds = ds
        self.idxs = idxs

    def get_sub_ds(self, i):
        return self.ds.get_sub_dataset(self.idxs[i])

    def get_sub_idxs(self, i):
        return self.idxs[i]


class Splitter:
    def get_idxs(self, ds: DictDataset) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def split_ds(self, ds: DictDataset) -> Split:
        idxs = self.get_idxs(ds)
        return Split(ds, idxs)

    def get_split_sizes(self, n_samples: int) -> Tuple:
        raise NotImplementedError()


class RandomSplitter(Splitter):
    def __init__(self, seed, first_fraction=0.8, max_n_first: Optional[int] = None):
        self.seed = seed
        self.first_fraction = first_fraction
        self.max_n_first = max_n_first

    def get_idxs(self, ds: DictDataset) -> Tuple[torch.Tensor, torch.Tensor]:
        # use ceil such that e.g. in the case of 1 sample, the sample ends up in the training set.
        split_idx = int(math.ceil(self.first_fraction * ds.n_samples))
        if self.max_n_first is not None:
            split_idx = min(split_idx, self.max_n_first)
        perm = seeded_randperm(ds.n_samples, ds.device, self.seed)
        return perm[:split_idx], perm[split_idx:]

    def get_split_sizes(self, n_samples: int) -> Tuple:
        split_idx = int(math.ceil(self.first_fraction * n_samples))
        if self.max_n_first is not None:
            split_idx = min(split_idx, self.max_n_first)
        return split_idx, n_samples-split_idx


class IndexSplitter(Splitter):
    def __init__(self, index):
        self.index = index

    def get_idxs(self, ds: DictDataset) -> Tuple[torch.Tensor, torch.Tensor]:
        idxs = torch.arange(ds.n_samples, device=ds.device, dtype=torch.long)
        return idxs[:self.index], idxs[self.index:]

    def get_split_sizes(self, n_samples: int) -> Tuple:
        return self.index, n_samples-self.index


class AllNothingSplitter(Splitter):
    def get_idxs(self, ds: DictDataset) -> Tuple[torch.Tensor, torch.Tensor]:
        all = torch.arange(ds.n_samples, device=ds.device, dtype=torch.long)
        nothing = torch.zeros(0, device=ds.device, dtype=torch.long)
        return all, nothing

    def split_ds(self, ds: DictDataset) -> Split:
        idxs = self.get_idxs(ds)
        return Split(ds, idxs)

    def get_split_sizes(self, n_samples: int) -> Tuple:
        return n_samples, 0


class MultiSplitter:
    def get_idxs(self, ds: DictDataset) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError()

    def split_ds(self, ds: DictDataset) -> List[Split]:
        idxs_list = self.get_idxs(ds)
        return [Split(ds, idxs) for idxs in idxs_list]


class KFoldSplitter(MultiSplitter):
    def __init__(self, k: int, seed: int, stratified=False):
        if k <= 1:
            raise ValueError(f'KFoldSplitter: required k>=2, but received {k=}')
        self.k = k
        self.seed = seed
        self.stratified = stratified

    def get_idxs(self, ds: DictDataset) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        idxs = seeded_randperm(ds.n_samples, device=ds.device, seed=self.seed)
        if self.stratified:
            # do it with random shuffling such that elements of the same class are still shuffled
            perm = torch.argsort(ds.tensors['y'][idxs, 0])
            idxs = idxs[perm]
        fold_len = (ds.n_samples // self.k) * self.k
        fold_idxs = [idxs[start:fold_len:self.k] for start in range(self.k)]
        rest_idxs = idxs[fold_len:]
        idxs_list = []
        for i in range(self.k):
            idxs_1 = torch.cat([fold_idxs[j] for j in range(self.k) if j != i] + [rest_idxs], dim=-1)
            idxs_list.append((idxs_1, fold_idxs[i]))
        return idxs_list

    def get_split_sizes(self, n_samples: int) -> Tuple:
        n_val = n_samples // self.k
        return n_samples - n_val, n_val


class SplitInfo:
    def __init__(self, splitter: Splitter, split_type: str, id: int, alg_seed: int, train_fraction: float = 0.75):
        self.splitter = splitter
        self.split_type = split_type  # one of "random", "default"
        self.id = id
        self.alg_seed = alg_seed
        self.train_fraction = train_fraction

    def get_sub_seed(self, split_idx: int, is_cv: bool):
        return utils.combine_seeds(self.alg_seed, 2 * split_idx + int(is_cv))
        # return self.alg_seed + 5000 * int(is_cv) + 10000 * split_idx

    def get_sub_splits(self, ds: DictDataset, n_splits: int, is_cv: bool) -> List[Split]:
        if not is_cv:
            split = AllNothingSplitter().split_ds(ds)
            return [split] * n_splits

        if n_splits <= 1:
            return [RandomSplitter(seed=self.alg_seed, first_fraction=self.train_fraction).split_ds(ds)]
        else:
            is_classification = ds.tensor_infos['y'].get_cat_sizes()[0].item() > 0
            return KFoldSplitter(n_splits, seed=self.alg_seed, stratified=is_classification).split_ds(ds)

    def get_train_and_val_size(self, n_samples: int, n_splits: int, is_cv: bool) -> Tuple[int, int]:
        n_trainval, n_test = self.splitter.get_split_sizes(n_samples)
        if not is_cv:
            return n_trainval, 0
        elif n_splits <= 1:
            return RandomSplitter(seed=self.alg_seed, first_fraction=self.train_fraction).get_split_sizes(n_trainval)
        else:
            # stratified doesn't influence split sizes
            return KFoldSplitter(n_splits, seed=self.alg_seed, stratified=False).get_split_sizes(n_samples)

