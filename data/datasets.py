from data import tiffs, dicoms
from pathlib import Path
import numpy as np
import torch
from itertools import combinations
from torch.utils.data import (
    Dataset
)
import tifffile
import pydicom


class TiffDataset(Dataset):
    def __init__(self, glob):
        super(TiffDataset, self).__init__()
        glob = Path(glob)
        self.glob = glob
        self.paths = tiffs.natural_sorted(Path(glob.parent).glob(glob.name))

    def __getitem__(self, i):
        img = tifffile.imread(str(self.paths[i]))
        
        if img.ndim == 2:
            img = img[None, ...]
        return torch.from_numpy(img.astype(np.float32))

    def __len__(self):
        return len(self.paths)


class DicomDataset(Dataset):
    def __init__(self, glob):
        super(DicomDataset, self).__init__()
        glob = Path(glob)
        self.glob = glob
        self.paths = dicoms.natural_sorted(Path(glob.parent).glob(glob.name))

    def __getitem__(self, i):
        img = pydicom.dcmread(str(self.paths[i]), force=True)
        
        if img.ndim == 2:
            img = img[None, ...]
        return torch.from_numpy(img.astype(np.float32))

    def __len__(self):
        return len(self.paths)


class SupervisedDataset(Dataset):
    def __init__(self, input_ds, target_ds):
        super(SupervisedDataset, self).__init__()
        self.input_ds = input_ds
        self.target_ds = target_ds

        assert len(input_ds) == len(target_ds)

    def __getitem__(self, i):
        return self.input_ds[i], self.target_ds[i]

    def __len__(self):
        return len(self.input_ds)


class StrategyDataset(Dataset):
    def __init__(self, *datasets, strategy="X:1"):
        super(StrategyDataset, self).__init__()

        self.datasets = datasets
        max_len = max(len(ds) for ds in datasets)
        min_len = min(len(ds) for ds in datasets)

        assert min_len == max_len

        assert strategy in ["X:1", "1:X"]
        self.strategy = strategy

        if strategy == "X:1":
            num_input = self.num_splits - 1
        else:
            num_input = 1

        split_idxs = set(range(self.num_splits))
        self.input_idxs = list(combinations(split_idxs, num_input))
        self.target_idxs = [split_idxs - set(idxs) for idxs in self.input_idxs]

    @property
    def num_splits(self):
        return len(self.datasets)

    @property
    def num_slices(self):
        return len(self.datasets[0])

    def __getitem__(self, i):
        num_splits = self.num_splits
        slice_idx = i // num_splits
        split_idx = i % num_splits

        input_idxs = self.input_idxs[split_idx]
        target_idxs = self.target_idxs[split_idx]

        slices = [ds[slice_idx] for ds in self.datasets]
        inputs = [slices[j] for j in input_idxs]
        targets = [slices[j] for j in target_idxs]
        inp = torch.mean(torch.stack(inputs), dim=0)
        tgt = torch.mean(torch.stack(targets), dim=0)

        return inp, tgt

    def __len__(self):
        return self.num_splits * self.num_slices
