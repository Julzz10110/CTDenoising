import tifffile
import pydicom
from tqdm import tqdm
import numpy as np
from pathlib import Path
import re


def natural_sorted(l):
    def key(x):
        return [int(c) if c.isdigit() else c for c in re.split("([0-9]+)", str(x))]
    return sorted(l, key=key)


def load_stack(paths, binning=1, use_tqdm=True):
    paths = list(paths)
    img0 = tifffile.imread(str(paths[0]))
    img0 = img0[::binning, ::binning]
    dtype = img0.dtype
    imgs = np.empty((len(paths), *img0.shape), dtype=dtype)

    progress = tqdm if use_tqdm else lambda x: x

    for i, p in progress(enumerate(paths)):
        imgs[i] = tifffile.imread(str(p))[::binning, ::binning]
    return imgs


def glob(dir_path):
    dir_path = Path(dir_path).expanduser().resolve()
    return natural_sorted(dir_path.glob("*.tif"))


def save_stack(path, stack, prefix="output", exist_ok=True, parents=False):
    path = Path(path).expanduser().resolve()
    path.mkdir(exist_ok=exist_ok, parents=parents)
    for i, s in tqdm(enumerate(stack), mininterval=10.0):
        opath = path / f"{prefix}_{i:05d}.tif"
        tifffile.imsave(str(opath), s)


def load_sino(paths, binning=1, dtype=None, flip_y=False):
    paths = list(paths)
    img0 = tifffile.imread(str(paths[0]))
    img0 = img0[::binning, ::binning]
    if dtype is None:
        dtype = img0.dtype
    imgs = np.empty((img0.shape[0], len(paths), img0.shape[1]), dtype=dtype)
    for i, p in tqdm(enumerate(paths)):
        if flip_y:
            imgs[:, i, :] = tifffile.imread(str(p))[::-binning, ::binning]
        else:
            imgs[:, i, :] = tifffile.imread(str(p))[::binning, ::binning]
    return imgs
