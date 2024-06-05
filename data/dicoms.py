import os
from pathlib import Path
from tqdm import tqdm
import pydicom
from pydicom.pixel_data_handlers import apply_modality_lut
import re

def natural_sorted(l):
    def key(x):
        return [int(c) if c.isdigit() else c for c in re.split("([0-9]+)", str(x))]
    return sorted(l, key=key)

def load_stack(dir_path):
    imgs = []
    for root, _, filenames in os.walk(dir_path):
        for filename in filenames:
            dcm_path = Path(root, filename)
            if dcm_path.suffix == ".dcm":
                try:
                    dicom = pydicom.dcmread(dcm_path, force=True)
                except IOError as e:
                    print(f"Невозможно загрузить {dcm_path.stem}")
                else:
                    hu = apply_modality_lut(dicom.pixel_array, dicom)
                    imgs.append(hu)
    return imgs

def glob(dir_path):
    dir_path = Path(dir_path).expanduser().resolve()
    return natural_sorted(dir_path.glob("*.dcm"))

def save_stack(path, stack, prefix="output", exist_ok=True, parents=False):
    path = Path(path).expanduser().resolve()
    path.mkdir(exist_ok=exist_ok, parents=parents)
    for i, s in tqdm(enumerate(stack), mininterval=10.0):
        opath = path / f"{prefix}_{i:05d}.dcm"
        ds = s
        ds.save_as(str(opath))