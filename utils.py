import yaml
from yaml.loader import SafeLoader
import h5py
import numpy as np

def get_offset(f: h5py.File) -> int:
    external_id_dset = f["external_id"]
    array = external_id_dset[:]
    non_zero_indexes = np.flatnonzero(array)
    return int(non_zero_indexes[-1]) + 1

def get_config(config_file: str):
    with  open('config.yaml') as f:
        return yaml.load(f, Loader=SafeLoader)
