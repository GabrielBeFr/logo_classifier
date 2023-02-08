import yaml
from yaml.loader import SafeLoader
import h5py
import numpy as np
import json

def get_offset(f: h5py.File) -> int:
    external_id_dset = f["external_id"]
    array = external_id_dset[:]
    non_zero_indexes = np.flatnonzero(array)
    return int(non_zero_indexes[-1]) + 1

def get_config(config_file: str):
    with  open('config.yaml') as f:
        return yaml.load(f, Loader=SafeLoader)

def get_labels(labels_path: str):
    ids = []
    str = []
    with open(labels_path, 'rb') as f:
        for row in f:
            dicti = json.loads(row)
            ids.append(dicti["id"])
            str.append(dicti["class"])
    return str, ids