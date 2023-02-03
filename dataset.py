import torch.utils.data as t_data
import h5py
from pathlib import Path
from typing import List, Any
import tqdm
from utils import get_offset

class LogoDataset(t_data.Dataset): 
    '''Class for main Dataset Classes''' 
    def __init__(self, hdf5_file: Path, transforms: List[str]):
        self.file = h5py.File(hdf5_file, 'r')
        offset = get_offset(self.file)
        self.embeddings = self.file['embedding'][:offset]
        self.ids = self.file['external_id'][:offset]
        self.classes = self.file['class'][:offset]
        self.transforms = transforms
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        for transform in self.transforms:
            exec("embedding =" + transform + "(embedding)")

        return embedding, self.classes[idx], self.ids[idx]

def get_datasets(train_path: Path, val_path: Path, test_path: Path, transforms : List[str]):
    train_dataset = LogoDataset(train_path, transforms)
    val_dataset = LogoDataset(val_path, transforms)
    test_dataset = LogoDataset(test_path, transforms)

    return train_dataset, val_dataset, test_dataset

def get_weights(datasets_list: List[LogoDataset], loader_batch_size: int, num_threads: int):
    res_list = []
    for dataset in datasets_list:
        amount_dict = {}
        data_gen = t_data.DataLoader(dataset=dataset, batch_size=loader_batch_size, num_workers=num_threads)
        print("Starting weight generation loop 1")
        for data_batch in tqdm.tqdm(data_gen):
            class_batch = data_batch[1]
            for classe in class_batch: 
                try:
                    amount_dict[str(classe.item())]+=1
                except KeyError:
                    amount_dict[str(classe.item())]=1

        weight_dict = {}
        for classe in amount_dict:
            if amount_dict[classe] == 0:
                print("there is no class")
            weight_dict[classe] = 1/amount_dict[classe]

        weight_list = []
        print("Starting weight generation loop 2")
        for data_batch in tqdm.tqdm(data_gen):
            class_batch = data_batch[1]
            for classe in class_batch:
                weight_list.append(weight_dict[str(classe.item())])

        res_list.append(weight_list)
                
    return res_list


def get_dataloader(train_path: Path, 
                val_path: float, 
                test_path: float, 
                transforms: List[str],
                num_threads: int,
                loader_batch_size: int,
                ):
    """
    Returns the three dataloaders for training, validation and test.
    
    Inputs:
     train_path: pathlib.Path of the hdf5 train dataset
     val_path: pathlib.Path of the hdf5 val dataset
     test_path: pathlib.Path of the hdf5 test dataset
     transforms: list of strings with the name of functions to use as transforms
     num_threads: int of the amount of CPU used to run the dataset
     loader_batch_size: int of the size of batches loaded at the same time by the dataloader
    """

    # get train, val and test datasets
    train_dataset, valid_dataset, test_dataset = get_datasets(train_path, val_path, test_path, transforms)

    # define samplers for train, val and test
    train_weights, valid_weights, test_weights = get_weights([train_dataset, valid_dataset, test_dataset], loader_batch_size, num_threads)

    train_sampler = t_data.WeightedRandomSampler(weights=train_weights, num_samples=len(train_dataset))
    valid_sampler = t_data.WeightedRandomSampler(weights=valid_weights, num_samples=len(valid_dataset))
    test_sampler = t_data.WeightedRandomSampler(weights=test_weights, num_samples=len(test_dataset))

    # create dataloader for train, val and test
    train_loader = t_data.DataLoader(dataset=train_dataset, batch_size=loader_batch_size, num_workers=num_threads, sampler=train_sampler)
    valid_loader = t_data.DataLoader(dataset=valid_dataset, batch_size=loader_batch_size, num_workers=num_threads, sampler=valid_sampler)
    test_loader = t_data.DataLoader(dataset=test_dataset, batch_size=loader_batch_size, num_workers=num_threads, sampler=test_sampler)

    return train_loader, valid_loader, test_loader

def identity_transform(element: Any):
    return element

if __name__ == '__main__':
    train_path = Path("/home/gabriel/off/logo_classifier/datasets/train_dataset.hdf5")
    val_path = Path("/home/gabriel/off/logo_classifier/datasets/val_dataset.hdf5")
    test_path = Path("/home/gabriel/off/logo_classifier/datasets/test_dataset.hdf5")
    train_loader, val_loader, test_loader = get_dataloader(train_path, val_path, test_path, ["identity_transform"], 1, 15)
    
    import numpy as np
    train_classes = np.zeros(168)
    val_classes = np.zeros(168)
    test_classes = np.zeros(168)
    print("Starting train_loader")
    for data in tqdm.tqdm(train_loader):
        for classe in data[1]:
            train_classes[classe] += 1
    print("Starting val_loader")
    for data in tqdm.tqdm(val_loader):
        for classe in data[1]:
            val_classes[classe] += 1
    print("Starting test_loader")
    for data in tqdm.tqdm(test_loader):
        for classe in data[1]:
            test_classes[classe] += 1

    import matplotlib.pyplot as plt
    plt.plot(train_classes)
    plt.show()
    plt.plot(val_classes)
    plt.show()
    plt.plot(test_classes)
    plt.show()