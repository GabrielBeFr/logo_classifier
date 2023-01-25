import torch.utils.data as t_data
import h5py
from pathlib import Path
from typing import List, Any

class LogoDataset(t_data.Dataset): 
    '''Class for main Dataset Classes''' 
    def __init__(self, hdf5_file, transform):
        self.file = h5py.File(hdf5_file, 'r')
        self.embeddings = self.file['embedding']
        self.ids = self.file['external_id']
        self.classes = self.file['class']
        self.weights = self.file['weight']
        self.transform = transform
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.transform(self.embeddings[idx]), self.classes[idx], self.ids[idx]

# class DatasetTransformer(t_data.Dataset):

#     def __init__(self, base_dataset, transform):
#         self.base_dataset = base_dataset
#         self.transform = transform

#     def __getitem__(self, index):
#         embedding, classe, id = self.base_dataset[index]
#         return self.transform(embedding), classe, id

#     def __len__(self):
#         return len(self.base_dataset)

# def transform_datasets(datasets_list: List[LogoDataset], transform):
#     res_list = []
#     for dataset in datasets_list:
#         res_list.append(DatasetTransformer(dataset, transform))
#     return res_list

def get_datasets(data_path: Path, valid_ratio: float, test_ratio: float, transform):
    complete_dataset = LogoDataset(data_path, transform)
    nb_train = int((1.0 - valid_ratio - test_ratio)*len(complete_dataset))
    nb_valid = int(valid_ratio*len(complete_dataset))
    nb_test = int(test_ratio*len(complete_dataset))

    return t_data.dataset.random_split(complete_dataset, [nb_train, nb_valid, nb_test])

def get_dataloader(data_path: Path, 
                valid_ratio: float, 
                test_ratio: float, 
                transform,
                num_threads: int,
                loader_batch_size: int,
                ):

    # get train, val and test datasets
    train_dataset, valid_dataset, test_dataset = get_datasets(data_path, valid_ratio, test_ratio, transform)

    # apply transforms on datasets
    # train_dataset, valid_dataset, test_dataset = transform_datasets([train_dataset, valid_dataset, test_dataset], transform)

    # define samplers for train, val and test
    breakpoint()
    train_sampler = t_data.WeightedRandomSampler(weights=train_dataset.weights, num_samples=len(train_dataset))
    valid_sampler = t_data.WeightedRandomSampler(weights=valid_dataset.weights, num_samples=len(valid_dataset))
    test_sampler = t_data.WeightedRandomSampler(weights=test_dataset.weights, num_samples=len(test_dataset))

    # create dataloader for train, val and test
    train_loader = t_data.DataLoader(dataset=train_dataset, batch_size=loader_batch_size, num_workers=num_threads, sampler=train_sampler)
    valid_loader = t_data.DataLoader(dataset=valid_dataset, batch_size=loader_batch_size, num_workers=num_threads, sampler=valid_sampler)
    test_loader = t_data.DataLoader(dataset=test_dataset, batch_size=loader_batch_size, num_workers=num_threads, sampler=test_sampler)

    return train_loader, valid_loader, test_loader

def identity_transform(element: Any):
    return element

if __name__ == '__main__':
    train_loader, valid_loader, test_loader = get_dataloader(Path("/home/gabriel/off/data/data_file.hdf5"), 0.1, 0.1, identity_transform, 1, 15)
    for data in train_loader:
        breakpoint()