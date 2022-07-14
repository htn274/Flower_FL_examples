import os
from pathlib import Path
from threading import local
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

from utils import DATA_ROOT, load_data
PARTITIONS_PATH = './dataset/fl_cifar10'

class CustomTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transform:
            x = self.transform(x.numpy().astype(np.uint8))
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

def load_dataset():
    transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            ) 
    trainset = CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=transform    
    )
    testset = CIFAR10(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=transform
    )
    return trainset, testset

def create_iid_dataset(num_clients, min_samples=100):
    trainset, testset = load_dataset()
    shuffle_indices = torch.randperm(len(trainset))
    X_train = trainset.data[shuffle_indices]
    y_train = torch.Tensor(trainset.targets)[shuffle_indices]

    split_size = len(trainset) // num_clients
    split_datasets = list(
        zip(
            torch.split(torch.Tensor(X_train), split_size),
            torch.split(torch.Tensor(y_train), split_size)
        )
    )
    return split_datasets[:num_clients], testset

def create_noniid_dataset(num_clients, num_shards):
    trainset, testset = load_dataset()
    sorted_indices = torch.argsort(torch.Tensor(trainset.targets))
    X_train = trainset.data[sorted_indices]
    y_train = torch.Tensor(trainset.targets)[sorted_indices]

    shard_size = len(trainset) // num_shards
    shard_inputs = list(torch.split(torch.Tensor(X_train), shard_size))
    shard_labels = list(torch.split(torch.Tensor(y_train), shard_size))

    num_classes = np.unique(y_train).shape[0]
    shard_inputs_sorted, shard_labels_sorted = [], []
    for i in range(num_shards // num_classes):
        for j in range(0, ((num_shards // num_classes) * num_classes)):
            shard_inputs_sorted.append(shard_inputs[i + j])
            shard_labels_sorted.append(shard_labels[i + j])

    shards_per_clients = num_shards // num_clients

    local_datasets = [
        (
            torch.cat(shard_inputs_sorted[i: i + shards_per_clients]),
            torch.cat(shard_labels_sorted[i : i + shards_per_clients]).long()
        )
        for i in range(0, len(shard_inputs_sorted), shards_per_clients)
    ]
    return local_datasets, testset

if __name__ == '__main__':
    local_datasets, testset = create_noniid_dataset(num_clients=2, num_shards=10)
    print(len(local_datasets[0]))
    for i, dataset in enumerate(local_datasets):
        labels = dataset[1]
        labels_dist = np.unique(labels, return_counts=True)
        print(f"Dataset {i + 1} size of {len(dataset[0])} samples")
        print("Label dist: ", labels_dist)
        save_dir = Path(PARTITIONS_PATH + "_noniid") 
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, save_dir / f"train_{i+1}.pt")