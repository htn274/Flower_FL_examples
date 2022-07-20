import argparse
from ast import parse
import os
from pathlib import Path
from threading import local
from venv import create
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, MNIST

from utils import DATA_ROOT, load_data
PARTITIONS_PATH = './datasets/fl_mnist'

torch.manual_seed(0)
np.random.seed(0)

def load_dataset():
    transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307, ), (0.3081))
                    # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            ) 
    trainset = MNIST(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=transform    
    )
    testset = MNIST(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=transform
    )
    return trainset, testset

def create_iid_dataset(num_clients, num_shards=0):
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
    """
    The implementation refers to 
    https://github.com/vaseline555/Federated-Averaging-PyTorch/blob/main/src/utils.py
    """
    trainset, testset = load_dataset()
    sorted_indices = torch.argsort(torch.Tensor(trainset.targets))
    X_train = trainset.data[sorted_indices]
    y_train = torch.Tensor(trainset.targets)[sorted_indices]

    shard_size = len(trainset) // num_shards
    shard_inputs = list(torch.split(torch.Tensor(X_train), shard_size))
    shard_labels = list(torch.split(torch.Tensor(y_train), shard_size))
    # print(f"Num shards {num_shards}, {shard_size} samples/shard")

    num_classes = np.unique(y_train).shape[0]
    shard_inputs_sorted, shard_labels_sorted = [], []
    for i in range(num_shards // num_classes):
        for j in range(i, num_shards, (num_shards//num_classes)):
            shard_inputs_sorted.append(shard_inputs[j])
            shard_labels_sorted.append(shard_labels[j])

    shards_per_clients = num_shards // num_clients

    local_datasets = [
        (
            torch.cat(shard_inputs_sorted[i: i + shards_per_clients]),
            torch.cat(shard_labels_sorted[i : i + shards_per_clients]).long()
        )
        for i in range(0, len(shard_inputs_sorted), shards_per_clients)
    ]
    return local_datasets, testset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        type=str,
        default="iid",
        help="type to split data: iid or noniid"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="type to split data: iid or noniid"
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=2,
        help="number of partitions (clients)"
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=0,
        help="number of shards in noniid"
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    if args.type == "iid":
        create_fn = create_iid_dataset
    else:
        create_fn = create_noniid_dataset

    local_datasets, testset = create_fn(num_clients=args.num_clients, num_shards=args.num_shards)

    print(len(local_datasets[0]))
    for i, dataset in enumerate(local_datasets):
        labels = dataset[1]
        labels_dist = np.unique(labels, return_counts=True)
        print(f"Dataset {i + 1} size of {len(dataset[0])} samples")
        print("Label dist: ", labels_dist)
        save_dir = Path(f"{args.data_dir}_{args.type}_{args.num_clients}clients") 
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, save_dir / f"train_{i+1}.pt")


    print("Test set info: ", len(testset))