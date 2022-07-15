from ast import parse
from tkinter import W
from typing_extensions import Required
import flwr as fl
from collections import OrderedDict

from numpy import partition
from models import MnistNet
from utils import train, test, load_partition
import torch
import argparse
import numpy as np

DEFAULT_SERVER_ADDRESS = "[::]:8080"
DEVICE = "mps"

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, trainloader, testloader=None):
        self.cid = cid
        self.model = model
        print(f"Client {cid} has {len(trainloader.dataset)} samples")
        self.trainloader = trainloader
        self.testloader = testloader
    
    def get_parameters(self):
        """ 
        Return a list of params of the local model
        """
        print(f"[Client {self.cid}] get_parameters")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, params):
        """ 
        Set the parameters for the local model
        """
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, params, config):
        """ 
        Train the local model
        Return a tuple of (trained params, size of local training set, some keys for authentication if available)
        """
        print(f"Client {self.cid} is training")
        self.set_parameters(params)
        train(self.model, self.trainloader, 
            config["num_epochs"], DEVICE, config["optim_lr"])
        return self.get_parameters(), len(self.trainloader.dataset), {}

    def evaluate(self, params, config):
        # print(f"Client {self.cid} evaluated on test set")
        self.set_parameters(params)
        loss, acc, num_samples = test(self.model, self.testloader, device=DEVICE)
        # print(f"Test acc: {acc} | num_samples: {num_samples}")
        return float(loss), num_samples, {"accuracy": float(acc)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cid",
        type=int,
        required=True,
        help="client id"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True
    )
    args = parser.parse_args()

    # Load data
    # trainloader, testloader = load_data()
    trainloader, testloader = load_partition(args.cid, data_dir=args.data_dir) 
    
    # Load model
    model = MnistNet()
    # Start client
    client = FlowerClient(args.cid, model, trainloader, testloader)
    fl.client.start_numpy_client(DEFAULT_SERVER_ADDRESS, client)
