import argparse
from collections import OrderedDict

import flwr as fl
import numpy as np
import torch
from flwr.common import parameters_to_weights, weights_to_parameters
from pyrsistent import v
from torch.utils.data import DataLoader, Dataset

from models import MnistNet
from utils import load_data, test, train

DEVICE = "mps"
DEFAULT_SERVER_ADDRESS = "[::]:8080"

def get_eval_fn():
    _, testloader = load_data()
    def centralized_eval(weights):
        model = MnistNet()
        params_dict = zip(model.state_dict().keys(), weights)
        state_dict = OrderedDict({k: torch.tensor(np.atleast_1d(v)) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.to(DEVICE)
        test_loss, test_acc, num_samples = test(model, testloader, device=DEVICE)
        metrics = {"centralized_acc": test_acc, "num_samples": num_samples}
        return test_loss, metrics
    return centralized_eval

def fit_config(rnd):
    # print(f"Round {rnd}")
    lr = 0.01
    if rnd > 10: 
        lr = 0.001
    config = {
        "local_batch_size": 32,
        "num_epochs": 5,
        "optim_lr": lr,
    }
    return config

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--rnd",
        type=int,
        default=10,
        help="number of rounds"
    )
    

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    sample_fraction = 1.0
    min_sample_size = 2
    min_num_clients = 2

    strategy = fl.server.strategy.FedAvg(
        fraction_fit = sample_fraction,
        min_fit_clients = min_sample_size,
        min_available_clients=min_num_clients,
        eval_fn = get_eval_fn(),
        on_fit_config_fn=fit_config,
    )
    fl.server.start_server(
        DEFAULT_SERVER_ADDRESS,
        config={"num_rounds": args.rnd},
        strategy=strategy
    )
    

if __name__ == '__main__':
    main()
