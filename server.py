import argparse
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
import pickle


import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models import MnistNet
from utils import load_data, test, train

from custom_strategy import SaveModelStrategy

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rnd", type=int, default=10, help="number of global rounds")
    parser.add_argument("--sample_fraction", type=float, default=1.0, help='fraction of participating clients at each round') 
    parser.add_argument("--num_clients", type=int, default=2, help="total number of clients")
    parser.add_argument("--batch_size", type=int, default=5, help="batch size for local update")
    parser.add_argument("--num_epochs", type=int, default=5, help="number of local epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="learning for of local update")
    parser.add_argument("--save_dir", type=str, default=None, help="saving directory for global training information")
    args = parser.parse_args()
    model_name = f"FedAvg_{args.rnd}_{args.num_clients}_{args.batch_size}_{args.num_epochs}_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    if args.save_dir is not None:
        args.save_dir = Path(args.save_dir) / model_name
        args.save_dir.mkdir(parents=True, exist_ok=True)
    return args

def save_hist(history, args):
    f = open(args.save_dir / "hist.pkl", "wb")
    pickle.dump(history, f)
    f.close()

def main():
    args = parse_args()
    
    def fit_config(rnd):
        config = {
            "local_batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "optim_lr": args.lr
        }
        return config

    strategy = SaveModelStrategy(
        fraction_fit = args.sample_fraction,
        min_fit_clients = int(args.num_clients*args.sample_fraction),
        min_available_clients=args.num_clients,
        eval_fn = get_eval_fn(),
        on_fit_config_fn=fit_config,
        save_dir=args.save_dir,
    )
    hist = fl.server.start_server(
        DEFAULT_SERVER_ADDRESS,
        config={"num_rounds": args.rnd},
        strategy=strategy
    )
    print("Saving training history")
    save_hist(hist, args)

if __name__ == '__main__':
    main()
