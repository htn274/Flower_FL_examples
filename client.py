from tkinter import W
import flwr as fl
from collections import OrderedDict
from models import MnistNet
from utils import train, test, load_partition, load_data
import torch
import argparse
import ray
from flwr.simulation.ray_transport.ray_client_proxy import RayClientProxy

DEFAULT_SERVER_ADDRESS = "[::]:8080"
DEVICE = "mps"

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid:int, fed_dir_data:str, verbose=False):
        self.cid = cid
        self.fed_dir = fed_dir_data
        self.model = MnistNet()
        self.device = DEVICE
        self.testloader = None
        self.verbose = verbose
    
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
        if self.verbose: 
            print(f"Client {self.cid} is training")
        # Get model's weights from server
        self.set_parameters(params)
        # Create train loader
        trainset  = load_partition(self.cid, data_dir=self.fed_dir) 
        if not "CPU" in ray.worker.get_resource_ids():
            num_workers = 8
        else:
            num_workers = len(ray.worker.get_resource_ids()["CPU"])
        kwargs = {"num_workers": num_workers, "pin_memory": True, }
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['local_batch_size'], 
                                                    shuffle=True, **kwargs)
        # Train with local dataset
        train(self.model, trainloader, 
            config["num_epochs"], self.device, config["optim_lr"], verbose=self.verbose)
            
        return self.get_parameters(), len(trainset), {}

    def evaluate(self, params, config):
        """
        Evaluate the local model with the centralized test.
        Hence we don't have set_params as in the original flower tutorial
        """
        if self.testloader is None:
            _, self.testloader = load_data()
        loss, acc, num_samples = test(self.model, self.testloader, device=DEVICE)
        if self.verbose:
            print(f"Test acc: {acc} | num_samples: {num_samples}")
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
    ray_init_args = {
            "ignore_reinit_error": True,
            "include_dashboard": False,
        }
    ray.init(**ray_init_args)
   # Start client
    client = FlowerClient(args.cid, args.data_dir, verbose=True)
    fl.client.start_numpy_client(DEFAULT_SERVER_ADDRESS, client)
