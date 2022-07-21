from tkinter import W
from custom_strategy import SaveModelStrategy
from models import MnistNet
from server import parse_args, get_eval_fn, save_hist
from client import FlowerClient
import flwr as fl
from flwr.common.parameter import weights_to_parameters

def get_initial_params():
    model = MnistNet()
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = weights_to_parameters(weights)
    return parameters

if __name__ == '__main__':
    args = parse_args()

    client_resources = {
        "num_cpus": 1
    }
    def fit_config(rnd):
        config = {
            "local_batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            # "optim_lr": args.lr
            "optim_lr": args.lr if rnd <= 10 else args.lr * args.lr_decay
        }
        return config


    strategy = fl.server.strategy.FedYogi(
        fraction_fit=args.sample_fraction,
        fraction_eval=args.sample_fraction,
        min_fit_clients=args.num_clients*args.sample_fraction,
        min_eval_clients=args.num_clients*args.sample_fraction,
        min_available_clients=args.num_clients,  # All clients should be available
        on_fit_config_fn=fit_config,
        accept_failures=False,
        eval_fn=get_eval_fn(), # centralised testset evaluation of global model
        # save_dir=args.save_dir,
        eta=0.001,
        eta_l=0.001,
        tau=0.001,
        initial_parameters=get_initial_params()
    )
    def client_fn(cid: str):
        # create a single client instance
        return FlowerClient(cid, args.fed_dir, verbose=False)

    ray_init_args = {"include_dashboard": False}   
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        client_resources=client_resources,
        num_rounds=args.rnd,
        strategy=strategy,
        ray_init_args=ray_init_args,
    )
    print("Saving training history")
    save_hist(hist, args)