from server import parse_args, get_eval_fn
from client import FlowerClient
import flwr as fl

if __name__ == '__main__':
    args = parse_args()

    client_resources = {
        "num_cpus": 1
    }
    def fit_config(rnd):
        config = {
            "local_batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "optim_lr": args.lr
        }
        return config

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.sample_fraction,
        fraction_eval=args.sample_fraction,
        min_fit_clients=args.num_clients*args.sample_fraction,
        min_eval_clients=args.num_clients*args.sample_fraction,
        min_available_clients=args.num_clients,  # All clients should be available
        on_fit_config_fn=fit_config,
        eval_fn=get_eval_fn(), # centralised testset evaluation of global model
    )
    def client_fn(cid: str):
        # create a single client instance
        return FlowerClient(cid, args.fed_dir, verbose=False)

    ray_init_args = {"include_dashboard": False}   
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        client_resources=client_resources,
        num_rounds=args.rnd,
        strategy=strategy,
        ray_init_args=ray_init_args,
    )