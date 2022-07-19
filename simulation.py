from server import *
from client import FlowerClient
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--rnd',
        type=int,
        required=True
    )
    args = parser.parse_args()
    pool_size = 2
    client_resources = {
        "num_cpus": 1
    }
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_eval=1.0,
        min_fit_clients=2,
        min_eval_clients=2,
        min_available_clients=pool_size,  # All clients should be available
        on_fit_config_fn=fit_config,
        eval_fn=get_eval_fn(), # centralised testset evaluation of global model
    )
    fed_dir = 'datasets/fl_mnist_noniid'
    def client_fn(cid: str):
        # create a single client instance
        return FlowerClient(cid, fed_dir, verbose=True)

    ray_init_args = {"include_dashboard": False}   
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        num_rounds=args.rnd,
        strategy=strategy,
        ray_init_args=ray_init_args,
    )