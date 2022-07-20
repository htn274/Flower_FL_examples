import flwr as fl 
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from pathlib import Path
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[
            Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        save_dir: Path = None,
        ):
        super().__init__(fraction_fit, fraction_eval, min_fit_clients, min_eval_clients,
        min_available_clients, eval_fn, on_fit_config_fn, on_evaluate_config_fn, accept_failures, 
        initial_parameters, fit_metrics_aggregation_fn, evaluate_metrics_aggregation_fn)
        self.save_dir = save_dir

    def aggregate_fit(
        self,
        rnd: int,
        results,
        failures,
    ):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None and self.save_dir is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(self.save_dir / f"last-weights.npz", *aggregated_weights)
        return aggregated_weights