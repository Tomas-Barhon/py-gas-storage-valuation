from py_gas_storage_valuation.prices.base_price_simulator import BaseForwardSimulator
import numpy as np


class MockForwardSimulator(BaseForwardSimulator):
    def __init__(self) -> None: ...

    def batch_simulate_forward(self, n_paths: int) -> np.ndarray:
        min_price = 12.0
        max_price = 20.0
        n_steps = 12
        curves = np.zeros(
            (n_paths, n_steps, n_steps),
            dtype=np.float32,
        )

        upper = np.triu_indices(n_steps)

        curves[:, upper[0], upper[1]] = np.random.uniform(
            min_price,
            max_price,
            size=(n_paths, len(upper[0])),
        )
        return curves
