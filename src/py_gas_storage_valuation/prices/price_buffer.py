import numpy as np

from py_gas_storage_valuation.prices.base_price_simulator import (
    BaseForwardSimulator,
)


# NOTE: consider more efficient implementation using queue
class ForwardCurvePathBuffer:
    def __init__(
        self,
        simulator: BaseForwardSimulator,
        buffer_size: int = 5_000,
        prefetch_threshold: float = 0.2,
    ):
        self.simulator = simulator
        self.buffer_size = buffer_size
        self.prefetch_threshold = prefetch_threshold

        self._paths = np.empty((0, 12, 12), dtype=np.float32)
        self._position = 0

        self._refill()

    def _refill(self):
        new_paths = self.simulator.batch_simulate_forward(self.buffer_size)

        # concat the new paths with the remaining paths in the buffer
        self._paths = np.concatenate(
            [self._paths[self._position :], new_paths],
            axis=0,
        )

        self._position = 0

    def get_path(self):
        if self.remaining < self.buffer_size * self.prefetch_threshold:
            self._refill()

        path = self._paths[self._position]
        self._position += 1

        return path

    def get_batch(self, batch_size: int):
        # TODO: check this logic
        if self.remaining < batch_size:
            self._refill()

        batch = self._paths[self._position : self._position + batch_size]
        self._position += batch_size

        return batch

    @property
    def remaining(self):
        return len(self._paths) - self._position
