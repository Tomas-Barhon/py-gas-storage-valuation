import numpy as np
from typing import Protocol


class BaseForwardSimulator(Protocol):
    """
    Base class for price simulators.
    """

    def batch_simulate_forward(self, n_paths: int) -> np.ndarray:
        """Return a batch of n_paths 2D arrays of simulated forward prices."""
        raise NotImplementedError("This method should be implemented by subclasses.")
