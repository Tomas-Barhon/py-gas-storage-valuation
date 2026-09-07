from py_gas_storage_valuation.data.prices import MaturingForwardCurve


class BasePriceSimulator(Protocol):
    """
    Base class for price simulators.
    """

    def simulate_forward_curve(self) -> MaturingForwardCurve:
        raise NotImplementedError("This method should be implemented by subclasses.")
