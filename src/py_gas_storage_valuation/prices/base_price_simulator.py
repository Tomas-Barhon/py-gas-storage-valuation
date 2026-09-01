class BasePriceSimulator(Protocol):
    """
    Base class for price simulators.
    """

    def step(self, current_price: float) -> float:
        """
        Simulate the next price step based on the current price.

        Parameters
        ----------
        current_price : float
            The current price.

        Returns
        -------
        float
            The next simulated price.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
