import polars as pl
from dataclasses import dataclass
from typing import Literal, Optional, Generator


@dataclass(frozen=True, slots=True)
class ForwardCurve:
    """A class to represent a forward curve for gas storage valuation.

    Attributes:
        df (pl.DataFrame): A DataFrame containing 'date' and 'price' columns
        period (str): The period of the forward curve, either 'M' for monthly or 'D' for daily.
        metadata (dict, optional): Additional metadata about the forward curve.
    """

    df: pl.DataFrame
    period: Literal["M", "D"]
    metadata: Optional[dict] = None

    @property
    def dates(self) -> pl.Series:
        return self.df["date"]

    @property
    def prices(self) -> pl.Series:
        return self.df["price"]

    def __post_init__(self):
        self._validate_data()
        return self

    def _validate_data(self):
        if "date" not in self.df.columns or "price" not in self.df.columns:
            raise ValueError("DataFrame must contain 'date' and 'price' columns.")
        if self.df["date"].dtype != pl.Date:
            raise ValueError("'date' column must be of type Date.")
        if self.df["price"].dtype != pl.Float32:
            raise ValueError("'price' column must be of type Float32.")
        if self.period not in ["M", "D"]:
            raise ValueError("Period must be either 'M' for monthly or 'D' for daily.")
        if len(self.df) == 0:
            raise ValueError("DataFrame cannot be empty.")
        if self.period == "M" and len(self.df) > 12:
            raise ValueError("Monthly forward curve cannot have more than 12 entries.")
        if self.period == "D" and len(self.df) > 31:
            raise ValueError("Daily forward curve cannot have more than 31 entries.")


class MaturingForwardCurve:
    """
    A generator class representing set of forward curves that mature in time. Prepended with zeros already passed maturity.
    Yields curves used as a state for the dynamic optimization.

    """

    def __init__(self, forward_curves: list[ForwardCurve]) -> None:
        self.forward_curves = forward_curves
        self._validate_forward_curves()

    def _validate_forward_curves(self) -> "MaturingForwardCurve":
        if not self.forward_curves:
            raise ValueError("The list of forward curves cannot be empty.")
        curve_length = len(self.forward_curves[0].df)
        for curve in self.forward_curves:
            if not isinstance(curve, ForwardCurve):
                raise ValueError(
                    "All items in the list must be instances of ForwardCurve."
                )
            if len(curve.df) != curve_length:
                raise ValueError(
                    "All forward curves must have the same number of entries."
                )
        return self

    def __iter__(self) -> Generator[pl.DataFrame, None, None]:
        for i in range(len(self.forward_curves)):
            yield self.forward_curves[i].df
