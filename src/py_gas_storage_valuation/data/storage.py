import polars as pl
from dataclasses import dataclass
import numpy as np
from rich.table import Table
from rich.console import Console
from io import StringIO


@dataclass
class InjectionWithdrawalCurve:
    bin_upper_thresholds: np.ndarray
    injection_rates: np.ndarray
    withdrawal_rates: np.ndarray

    def __post_init__(self):
        if not (
            len(self.bin_upper_thresholds)
            == len(self.withdrawal_rates)
            == len(self.injection_rates)
        ):
            raise ValueError("All arrays must have the same length.")
        if not np.all(np.diff(self.bin_upper_thresholds) > 0):
            raise ValueError("bin_upper_thresholds must be strictly increasing.")
        if np.any(self.withdrawal_rates < 0) or np.any(self.injection_rates < 0):
            raise ValueError("Rates must be between 0 and 1.")
        if np.any(self.withdrawal_rates > 1) or np.any(self.injection_rates > 1):
            raise ValueError("Rates must be between 0 and 1.")

    def _get_rate(self, value: float, injection: bool = True) -> float:
        """Return injection or withdrawal_rates for the bin containing value."""
        idx = np.searchsorted(self.bin_upper_thresholds, value, side="right")
        if idx == len(self.bin_upper_thresholds):
            idx -= 1
        return self.injection_rates[idx] if injection else self.withdrawal_rates[idx]

    def get_injection_rate(self, value: float) -> float:
        """Return injection rate for the bin containing value."""
        return self._get_rate(value, injection=True)

    def get_withdrawal_rate(self, value: float) -> float:
        """Return withdrawal rate for the bin containing value."""
        return self._get_rate(value, injection=False)

    @property
    def num_bins(self) -> int:
        """Return the number of bins defined by the upper thresholds."""
        return len(self.bin_upper_thresholds)

    @property
    def max_injection_rate(self) -> float:
        """Return the maximum injection rate across all bins."""
        return np.max(self.injection_rates)

    @property
    def max_withdrawal_rate(self) -> float:
        """Return the maximum withdrawal rate across all bins."""
        return np.max(self.withdrawal_rates)

    def __repr__(self) -> str:
        return (
            f"InjectionWithdrawalCurve(num_bins={self.num_bins}, "
            f"bin_upper_thresholds={self.bin_upper_thresholds}, "
            f"injection_rates={self.injection_rates}, "
            f"withdrawal_rates={self.withdrawal_rates})"
        )

    def __str__(self) -> str:
        table = Table(title="Injection/Withdrawal Curve")
        table.add_column("Bin", justify="right")
        table.add_column("Upper Threshold", justify="right")
        table.add_column("Withdrawal Rate", justify="right")
        table.add_column("Injection Rate", justify="right")

        lower_bound = 0.0
        for thresh, w_rate, i_rate in zip(
            self.bin_upper_thresholds, self.withdrawal_rates, self.injection_rates
        ):
            table.add_row(
                str(lower_bound),
                f"{thresh:.4g}",
                f"{w_rate:.4g}",
                f"{i_rate:.4g}",
            )
            lower_bound = thresh
        console = Console(file=StringIO(), force_terminal=True, width=80)
        console.print(table)
        return console.file.getvalue()


class GasStorage:
    """A class to represent a gas storage facility for valuation purposes.

    Attributes:
        capacity (float): The maximum storage capacity in cubic meters.

    """

    def __init__(
        self,
        capacity: float,
        injection_withdrawal_curve: InjectionWithdrawalCurve,
    ) -> None:
        self._capacity = capacity
        self._injection_withdrawal_curve = injection_withdrawal_curve

    @property
    def capacity(self) -> float:
        """Return the maximum storage capacity."""
        return self._capacity

    @property
    def max_injection_rate(self) -> float:
        """Return the maximum injection rate across all bins."""
        return self._injection_withdrawal_curve.max_injection_rate

    @property
    def max_withdrawal_rate(self) -> float:
        """Return the maximum withdrawal rate across all bins."""
        return self._injection_withdrawal_curve.max_withdrawal_rate

    def __repr__(self) -> str:
        return (
            f"GasStorage(capacity={self.capacity}, "
            f"injection_withdrawal_curve={self._injection_withdrawal_curve})"
        )

    def __str__(self) -> str:
        # TODO: adjust after storage complete
        return (
            f"GasStorage(capacity={self.capacity}, "
            f"injection_withdrawal_curve=\n{self._injection_withdrawal_curve})"
        )
