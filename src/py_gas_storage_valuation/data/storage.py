from dataclasses import dataclass
from io import StringIO
from typing import Literal

import numpy as np
from rich.console import Console
from rich.table import Table


@dataclass(frozen=True, slots=True)
class StorageOperatingRights:
    """Nominal daily MWh rights scaled by inventory-dependent rate curves."""

    max_injection_mwh_per_day: float
    max_withdrawal_mwh_per_day: float
    injection_bin_upper_thresholds: np.ndarray
    withdrawal_bin_upper_thresholds: np.ndarray
    injection_rates: np.ndarray
    withdrawal_rates: np.ndarray

    def __post_init__(self):
        if not all(
            np.isfinite(rate) and rate >= 0
            for rate in (
                self.max_injection_mwh_per_day,
                self.max_withdrawal_mwh_per_day,
            )
        ):
            raise ValueError(
                "Daily contracted rates must be finite and nonnegative."
            )
        for direction in ("injection", "withdrawal"):
            thresholds = getattr(self, f"{direction}_bin_upper_thresholds")
            rates = getattr(self, f"{direction}_rates")
            if not len(thresholds) or len(thresholds) != len(rates):
                raise ValueError(
                    f"{direction} thresholds and rates must have the same "
                    "nonzero length."
                )
            if not np.all(np.isfinite(thresholds)) or not np.all(
                (thresholds > 0) & (thresholds <= 1)
            ):
                raise ValueError(
                    f"{direction} thresholds must be finite and in (0, 1]."
                )
            if not np.all(np.diff(thresholds) > 0) or thresholds[-1] != 1:
                raise ValueError(
                    f"{direction} thresholds must be strictly increasing "
                    "and end at 1."
                )
            if not np.all(np.isfinite(rates)) or np.any(
                (rates < 0) | (rates > 1)
            ):
                raise ValueError("Rates must be finite and between 0 and 1.")

    def _get_rate(self, value: float, injection: bool = True) -> float:
        """Return rate for the bin containing value."""
        thresholds = (
            self.injection_bin_upper_thresholds
            if injection
            else self.withdrawal_bin_upper_thresholds
        )
        idx = np.searchsorted(thresholds, value, side="right")
        if idx == len(thresholds):
            idx -= 1
        return (
            self.injection_rates[idx]
            if injection
            else self.withdrawal_rates[idx]
        )

    def get_injection_rate(self, value: float) -> float:
        """Return injection rate for the bin containing value."""
        return self._get_rate(value, injection=True)

    def get_withdrawal_rate(self, value: float) -> float:
        """Return withdrawal rate for the bin containing value."""
        return self._get_rate(value, injection=False)

    def daily_limits(
        self, inventory: float, capacity: float
    ) -> tuple[float, float]:
        """Return (maximum withdrawal, injection) for one day in MWh."""
        if not np.isfinite(capacity) or capacity <= 0:
            raise ValueError("Capacity must be positive and finite.")
        if not np.isfinite(inventory) or not 0 <= inventory <= capacity:
            raise ValueError("Inventory must be between zero and capacity.")
        fraction = inventory / capacity
        withdrawal = min(
            inventory,
            self.get_withdrawal_rate(fraction)
            * self.max_withdrawal_mwh_per_day,
        )
        injection = min(
            capacity - inventory,
            self.get_injection_rate(fraction) * self.max_injection_mwh_per_day,
        )
        return float(withdrawal), float(injection)

    @property
    def num_bins(self) -> int:
        """Return the total number of injection and withdrawal bins."""
        return len(self.injection_rates) + len(self.withdrawal_rates)

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
            "StorageOperatingRights("
            f"max_injection_mwh_per_day={self.max_injection_mwh_per_day}, "
            f"max_withdrawal_mwh_per_day={self.max_withdrawal_mwh_per_day}, "
            "injection_bin_upper_thresholds="
            f"{self.injection_bin_upper_thresholds}, "
            "withdrawal_bin_upper_thresholds="
            f"{self.withdrawal_bin_upper_thresholds}, "
            f"injection_rates={self.injection_rates}, "
            f"withdrawal_rates={self.withdrawal_rates})"
        )

    def __str__(self) -> str:
        tables = []
        for direction in ("injection", "withdrawal"):
            table = Table(title=f"{direction.title()} Curve")
            table.add_column("Lower Threshold", justify="right")
            table.add_column("Upper Threshold", justify="right")
            table.add_column("Rate", justify="right")

            lower_bound = 0.0
            thresholds = getattr(self, f"{direction}_bin_upper_thresholds")
            rates = getattr(self, f"{direction}_rates")
            for threshold, rate in zip(thresholds, rates, strict=True):
                table.add_row(
                    f"{lower_bound:.4g}",
                    f"{threshold:.4g}",
                    f"{rate:.4g}",
                )
                lower_bound = threshold
            tables.append(table)
        buffer = StringIO()
        console = Console(file=buffer, force_terminal=True, width=80)
        for table in tables:
            console.print(table)
        return buffer.getvalue()


@dataclass(slots=True)
class GasStorage:
    """A class to represent a gas storage facility for valuation purposes.

    Attributes:
        capacity (float): The maximum storage capacity in MWh.

    """

    _capacity: float
    _operating_rights: StorageOperatingRights
    _storage_period: Literal["M", "Y"] = "Y"
    _current_inventory: float = 0.0

    @property
    def capacity(self) -> float:
        return self._capacity

    @property
    def max_injection_rate(self) -> float:
        """Return the maximum injection rate across all bins."""
        return self._operating_rights.max_injection_rate

    @property
    def max_withdrawal_rate(self) -> float:
        """Return the maximum withdrawal rate across all bins."""
        return self._operating_rights.max_withdrawal_rate

    def __repr__(self) -> str:
        return (
            f"GasStorage(capacity={self._capacity}, "
            f"operating_rights={self._operating_rights})"
        )

    def __str__(self) -> str:
        # TODO: adjust after storage complete
        return (
            f"GasStorage(capacity={self._capacity}, "
            f"operating_rights=\n{self._operating_rights})"
        )

    def inject(self, amount):
        _, limit = self._operating_rights.daily_limits(
            self._current_inventory, self._capacity
        )
        if np.isfinite(amount) and 0 <= amount <= limit:
            self._current_inventory += amount
        else:
            raise ValueError("Cannot inject more than daily rights allow.")

    def withdraw(self, amount):
        limit, _ = self._operating_rights.daily_limits(
            self._current_inventory, self._capacity
        )
        if np.isfinite(amount) and 0 <= amount <= limit:
            self._current_inventory -= amount
        else:
            raise ValueError("Cannot withdraw more than daily rights allow.")

    def apply_monthly_flow(self, flow_mwh: float) -> None:
        """Apply a signed monthly flow certified by the schedule decoder."""
        next_inventory = self._current_inventory + flow_mwh
        if (
            not np.isfinite(flow_mwh)
            or not 0 <= next_inventory <= self._capacity
        ):
            raise ValueError("Monthly flow exceeds storage capacity bounds.")
        self._current_inventory = next_inventory
