"""Decode storage actions into terminal-feasible physical schedules."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import isfinite, nextafter
from typing import Self

import numpy as np

from py_gas_storage_valuation.data.storage import (
    GasStorage,
    StorageOperatingRights,
)

type InventoryIntervals = tuple[tuple[float, float], ...]
type ReachableInventories = tuple[InventoryIntervals, ...]
type RightsConfiguration = tuple[
    float,
    float,
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
]

# April through March in a non-leap storage year; override for leap years.
DEFAULT_MONTH_DAYS = (30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31)


def _merge_intervals(
    intervals: list[tuple[float, float]],
) -> InventoryIntervals:
    """Union closed intervals, including adjacent floating-point endpoints."""
    merged: list[tuple[float, float]] = []
    for lower, upper in sorted(intervals):
        if merged and lower <= nextafter(merged[-1][1], float("inf")):
            merged[-1] = (merged[-1][0], max(merged[-1][1], upper))
        else:
            merged.append((lower, upper))
    return tuple(merged)


@dataclass(frozen=True, slots=True)
class FeasibleSchedule:
    """Physical flows in MWh, indexed by absolute delivery month.

    ``flows`` includes zeroes for expired months. Creating a schedule must
    not change the storage's actual inventory; the environment executes only
    the current month's flow after using the schedule for futures positions.
    """

    flows: np.ndarray

    @classmethod
    def precompute_reachable_inventories(
        cls,
        storage: GasStorage,
        days_per_month: tuple[int, ...] = DEFAULT_MONTH_DAYS,
    ) -> ReachableInventories:
        """Return year-end-emptyable inventories at each month boundary.

        ``result[m]`` is the inventory set just before month ``m``;
        ``result[horizon]`` is exactly zero. Equal rights and calendars share
        a cached result across environments in the same process.
        """
        if not days_per_month or any(
            not isinstance(days, int) or days <= 0 for days in days_per_month
        ):
            raise ValueError("Each month must have a positive number of days.")
        if not isfinite(storage.capacity) or storage.capacity <= 0:
            raise ValueError("Storage capacity must be positive and finite.")
        rights = storage._operating_rights
        configuration: RightsConfiguration = (
            float(rights.max_injection_mwh_per_day),
            float(rights.max_withdrawal_mwh_per_day),
            tuple(map(float, rights.injection_bin_upper_thresholds)),
            tuple(map(float, rights.withdrawal_bin_upper_thresholds)),
            tuple(map(float, rights.injection_rates)),
            tuple(map(float, rights.withdrawal_rates)),
        )
        return cls._cached_reachable_inventories(
            float(storage.capacity), configuration, tuple(days_per_month)
        )

    @staticmethod
    @lru_cache(maxsize=32)
    def _cached_reachable_inventories(
        capacity: float,
        configuration: RightsConfiguration,
        days_per_month: tuple[int, ...],
    ) -> ReachableInventories:
        """Compute backward sets once per immutable storage configuration."""
        injection, withdrawal, injection_bins, withdrawal_bins, ir, wr = (
            configuration
        )
        rights = StorageOperatingRights(
            max_injection_mwh_per_day=injection,
            max_withdrawal_mwh_per_day=withdrawal,
            injection_bin_upper_thresholds=np.array(injection_bins),
            withdrawal_bin_upper_thresholds=np.array(withdrawal_bins),
            injection_rates=np.array(ir),
            withdrawal_rates=np.array(wr),
        )
        storage = GasStorage(capacity, rights)
        horizon = len(days_per_month)
        reachable: list[InventoryIntervals] = [()] * (horizon + 1)
        reachable[horizon] = ((0.0, 0.0),)
        for month in range(horizon - 1, -1, -1):
            reachable[month] = FeasibleSchedule._month_predecessors(
                reachable[month + 1],
                storage=storage,
                days=days_per_month[month],
            )
        return tuple(reachable)

    @staticmethod
    def _bins(
        storage: GasStorage, *, injection: bool
    ) -> tuple[tuple[float, float, float], ...]:
        """Return daily MWh limits using the rights' exact bin convention."""
        rights = storage._operating_rights
        thresholds = (
            rights.injection_bin_upper_thresholds
            if injection
            else rights.withdrawal_bin_upper_thresholds
        )
        rates = (
            rights.injection_rates if injection else rights.withdrawal_rates
        )
        nominal = (
            rights.max_injection_mwh_per_day
            if injection
            else rights.max_withdrawal_mwh_per_day
        )
        bins: list[tuple[float, float, float]] = []
        lower = 0.0
        for index, (threshold, rate) in enumerate(
            zip(thresholds, rates, strict=True)
        ):
            boundary = float(threshold) * storage.capacity
            if index < len(thresholds) - 1:
                # Match daily_limits' rounded inventory / capacity comparison.
                while boundary / storage.capacity < threshold:
                    boundary = nextafter(boundary, float("inf"))
                previous = nextafter(boundary, float("-inf"))
                while (
                    previous >= 0 and previous / storage.capacity >= threshold
                ):
                    boundary = previous
                    previous = nextafter(boundary, float("-inf"))
            upper = (
                storage.capacity
                if index == len(thresholds) - 1
                else nextafter(boundary, float("-inf"))
            )
            if lower <= upper:
                bins.append((lower, upper, float(rate) * nominal))
            lower = boundary
        return tuple(bins)

    @staticmethod
    def _daily_predecessors(
        target: InventoryIntervals,
        *,
        bins: tuple[tuple[float, float, float], ...],
        injection: bool,
    ) -> InventoryIntervals:
        """Find starting inventories that can reach the target in one day."""
        predecessors: list[tuple[float, float]] = []
        for bin_lower, bin_upper, limit in bins:
            for target_lower, target_upper in target:
                if injection:
                    lower = max(bin_lower, target_lower - limit)
                    upper = min(bin_upper, target_upper)
                else:
                    lower = max(bin_lower, target_lower)
                    upper = min(bin_upper, target_upper + limit)
                if lower <= upper:
                    predecessors.append((lower, upper))
        return _merge_intervals(predecessors)

    @staticmethod
    def _daily_successors(
        start: InventoryIntervals,
        *,
        bins: tuple[tuple[float, float, float], ...],
        capacity: float,
        injection: bool,
    ) -> InventoryIntervals:
        """Find inventories attainable after one day in one direction."""
        successors: list[tuple[float, float]] = []
        for bin_lower, bin_upper, limit in bins:
            for start_lower, start_upper in start:
                lower = max(bin_lower, start_lower)
                upper = min(bin_upper, start_upper)
                if lower <= upper:
                    if injection:
                        successors.append(
                            (lower, min(capacity, upper + limit))
                        )
                    else:
                        successors.append((max(0.0, lower - limit), upper))
        return _merge_intervals(successors)

    @classmethod
    def _month_predecessors(
        cls,
        target: InventoryIntervals,
        *,
        storage: GasStorage,
        days: int,
    ) -> InventoryIntervals:
        """Find month starts reaching target without reversing flow."""
        predecessors = []
        for injection in (True, False):
            current = target
            bins = cls._bins(storage, injection=injection)
            for _ in range(days):
                current = cls._daily_predecessors(
                    current, bins=bins, injection=injection
                )
            predecessors.extend(current)
        return _merge_intervals(predecessors)

    @classmethod
    def _monthly_reachable_bounds(
        cls,
        inventory: float,
        *,
        storage: GasStorage,
        days: int,
    ) -> tuple[float, float]:
        """Return extreme month-end inventories under daily dispatch."""
        attainable = cls._monthly_reachable_intervals(
            inventory, storage=storage, days=days
        )
        return attainable[0][0], attainable[-1][1]

    @classmethod
    def _monthly_reachable_intervals(
        cls,
        inventory: float,
        *,
        storage: GasStorage,
        days: int,
    ) -> InventoryIntervals:
        """Return month-end inventory sets with no daily flow reversal."""
        if days <= 0:
            raise ValueError("Month must have a positive number of days.")
        if not isfinite(inventory) or not 0 <= inventory <= storage.capacity:
            raise ValueError("Inventory must be between zero and capacity.")
        endpoints: list[tuple[float, float]] = []
        for injection in (False, True):
            current = ((inventory, inventory),)
            bins = cls._bins(storage, injection=injection)
            for _ in range(days):
                current = cls._daily_successors(
                    current,
                    bins=bins,
                    capacity=storage.capacity,
                    injection=injection,
                )
            endpoints.extend(current)
        return _merge_intervals(endpoints)

    @staticmethod
    def _project_inventory(
        requested: float,
        attainable: InventoryIntervals,
        reachable: InventoryIntervals,
    ) -> float:
        """Project onto attainable inventories that can still finish empty."""
        if not isfinite(requested):
            raise ValueError("Inventory request must be finite.")
        candidates = []
        for attainable_lower, attainable_upper in attainable:
            for reachable_lower, reachable_upper in reachable:
                low = max(attainable_lower, reachable_lower)
                high = min(attainable_upper, reachable_upper)
                if low <= high:
                    candidates.append(min(max(requested, low), high))
        if not candidates:
            raise ValueError("No terminal-feasible monthly inventory exists.")
        return min(
            candidates, key=lambda value: (abs(value - requested), value)
        )

    @classmethod
    def from_action(
        cls,
        action: np.ndarray,
        *,
        storage: GasStorage,
        current_step: int,
        reachable_inventories: ReachableInventories,
        days_per_month: tuple[int, ...] = DEFAULT_MONTH_DAYS,
    ) -> Self:
        """Decode remaining-month multipliers without mutating ``storage``.

        ``reachable_inventories[month + 1]`` contains inventories from which
        zero is reachable after month ``month``. Reuse the same storage and
        calendar that were passed to ``precompute_reachable_inventories``.
        """
        if not days_per_month or any(
            not isinstance(days, int) or days <= 0 for days in days_per_month
        ):
            raise ValueError("Each month must have a positive number of days.")
        if not isfinite(storage.capacity) or storage.capacity <= 0:
            raise ValueError("Storage capacity must be positive and finite.")
        horizon = len(days_per_month)
        if len(reachable_inventories) != horizon + 1:
            raise ValueError(
                "Reachability must have one entry per month boundary."
            )
        if action.shape != (horizon,):
            raise ValueError(f"Action must have shape ({horizon},).")
        if not 0 <= current_step < horizon:
            raise ValueError("current_step must be within the horizon.")
        if not np.all(np.isfinite(action)) or np.any(np.abs(action) > 1):
            raise ValueError(
                "Action multipliers must be finite and in [-1, 1]."
            )

        flows = np.zeros(horizon, dtype=np.float64)
        inventory = float(storage._current_inventory)
        if not isfinite(inventory) or not 0 <= inventory <= storage.capacity:
            raise ValueError("Inventory must be between zero and capacity.")

        for month in range(current_step, horizon):
            attainable = cls._monthly_reachable_intervals(
                inventory,
                storage=storage,
                days=days_per_month[month],
            )
            minimum, maximum = attainable[0][0], attainable[-1][1]
            multiplier = float(action[month])
            proposed_flow = multiplier * (
                maximum - inventory if multiplier >= 0 else inventory - minimum
            )
            next_inventory = cls._project_inventory(
                requested=inventory + proposed_flow,
                attainable=attainable,
                reachable=reachable_inventories[month + 1],
            )
            flows[month] = next_inventory - inventory
            inventory = next_inventory

        return cls(flows=flows)
