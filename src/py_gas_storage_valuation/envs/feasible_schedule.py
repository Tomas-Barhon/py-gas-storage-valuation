"""Decode storage actions into terminal-feasible physical schedules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np

from py_gas_storage_valuation.data.storage import GasStorage

type InventoryIntervals = tuple[tuple[float, float], ...]
type ReachableInventories = tuple[InventoryIntervals, ...]


@dataclass(frozen=True, slots=True)
class FeasibleSchedule:
    """Physical flows in MWh, indexed by absolute delivery month.

    ``flows`` includes zeroes for expired months. Creating a schedule must
    not change the storage's actual inventory; the environment executes only
    the current month's flow after using the schedule for futures positions.
    """

    flows: np.ndarray

    @staticmethod
    def precompute_reachable_inventories(
        storage: GasStorage, horizon: int
    ) -> ReachableInventories:
        """Return inventories that can be emptied in each number of months.

        Index 0 is ``((0.0, 0.0),)``; index ``n`` contains merged inventory
        intervals from which zero is reachable with ``n`` monthly flows.
        Compute once for each storage configuration, not on every action.

        TODO: work backward through the injection/withdrawal rate bins,
        intersect predecessor intervals with each bin, then merge overlaps.
        Keep interval sets: a single upper bound can exclude feasible states
        or include infeasible ones when rates change between bins.
        """
        raise NotImplementedError

    @staticmethod
    def _project_inventory(
        requested: float,
        minimum: float,
        maximum: float,
        reachable: InventoryIntervals,
    ) -> float:
        """Choose a feasible next inventory closest to ``requested``.

        TODO: intersect [minimum, maximum] with the reachable intervals,
        project onto that intersection, and reject an empty intersection.
        """
        raise NotImplementedError

    @classmethod
    def from_action(
        cls,
        action: np.ndarray,
        *,
        storage: GasStorage,
        current_step: int,
        reachable_inventories: ReachableInventories,
    ) -> Self:
        """Decode remaining-month multipliers without mutating ``storage``.

        ``reachable_inventories[n]`` refers to the number of months still
        available AFTER the proposed flow. The environment passes its own
        storage instance and reuses precomputed reachable intervals.
        """
        horizon = len(reachable_inventories) - 1
        if action.shape != (horizon,):
            raise ValueError(f"Action must have shape ({horizon},).")
        if not 0 <= current_step < horizon:
            raise ValueError("current_step must be within the horizon.")

        flows = np.zeros(horizon, dtype=np.float64)
        inventory = float(storage._current_inventory)
        curve = storage._injection_withdrawal_curve

        for month in range(current_step, horizon):
            max_injection = curve.get_max_flow(inventory, storage.capacity)
            max_withdrawal = curve.get_max_flow(
                inventory, storage.capacity, injection=False
            )
            multiplier = float(action[month])
            proposed_flow = multiplier * (
                max_injection if multiplier >= 0 else max_withdrawal
            )
            next_inventory = cls._project_inventory(
                requested=inventory + proposed_flow,
                minimum=inventory - max_withdrawal,
                maximum=inventory + max_injection,
                reachable=reachable_inventories[horizon - month - 1],
            )
            flows[month] = next_inventory - inventory
            inventory = next_inventory

        return cls(flows=flows)
