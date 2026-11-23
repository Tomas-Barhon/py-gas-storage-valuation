import numpy as np
import pytest

from py_gas_storage_valuation.data.storage import (
    GasStorage,
    StorageOperatingRights,
)
from py_gas_storage_valuation.envs.feasible_schedule import FeasibleSchedule


def _storage(
    *,
    injection: float = 25.0,
    withdrawal: float = 25.0,
    injection_thresholds: tuple[float, ...] = (1.0,),
    injection_rates: tuple[float, ...] = (1.0,),
    withdrawal_thresholds: tuple[float, ...] = (1.0,),
    withdrawal_rates: tuple[float, ...] = (1.0,),
) -> GasStorage:
    rights = StorageOperatingRights(
        max_injection_mwh_per_day=injection,
        max_withdrawal_mwh_per_day=withdrawal,
        injection_bin_upper_thresholds=np.array(injection_thresholds),
        withdrawal_bin_upper_thresholds=np.array(withdrawal_thresholds),
        injection_rates=np.array(injection_rates),
        withdrawal_rates=np.array(withdrawal_rates),
    )
    return GasStorage(100.0, rights)


def test_projection_selects_nearest_viable_interval_and_rejects_gap():
    viable = ((0.0, 20.0), (50.0, 70.0))

    assert (
        FeasibleSchedule._project_inventory(40, ((10.0, 80.0),), viable) == 50
    )
    assert (
        FeasibleSchedule._project_inventory(35, ((10.0, 80.0),), viable) == 20
    )
    with pytest.raises(ValueError, match="No terminal-feasible"):
        FeasibleSchedule._project_inventory(40, ((30.0, 45.0),), viable)


def test_projection_respects_disconnected_attainable_intervals():
    attainable = ((0.0, 20.0), (50.0, 70.0))

    assert (
        FeasibleSchedule._project_inventory(40, attainable, ((10.0, 80.0),))
        == 50
    )
    assert (
        FeasibleSchedule._project_inventory(35, attainable, ((10.0, 80.0),))
        == 20
    )
    with pytest.raises(ValueError, match="No terminal-feasible"):
        FeasibleSchedule._project_inventory(40, attainable, ((25.0, 45.0),))


def test_withdrawal_bin_jump_creates_disconnected_emptyable_inventories():
    storage = _storage(
        withdrawal=100,
        withdrawal_thresholds=(0.5, 1.0),
        withdrawal_rates=(0.25, 0.5),
    )
    viable = FeasibleSchedule.precompute_reachable_inventories(
        storage, days_per_month=(1,)
    )

    assert viable == (((0.0, 25.0), (50.0, 50.0)), ((0.0, 0.0),))
    assert (
        FeasibleSchedule._project_inventory(40, ((0.0, 100.0),), viable[0])
        == 50
    )


def test_threshold_membership_matches_fractional_inventory_lookup():
    capacity = 941774.3846991474
    threshold = 0.6320182474041497
    boundary = threshold * capacity
    assert boundary / capacity < threshold
    rights = StorageOperatingRights(
        max_injection_mwh_per_day=0.0,
        max_withdrawal_mwh_per_day=capacity,
        injection_bin_upper_thresholds=np.array([1.0]),
        withdrawal_bin_upper_thresholds=np.array([threshold, 1.0]),
        injection_rates=np.array([0.0]),
        withdrawal_rates=np.array([0.0, 1.0]),
    )
    storage = GasStorage(capacity, rights)
    viable = FeasibleSchedule.precompute_reachable_inventories(
        storage, days_per_month=(1,)
    )

    with pytest.raises(ValueError, match="No terminal-feasible"):
        FeasibleSchedule._project_inventory(
            boundary, ((boundary, boundary),), viable[0]
        )
    first_fast_inventory = np.nextafter(boundary, np.inf)
    assert rights.daily_limits(first_fast_inventory, capacity)[0] > 0
    assert (
        FeasibleSchedule._project_inventory(
            first_fast_inventory,
            ((first_fast_inventory, first_fast_inventory),),
            viable[0],
        )
        == first_fast_inventory
    )


def test_reachability_cache_shares_configuration_and_tracks_mutations():
    FeasibleSchedule._cached_reachable_inventories.cache_clear()
    storage = _storage(withdrawal_rates=(1.0,))
    another_storage = _storage(withdrawal_rates=(1.0,))
    calendar = (2, 1)

    first = FeasibleSchedule.precompute_reachable_inventories(
        storage, calendar
    )
    assert (
        FeasibleSchedule.precompute_reachable_inventories(
            another_storage, calendar
        )
        is first
    )
    cache_info = FeasibleSchedule._cached_reachable_inventories.cache_info()
    assert cache_info.hits == 1

    another_storage._operating_rights.withdrawal_rates[0] = 0.5
    assert (
        FeasibleSchedule.precompute_reachable_inventories(
            another_storage, calendar
        )
        is not first
    )
    assert (
        FeasibleSchedule.precompute_reachable_inventories(storage, (1, 2))
        is not first
    )
    assert (
        FeasibleSchedule.precompute_reachable_inventories(
            GasStorage(200.0, storage._operating_rights), calendar
        )
        is not first
    )


def test_month_cannot_reverse_daily_flow_to_end_empty():
    storage = _storage()
    viable = FeasibleSchedule.precompute_reachable_inventories(
        storage, days_per_month=(2,)
    )

    # Injecting on day one and withdrawing on day two is not a monthly flow.
    schedule = FeasibleSchedule.from_action(
        np.array([1.0]),
        storage=storage,
        current_step=0,
        reachable_inventories=viable,
        days_per_month=(2,),
    )
    np.testing.assert_array_equal(schedule.flows, [0.0])


def test_daily_bin_crossing_can_outperform_max_rate_greedy():
    storage = _storage(
        injection=20,
        injection_thresholds=(0.5, 1.0),
        injection_rates=(1.0, 0.1),
    )
    minimum, maximum = FeasibleSchedule._monthly_reachable_bounds(
        40.0, storage=storage, days=2
    )

    # Greedy daily injection gives 40 -> 60 -> 62; 40 -> 49 -> 69 works.
    assert minimum == 0.0
    assert maximum >= 69.0


def test_decoder_empties_by_deadline_without_mutating_storage():
    storage = _storage()
    viable = FeasibleSchedule.precompute_reachable_inventories(
        storage, days_per_month=(1, 1, 1)
    )
    schedule = FeasibleSchedule.from_action(
        np.ones(3),
        storage=storage,
        current_step=0,
        reachable_inventories=viable,
        days_per_month=(1, 1, 1),
    )

    np.testing.assert_array_equal(schedule.flows, [25.0, 0.0, -25.0])
    assert storage._current_inventory == 0.0
    assert sum(schedule.flows) == 0.0


def test_decoder_forces_early_withdrawal_and_zeros_expired_slots():
    storage = _storage()
    storage._current_inventory = 50.0
    viable = FeasibleSchedule.precompute_reachable_inventories(
        storage, days_per_month=(1, 1, 1)
    )

    schedule = FeasibleSchedule.from_action(
        np.ones(3),
        storage=storage,
        current_step=1,
        reachable_inventories=viable,
        days_per_month=(1, 1, 1),
    )
    np.testing.assert_array_equal(schedule.flows, [0.0, -25.0, -25.0])
    assert storage._current_inventory == 50.0


def test_impossible_inventory_before_last_month_is_rejected():
    storage = _storage()
    storage._current_inventory = 50.0
    viable = FeasibleSchedule.precompute_reachable_inventories(
        storage, days_per_month=(1,)
    )
    with pytest.raises(ValueError, match="No terminal-feasible"):
        FeasibleSchedule.from_action(
            np.array([0.0]),
            storage=storage,
            current_step=0,
            reachable_inventories=viable,
            days_per_month=(1,),
        )


def test_backward_sets_match_enumerated_daily_dispatch():
    rights = StorageOperatingRights(
        max_injection_mwh_per_day=2.0,
        max_withdrawal_mwh_per_day=5.0,
        injection_bin_upper_thresholds=np.array([0.5, 1.0]),
        withdrawal_bin_upper_thresholds=np.array([0.5, 1.0]),
        injection_rates=np.array([1.0, 0.5]),
        withdrawal_rates=np.array([0.2, 0.8]),
    )
    storage = GasStorage(10.0, rights)
    viable = FeasibleSchedule.precompute_reachable_inventories(
        storage, days_per_month=(2, 2)
    )

    def can_empty(start: int) -> bool:
        states = {start}
        for _ in range(2):
            month_end = set()
            for beginning in states:
                for injection in (False, True):
                    daily = {beginning}
                    for _ in range(2):
                        following = set()
                        for inventory in daily:
                            withdrawal, injection_limit = rights.daily_limits(
                                inventory, 10.0
                            )
                            limit = (
                                injection_limit if injection else withdrawal
                            )
                            for amount in range(int(limit) + 1):
                                following.add(
                                    inventory + amount
                                    if injection
                                    else inventory - amount
                                )
                        daily = following
                    month_end.update(daily)
            states = month_end
        return 0 in states

    for starting_inventory in range(11):
        in_precomputed_set = any(
            lower <= starting_inventory <= upper for lower, upper in viable[0]
        )
        assert in_precomputed_set == can_empty(starting_inventory)
