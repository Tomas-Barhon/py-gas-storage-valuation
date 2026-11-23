import numpy as np
import pytest

from py_gas_storage_valuation.data.storage import (
    GasStorage,
    StorageOperatingRights,
)


def test_gas_storage_inject():
    rights = StorageOperatingRights(
        max_injection_mwh_per_day=100_000.0,
        max_withdrawal_mwh_per_day=100_000.0,
        injection_bin_upper_thresholds=np.array([0.25, 0.5, 0.75, 1.0]),
        withdrawal_bin_upper_thresholds=np.array([0.25, 0.5, 0.75, 1.0]),
        injection_rates=np.array([0.1, 0.08, 0.05, 0.02]),
        withdrawal_rates=np.array([0.2, 0.15, 0.1, 0.05]),
    )
    storage = GasStorage(100_000.0, rights, "Y")
    storage.inject(10_000.0)
    assert storage._current_inventory == 10_000.0


def test_gas_storage_capacity():
    rights = StorageOperatingRights(
        max_injection_mwh_per_day=100.0,
        max_withdrawal_mwh_per_day=100.0,
        injection_bin_upper_thresholds=np.array([1.0]),
        withdrawal_bin_upper_thresholds=np.array([1.0]),
        injection_rates=np.array([0.5]),
        withdrawal_rates=np.array([0.5]),
    )
    storage = GasStorage(50_000.0, rights, "Y")
    assert storage.capacity == 50_000.0


def test_storage_flows_use_capacity_and_inventory_limits():
    rights = StorageOperatingRights(
        max_injection_mwh_per_day=100.0,
        max_withdrawal_mwh_per_day=100.0,
        injection_bin_upper_thresholds=np.array([1.0]),
        withdrawal_bin_upper_thresholds=np.array([1.0]),
        injection_rates=np.array([0.5]),
        withdrawal_rates=np.array([0.5]),
    )
    storage = GasStorage(100.0, rights, "Y")

    assert rights.daily_limits(0.0, 100.0)[0] == 0.0
    storage.inject(50.0)
    storage.inject(50.0)
    assert storage._current_inventory == 100.0
    with pytest.raises(ValueError):
        storage.inject(1.0)
    storage.withdraw(50.0)
    storage.withdraw(50.0)
    assert storage._current_inventory == 0.0
    with pytest.raises(ValueError):
        storage.withdraw(1.0)


def test_storage_rate_uses_fractional_inventory_bin():
    rights = StorageOperatingRights(
        max_injection_mwh_per_day=100.0,
        max_withdrawal_mwh_per_day=100.0,
        injection_bin_upper_thresholds=np.array([0.5, 1.0]),
        withdrawal_bin_upper_thresholds=np.array([0.5, 1.0]),
        injection_rates=np.array([0.5, 0.25]),
        withdrawal_rates=np.array([0.25, 0.5]),
    )
    storage = GasStorage(100.0, rights, "Y")

    storage.inject(50.0)
    assert rights.daily_limits(50.0, 100.0) == (50.0, 25.0)
    storage.inject(25.0)
    assert storage._current_inventory == 75.0
    with pytest.raises(ValueError):
        storage.inject(26.0)


def test_injection_and_withdrawal_use_independent_inventory_bins():
    rights = StorageOperatingRights(
        max_injection_mwh_per_day=100.0,
        max_withdrawal_mwh_per_day=100.0,
        injection_bin_upper_thresholds=np.array([0.25, 0.5, 1.0]),
        withdrawal_bin_upper_thresholds=np.array([0.4, 1.0]),
        injection_rates=np.array([0.8, 0.6, 0.2]),
        withdrawal_rates=np.array([0.1, 0.7]),
    )

    assert rights.get_injection_rate(0.3) == 0.6
    assert rights.get_withdrawal_rate(0.3) == 0.1
    assert rights.get_injection_rate(0.4) == 0.6
    assert rights.get_withdrawal_rate(0.4) == 0.7
    assert rights.daily_limits(40.0, 100.0) == (40.0, 60.0)


def test_daily_limits_scale_nominal_rights_not_storage_capacity():
    rights = StorageOperatingRights(
        max_injection_mwh_per_day=80.0,
        max_withdrawal_mwh_per_day=40.0,
        injection_bin_upper_thresholds=np.array([1.0]),
        withdrawal_bin_upper_thresholds=np.array([1.0]),
        injection_rates=np.array([0.5]),
        withdrawal_rates=np.array([0.5]),
    )

    assert rights.daily_limits(100.0, 1_000.0) == (20.0, 40.0)
    assert rights.daily_limits(990.0, 1_000.0) == (20.0, 10.0)


def test_monthly_flow_updates_inventory_without_weakening_daily_rights():
    rights = StorageOperatingRights(
        max_injection_mwh_per_day=5.0,
        max_withdrawal_mwh_per_day=5.0,
        injection_bin_upper_thresholds=np.array([1.0]),
        withdrawal_bin_upper_thresholds=np.array([1.0]),
        injection_rates=np.array([1.0]),
        withdrawal_rates=np.array([1.0]),
    )
    storage = GasStorage(100.0, rights)

    storage.apply_monthly_flow(100.0)
    assert storage._current_inventory == 100.0
    with pytest.raises(ValueError, match="daily rights"):
        storage.withdraw(100.0)
    with pytest.raises(ValueError, match="capacity"):
        storage.apply_monthly_flow(1.0)
    with pytest.raises(ValueError, match="capacity"):
        storage.apply_monthly_flow(-101.0)
    with pytest.raises(ValueError, match="capacity"):
        storage.apply_monthly_flow(float("nan"))
    assert storage._current_inventory == 100.0
    storage.apply_monthly_flow(-100.0)
    assert storage._current_inventory == 0.0


def test_curve_rejects_mismatched_directional_bins():
    with pytest.raises(ValueError, match="withdrawal thresholds and rates"):
        StorageOperatingRights(
            max_injection_mwh_per_day=100.0,
            max_withdrawal_mwh_per_day=100.0,
            injection_bin_upper_thresholds=np.array([0.5, 1.0]),
            withdrawal_bin_upper_thresholds=np.array([0.4, 1.0]),
            injection_rates=np.array([0.4, 0.2]),
            withdrawal_rates=np.array([0.3]),
        )


@pytest.mark.parametrize("amount", [-1.0, float("nan"), float("inf")])
def test_storage_rejects_invalid_flow(amount):
    rights = StorageOperatingRights(
        max_injection_mwh_per_day=100.0,
        max_withdrawal_mwh_per_day=100.0,
        injection_bin_upper_thresholds=np.array([1.0]),
        withdrawal_bin_upper_thresholds=np.array([1.0]),
        injection_rates=np.array([0.5]),
        withdrawal_rates=np.array([0.5]),
    )
    storage = GasStorage(100.0, rights, "Y")

    with pytest.raises(ValueError):
        storage.inject(amount)
    with pytest.raises(ValueError):
        storage.withdraw(amount)
    assert storage._current_inventory == 0.0
