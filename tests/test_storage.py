import numpy as np
import pytest

from py_gas_storage_valuation.data.storage import (
    GasStorage,
    InjectionWithdrawalCurve,
)


def test_gas_storage_inject():
    curve = InjectionWithdrawalCurve(
        bin_upper_thresholds=np.array([0.25, 0.5, 0.75, 1.0]),
        injection_rates=np.array([0.1, 0.08, 0.05, 0.02]),
        withdrawal_rates=np.array([0.2, 0.15, 0.1, 0.05]),
    )
    storage = GasStorage(100_000.0, curve, "Y")
    storage.inject(10_000.0)
    assert storage._current_inventory == 10_000.0


def test_gas_storage_capacity():
    curve = InjectionWithdrawalCurve(
        bin_upper_thresholds=np.array([1.0]),
        injection_rates=np.array([0.5]),
        withdrawal_rates=np.array([0.5]),
    )
    storage = GasStorage(50_000.0, curve, "Y")
    assert storage.capacity == 50_000.0


def test_storage_flows_use_capacity_and_inventory_limits():
    curve = InjectionWithdrawalCurve(
        bin_upper_thresholds=np.array([1.0]),
        injection_rates=np.array([0.5]),
        withdrawal_rates=np.array([0.5]),
    )
    storage = GasStorage(100.0, curve, "Y")

    assert curve.get_max_flow(0.0, 100.0, injection=False) == 0.0
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
    curve = InjectionWithdrawalCurve(
        bin_upper_thresholds=np.array([0.5, 1.0]),
        injection_rates=np.array([0.5, 0.25]),
        withdrawal_rates=np.array([0.25, 0.5]),
    )
    storage = GasStorage(100.0, curve, "Y")

    storage.inject(50.0)
    assert curve.get_max_flow(50.0, 100.0) == 25.0
    assert curve.get_max_flow(50.0, 100.0, injection=False) == 50.0
    storage.inject(25.0)
    assert storage._current_inventory == 75.0
    with pytest.raises(ValueError):
        storage.inject(26.0)


@pytest.mark.parametrize("amount", [-1.0, float("nan"), float("inf")])
def test_storage_rejects_invalid_flow(amount):
    curve = InjectionWithdrawalCurve(
        bin_upper_thresholds=np.array([1.0]),
        injection_rates=np.array([0.5]),
        withdrawal_rates=np.array([0.5]),
    )
    storage = GasStorage(100.0, curve, "Y")

    with pytest.raises(ValueError):
        storage.inject(amount)
    with pytest.raises(ValueError):
        storage.withdraw(amount)
    assert storage._current_inventory == 0.0
