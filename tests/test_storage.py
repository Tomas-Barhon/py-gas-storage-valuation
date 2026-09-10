import numpy as np

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
    storage.inject(0.05)
    assert storage._current_inventory == 0.05


def test_gas_storage_capacity():
    curve = InjectionWithdrawalCurve(
        bin_upper_thresholds=np.array([1.0]),
        injection_rates=np.array([0.5]),
        withdrawal_rates=np.array([0.5]),
    )
    storage = GasStorage(50_000.0, curve, "Y")
    assert storage.capacity == 50_000.0
