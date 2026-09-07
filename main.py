import numpy as np
from py_gas_storage_valuation.data.storage import InjectionWithdrawalCurve, GasStorage


def main():
    # Example usage of the GasStorage and InjectionWithdrawalCurve classes
    bin_upper_thresholds = np.array([0.25, 0.5, 0.75, 1.0])
    injection_rates = np.array([0.1, 0.08, 0.05, 0.02])
    withdrawal_rates = np.array([0.2, 0.15, 0.1, 0.05])

    injection_withdrawal_curve = InjectionWithdrawalCurve(
        bin_upper_thresholds=bin_upper_thresholds,
        injection_rates=injection_rates,
        withdrawal_rates=withdrawal_rates,
    )

    storage_capacity = 100_000.0
    gas_storage = GasStorage(storage_capacity, injection_withdrawal_curve, "Y")

    print(gas_storage)


if __name__ == "__main__":
    main()
