import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

from py_gas_storage_valuation.data.storage import (
    InjectionWithdrawalCurve,
)
from py_gas_storage_valuation.envs.gas_storage_env import (
    dummy_vectorize_gas_storage_env,
)
from py_gas_storage_valuation.prices.mock_forward_simulator import (
    MockForwardSimulator,
)
from py_gas_storage_valuation.prices.price_buffer import ForwardCurvePathBuffer
from py_gas_storage_valuation.utils.profiling import Timer


def main():
    bin_upper_thresholds = np.array([0.25, 0.5, 0.75, 1.0])
    injection_rates = np.array([0.1, 0.08, 0.05, 0.02])
    withdrawal_rates = np.array([0.2, 0.15, 0.1, 0.05])

    injection_withdrawal_curve = InjectionWithdrawalCurve(
        bin_upper_thresholds=bin_upper_thresholds,
        injection_rates=injection_rates,
        withdrawal_rates=withdrawal_rates,
    )

    simulator = MockForwardSimulator()
    price_buffer = ForwardCurvePathBuffer(simulator)
    storage_kwargs = {
        "_capacity": 100_000.0,
        "_injection_withdrawal_curve": injection_withdrawal_curve,
    }

    vectorized_env = dummy_vectorize_gas_storage_env(
        price_buffer=price_buffer, num_envs=32, **storage_kwargs
    )

    with Timer("Training SAC agent"):
        agent = SAC(
            "MlpPolicy",
            vectorized_env,
            verbose=1,
            tensorboard_log="./run_logs/sac_gas_storage",
        )
        agent.learn(total_timesteps=100_000)


if __name__ == "__main__":
    main()
