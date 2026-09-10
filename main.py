import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

from py_gas_storage_valuation.data.storage import (
    GasStorage,
    InjectionWithdrawalCurve,
)
from py_gas_storage_valuation.prices.mock_forward_simulator import (
    MockForwardSimulator,
)
from py_gas_storage_valuation.prices.price_buffer import ForwardCurvePathBuffer


def main():
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

    simulator = MockForwardSimulator()
    price_buffer = ForwardCurvePathBuffer(simulator)

    env = gym.make(
        "GasStorage-v0", gas_storage=gas_storage, max_episode_steps=12
    )

    agent = SAC("MlpPolicy", env, verbose=1)
    agent.learn(total_timesteps=10000)

    obs, info = env.reset(options={"forward_curve": price_buffer.get_path()})
    done = False
    while not done:
        action, _ = agent.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step = env._current_step
        inv = env._gas_storage._current_inventory
        print(
            f"Step: {step}, Action: {action},"
            f" Reward: {reward}, Inventory: {inv}"
        )


if __name__ == "__main__":
    main()
