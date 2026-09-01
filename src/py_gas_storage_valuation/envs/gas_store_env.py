"This module contains the GasStorageEnv class, which is a custom gymnasium compatible environment for gas storage valuation."

import gymnasium as gym
from py_gas_storage_valuation.data.storage import GasStorage
from py_gas_storage_valuation.prices.base_price_simulator import BasePriceSimulator
import numpy as np


class GasStorageEnv(gym.Env):
    def __init__(
        self,
        gas_storage: GasStorage,
        price_model: BasePriceSimulator,
        max_steps: int = 12,
        injection_costs: float = 0.0,
        withdrawal_costs: float = 0.0,
    ):
        super().__init__()
        self._gas_storage = gas_storage
        self._price_model = price_model
        self._current_step = 0
        self._max_steps = max_steps

        self.action_space = gym.spaces.Box(
            low=-self._gas_storage.max_withdrawal_rate,
            high=self._gas_storage.max_injection_rate,
            shape=(1,),
            dtype=float,
        )

        # (sine,cosine, inventory, F1, F2, ..., F12)
        low = np.array([-1.0, -1.0, 0.0] + [-np.inf] * 12, dtype=np.float32)
        high = np.array(
            [1.0, 1.0, self._gas_storage.capacity] + [np.inf] * 12, dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_step = 0
        self._gas_storage = GasStorage(
            capacity=self._gas_storage.capacity,
            injection_withdrawal_curve=self._gas_storage._injection_withdrawal_curve,
            storage_period=self._gas_storage.storage_period,
        )

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if action > 0:
            self._gas_storage.inject(action)
        else:
            self._gas_storage.withdraw(-action)

        current_price = self._price_model.step(self.current_step)
        self._current_step += 1

        # TODO: implement reward
        reward = ...

        done = self.current_step >= self._max_steps

        # TODO: implement observation
        observation = ...

        return observation, reward, done, {}

    # TODO: implement rendering that will show the injection withdrawal curve
    # history and current state and action
    def render(self): ...
