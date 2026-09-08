"This module contains the GasStorageEnv class, which is a custom gymnasium compatible environment for gas storage valuation."

import gymnasium as gym
from numpy.random import laplace
from py_gas_storage_valuation.data.storage import GasStorage
from py_gas_storage_valuation.prices.base_price_simulator import BasePriceSimulator
from py_gas_storage_valuation.prices.price_buffer import ForwardCurvePathBuffer
import numpy as np


class GasStorageEnv(gym.Env):
    def __init__(
        self,
        gas_storage: GasStorage,
        max_steps: int = 12,
        injection_costs: float = 0.0,
        withdrawal_costs: float = 0.0,
    ):
        super().__init__()
        self._gas_storage = gas_storage
        self._current_step = 0
        self._max_steps = max_steps
        self._injection_costs = injection_costs
        self._withdrawal_costs = withdrawal_costs
        self._forward_curve = None

        self.action_space = gym.spaces.Box(
            low=-self._gas_storage.max_withdrawal_rate,
            high=self._gas_storage.max_injection_rate,
            shape=(1,),
            dtype=np.float32,
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

    def reset(self, *, seed=None, options=None):
        # the forward curve is passed during reset
        super().reset(seed=seed)
        if options is None or "forward_curve" not in options:
            raise ValueError("A forward curve must be provided.")

        # NOTE: consider input validation
        self._forward_curve = options["forward_curve"]

        self._current_step = 0

        self._gas_storage = GasStorage(
            self._gas_storage._capacity,
            self._gas_storage._injection_withdrawal_curve,
            self._gas_storage._storage_period,
        )

        observation = np.concatenate(
            [
                self._encode_seasonality(),
                np.array([self._gas_storage._current_inventory], dtype=np.float32),
                self._forward_curve[self._current_step],
            ]
        )

        return observation, {}

    def _evaluate_action_costs(self, action):
        costs = 0.0
        if action > 0:
            costs += action * self._injection_costs
        else:
            costs += -action * self._withdrawal_costs
        return costs

    def _encode_seasonality(self) -> np.ndarray:
        """_encode_seasonality encodes the current step as a sine and cosine value to capture seasonality effects."""
        seasonality = 2 * np.pi * self._current_step / self._max_steps
        return np.array([np.sin(seasonality), np.cos(seasonality)], dtype=np.float32)

    def _calculate_cash_flow(self, action, current_forward):
        cash_flow = -action * current_forward[self._current_step]  # Buying gas
        return cash_flow

    def step(self, action):
        # inject or withdraw current t
        action = np.float32(
            np.clip(action, self.action_space.low, self.action_space.high)[0]
        )
        # injection and withdrawal costs
        costs = self._evaluate_action_costs(action)

        cash_flow = self._calculate_cash_flow(
            action, self._forward_curve[self._current_step]
        )

        if action > 0:
            self._gas_storage.inject(action)
        else:
            self._gas_storage.withdraw(-action)

        self._current_step += 1

        # TODO: implementnt reward
        reward = cash_flow - costs

        done = self._current_step >= self._max_steps - 1

        # TODO: implement observation
        observation = np.concatenate(
            [
                self._encode_seasonality(),
                [self._gas_storage._current_inventory],
                self._forward_curve[self._current_step],
            ]
        )

        return observation, reward, done, {}
