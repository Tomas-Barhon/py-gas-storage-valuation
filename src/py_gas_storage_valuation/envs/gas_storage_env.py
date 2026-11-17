"""Gas storage valuation environment for Gymnasium."""

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from py_gas_storage_valuation.data.storage import GasStorage
from py_gas_storage_valuation.prices.price_buffer import ForwardCurvePathBuffer


def build_gas_storage_env(
    price_buffer: ForwardCurvePathBuffer,
    max_episode_steps: int,
    **storage_kwargs,
):
    def _init():
        env = gym.make(
            "GasStorage-v0",
            gas_storage=GasStorage(
                **storage_kwargs
            ),  # each env has own storage
            price_buffer=price_buffer,
            max_steps=max_episode_steps,
            max_episode_steps=max_episode_steps,
        )
        return env

    return _init


def dummy_vectorize_gas_storage_env(
    price_buffer: ForwardCurvePathBuffer, num_envs: int, **storage_kwargs
):
    """Vectorize the gas storage environment using DummyVecEnv."""
    return DummyVecEnv(
        [
            build_gas_storage_env(
                price_buffer, max_episode_steps=12, **storage_kwargs
            )
            for _ in range(num_envs)
        ]
    )


class GasStorageEnv(gym.Env):
    def __init__(
        self,
        gas_storage: GasStorage,
        price_buffer: ForwardCurvePathBuffer,
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
        self._price_buffer = price_buffer
        self._futures_positions = np.zeros(
            (self._max_steps,), dtype=np.float32
        )

        # remaining-month schedule converted to absolutes internally
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(self._max_steps,),
            dtype=np.float32,
        )

        # (sine,cosine, inventory, F1, F2, ..., F12)
        low = np.array(
            [-1.0, -1.0, 0.0] + [-np.inf] * self._max_steps, dtype=np.float32
        )
        high = np.array(
            [1.0, 1.0, self._gas_storage.capacity]
            + [np.inf] * self._max_steps,
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        # the forward curve is passed during reset
        super().reset(seed=seed)

        # NOTE: consider input validation
        self._forward_curve = self._price_buffer.get_path()

        self._current_step = 0

        self._gas_storage = GasStorage(
            self._gas_storage._capacity,
            self._gas_storage._injection_withdrawal_curve,
            self._gas_storage._storage_period,
        )

        observation = self._concat_observation()
        return observation, {}

    def _evaluate_action_costs(self, schedule):
        costs = 0.0
        if schedule[self._current_step] > 0:
            costs += self._injection_costs * schedule[self._current_step]
        elif schedule[self._current_step] < 0:
            costs -= self._withdrawal_costs * schedule[self._current_step]
        return costs

    def _encode_seasonality(self) -> np.ndarray:
        """Encode the current step as sin/cos for seasonality."""
        seasonality = 2 * np.pi * self._current_step / self._max_steps
        return np.array(
            [np.sin(seasonality), np.cos(seasonality)], dtype=np.float32
        )

    def _concat_observation(self):
        seasonality = self._encode_seasonality()
        inventory = np.array(
            [self._gas_storage._current_inventory], dtype=np.float32
        )
        if self._forward_curve is not None:
            forward_curve = self._forward_curve[self._current_step]
        else:
            raise ValueError(
                "Forward curve is not set. Call reset() before step()."
            )

        observation = np.concatenate(
            [seasonality, inventory.flatten(), forward_curve]
        )
        return observation

    def _calculate_cash_flow(self, schedule):
        # calc cash flows from physical injection/withdrawal
        current_action = schedule[self._current_step]
        if self._forward_curve is None:
            raise ValueError(
                "Forward curve is not set. Call reset() before step()."
            )
        return (
            self._gas_storage._current_inventory
            * current_action
            * self._forward_curve[self._current_step]
        )

    def _calculate_futures_pnl_mtm(self):
        if self._current_step == 0:
            return 0.0

        if self._forward_curve is not None:
            # calc mtm changes in non expired futures
            price_diffs = (
                self._forward_curve[self._current_step]
                - self._forward_curve[self._current_step - 1]
            )
            futures_pnl = np.sum(
                self._futures_positions[self._current_step :]
                * price_diffs[self._current_step :]
            )

        else:
            raise ValueError(
                "Forward curve is not set. Call reset() before step()."
            )

        return futures_pnl

    def _rebalance_futures_positions(self, schedule):
        # we already assume passed actions are zeroed out
        self._futures_positions = schedule
        return self

    def _decode_schedule(self, action):
        # decode the action into a feasible schedule
        # we will zero out passed time steps here
        schedule = np.zeros_like(action, dtype=np.float32)

        return schedule

    def _settle_current_flows(self, schedule):
        phys_cash_flow = self._calculate_cash_flow(schedule)
        costs = self._evaluate_action_costs(schedule)
        futures_pnl = self._calculate_futures_pnl_mtm()

        return phys_cash_flow, costs, futures_pnl

    def step(self, action):
        # settle money
        schedule = self._decode_schedule(action)

        self._rebalance_futures_positions(schedule)

        phys_cf, costs, futures_pnl = self._settle_current_flows(schedule)

        if schedule[self._current_step] > 0:
            self._gas_storage.inject(schedule[self._current_step])
        else:
            self._gas_storage.withdraw(-schedule[self._current_step])

        observation = self._concat_observation()
        self._current_step += 1

        reward = np.array([phys_cf - costs + futures_pnl], dtype=np.float32)
        truncated = self._current_step >= self._max_steps

        # env never terminates, only truncated when max_steps is reached
        return (
            observation,
            reward[0],
            False,
            truncated,
            {"phys_cf": phys_cf, "costs": costs, "futures_pnl": futures_pnl},
        )
