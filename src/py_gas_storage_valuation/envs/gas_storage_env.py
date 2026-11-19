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
        self._accumulated_pnl = 0.0
        # remaining-month schedule converted to absolutes internally
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(self._max_steps,),
            dtype=np.float32,
        )

        # (
        #   sine, cosine, inventory, accumulated_pnl
        #   F1, F2, ..., F12,
        #   F1-pos, F2-pos, ..., F12-pos
        # )
        low = np.array(
            [-1.0, -1.0, 0.0, -np.inf] + [-np.inf] * (self._max_steps * 2),
            dtype=np.float32,
        )
        high = np.array(
            [1.0, 1.0, self._gas_storage.capacity, np.inf]
            + [np.inf] * (self._max_steps * 2),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    @property
    def accumulated_pnl(self):
        return self._accumulated_pnl

    def validate_forward_curve(self, forward_curve):
        if not isinstance(forward_curve, np.ndarray):
            raise ValueError("Forward curve must be a numpy array.")
        if forward_curve.shape != (self._max_steps, self._max_steps):
            error_message = (
                "Forward curve must have shape "
                f"({self._max_steps}, {self._max_steps}), "
                f"got {forward_curve.shape}."
            )
            raise ValueError(error_message)

        if not np.all(np.isfinite(forward_curve)):
            raise ValueError("Forward curve contains non-finite values.")

    def reset(self, *, seed=None, options=None):
        # the forward curve is passed during reset
        super().reset(seed=seed)

        # NOTE: consider input validation
        self._forward_curve = self._price_buffer.get_path()
        self.validate_forward_curve(self._forward_curve)

        self._current_step = 0

        self._gas_storage = GasStorage(
            self._gas_storage._capacity,
            self._gas_storage._injection_withdrawal_curve,
            self._gas_storage._storage_period,
        )

        self._futures_positions = np.zeros(
            (self._max_steps,), dtype=np.float32
        )
        self._accumulated_pnl = 0.0

        observation = self._concat_observation()
        return observation, {}

    def _evaluate_action_costs(self, schedule):
        costs = 0.0
        if schedule[self._current_step] > 0:
            costs += self._injection_costs * schedule[self._current_step]
        elif schedule[self._current_step] < 0:
            costs -= self._withdrawal_costs * schedule[self._current_step]
        return costs

    @property
    def observation_shape(self):
        return self.observation_space.shape

    def _encode_seasonality(self) -> np.ndarray:
        """Encode the current step as sin/cos for seasonality."""
        seasonality = 2 * np.pi * self._current_step / self._max_steps
        return np.array(
            [np.sin(seasonality), np.cos(seasonality)], dtype=np.float32
        )

    def _concat_observation(self, terminated=False):
        seasonality = self._encode_seasonality()
        inventory = np.array(
            [self._gas_storage._current_inventory], dtype=np.float32
        )
        if self._forward_curve is not None:
            if not terminated:
                forward_curve = self._forward_curve[self._current_step]
            else:
                forward_curve = np.zeros((self._max_steps,), dtype=np.float32)

        else:
            raise ValueError(
                "Forward curve is not set. Call reset() before step()."
            )

        observation = np.concatenate(
            [
                seasonality,
                inventory.flatten(),
                np.array([self._accumulated_pnl], dtype=np.float32),
                forward_curve,
                self._futures_positions.flatten(),
            ],
            dtype=np.float32,
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
            -current_action
            * self._forward_curve[self._current_step, self._current_step]
        )

    def _calculate_futures_pnl_mtm(self):
        # handle first step here for cleaner api
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
        return self._futures_positions

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

    def validate_action(self, action):
        if not isinstance(action, np.ndarray):
            raise ValueError("Action must be a numpy array.")
        if action.shape != (self._max_steps,):
            error_message = (
                "Action must have shape "
                f"({self._max_steps},), got {action.shape}."
            )
            raise ValueError(error_message)

        # test action not infinite
        if not np.all(np.isfinite(action)):
            raise ValueError("Action contains non-finite values.")
        # test action in range and not NaN
        if not np.all((action >= -1) & (action <= 1)):
            raise ValueError("Action must be in range [-1, 1].")
        if np.any(np.isnan(action)):
            raise ValueError("Action contains NaN values.")

    def step(self, action):
        self.validate_action(action)
        # settle money
        schedule = self._decode_schedule(action)

        phys_cf, costs, futures_pnl = self._settle_current_flows(schedule)
        reward = np.array([phys_cf - costs + futures_pnl], dtype=np.float32)
        self._accumulated_pnl += reward[0]

        new_schedule = self._rebalance_futures_positions(schedule)

        if schedule[self._current_step] > 0:
            self._gas_storage.inject(schedule[self._current_step])
        else:
            self._gas_storage.withdraw(-schedule[self._current_step])

        self._current_step += 1

        terminated = self._current_step >= self._max_steps

        if terminated:
            observation = self._concat_observation(terminated=True)
        else:
            observation = self._concat_observation()

        return (
            observation,
            reward[0],
            terminated,
            False,
            {
                "phys_cf": phys_cf,
                "costs": costs,
                "futures_pnl": futures_pnl,
                "schedule": new_schedule,
            },
        )
