"""Contract tests for the futures-enabled gas storage environment.

``price_buffer.get_path()`` returns one (12, 12) array.  At step t the current
row's diagonal F[t, t] is the price for physical settlement in month t.  Only
positions held since an earlier row earn mark-to-market on a price change;
positions opened at the current quote cannot earn a past price movement.
Entries before the diagonal are expired-price placeholders.

This proposed contract treats storage capacity, physical flows, and futures
positions as MWh.  That is intentional even though the current ``GasStorage``
docstring describes capacity as cubic metres.  The tests specify the proposed
environment contract and do not change that production docstring.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from py_gas_storage_valuation.data.storage import (
    GasStorage,
    InjectionWithdrawalCurve,
)
from py_gas_storage_valuation.envs.gas_storage_env import GasStorageEnv
from py_gas_storage_valuation.prices.price_buffer import ForwardCurvePathBuffer

N_MONTHS = 12
PHYSICAL_CASH_FLOW = "physical_cash_flow"
FUTURES_PNL = "futures_pnl"
COSTS = "costs"
PHYSICAL_FLOW = "physical_flow"


class StubPriceBuffer:
    """Return deterministic forward-curve paths without copying them."""

    def __init__(self, paths: list[np.ndarray]):
        self.paths = paths
        self.position = 0

    def get_path(self) -> np.ndarray:
        path = self.paths[self.position]
        self.position += 1
        return path


def _path(*, curves: np.ndarray | None = None) -> np.ndarray:
    if curves is None:
        curves = np.zeros((N_MONTHS, N_MONTHS), dtype=np.float64)
        assert curves is not None
        for step in range(N_MONTHS):
            curves[step, step:] = 100.0 + np.arange(
                N_MONTHS - step, dtype=np.float64
            )
    assert curves is not None
    return curves


def _storage(capacity: float = 100.0, rate: float = 0.5) -> GasStorage:
    curve = InjectionWithdrawalCurve(
        bin_upper_thresholds=np.array([1.0]),
        injection_rates=np.array([rate]),
        withdrawal_rates=np.array([rate]),
    )
    return GasStorage(capacity, curve, "Y")


def _env(
    paths: list[np.ndarray],
    *,
    capacity: float = 100.0,
    rate: float = 0.5,
    injection_costs: float = 0.0,
    withdrawal_costs: float = 0.0,
) -> GasStorageEnv:
    return GasStorageEnv(
        gas_storage=_storage(capacity, rate),
        price_buffer=cast(ForwardCurvePathBuffer, StubPriceBuffer(paths)),
        max_steps=N_MONTHS,
        injection_costs=injection_costs,
        withdrawal_costs=withdrawal_costs,
    )


def _action(*multipliers: float) -> np.ndarray:
    action = np.zeros(N_MONTHS, dtype=np.float32)
    action[: len(multipliers)] = multipliers
    return action


def _custom_two_month_path(*, second_quote: float) -> np.ndarray:
    curves = _path()
    curves[0, 0] = 40.0
    curves[0, 1] = 60.0
    curves[1, 0] = 0.0  # expired-price placeholder, not a tradable quote
    curves[1, 1] = second_quote
    return curves


def test_step_reward_is_scalar_and_breakdown_sums_to_reward():
    curves = _path()
    curves[0, 0] = 10.0
    env = _env([curves])
    env.reset()

    _, reward, terminated, truncated, info = env.step(_action(0.5, -0.5))

    assert np.isscalar(reward)
    assert not terminated
    assert not truncated
    assert info[PHYSICAL_CASH_FLOW] == pytest.approx(-25.0 * 10.0)
    assert info[FUTURES_PNL] == pytest.approx(0.0)
    assert info[COSTS] == pytest.approx(0.0)
    assert reward == pytest.approx(
        info[PHYSICAL_CASH_FLOW] + info[FUTURES_PNL] - info[COSTS]
    )


def test_held_contract_mtm_uses_current_diagonal_settlement_quote():
    curves = _custom_two_month_path(second_quote=55.0)
    env = _env([curves])
    env.reset()
    fixed_schedule = _action(1.0, -1.0)

    _, first_reward, _, _, first_info = env.step(fixed_schedule)
    _, second_reward, _, _, second_info = env.step(fixed_schedule)

    # Month 0 is already priced at 40.  The month-1 short hedge entered
    # at 60 earns (55 - 60) * -50 at the next decision/settlement.
    # The expired F[1, 0] placeholder must not generate a price movement.
    assert first_info[PHYSICAL_CASH_FLOW] == pytest.approx(-50.0 * 40.0)
    assert first_info[FUTURES_PNL] == pytest.approx(0.0)
    assert first_reward == pytest.approx(-2000.0)
    assert second_info[PHYSICAL_CASH_FLOW] == pytest.approx(50.0 * 55.0)
    assert second_info[FUTURES_PNL] == pytest.approx(-50.0 * (55.0 - 60.0))
    assert second_reward == pytest.approx(3000.0)


def test_rebalancing_continuing_position_has_no_phantom_pnl():
    curves = _path()
    curves[:, :] = 0.0
    curves[0, 2] = 100.0
    curves[1, 2] = 110.0
    curves[0, 3] = curves[1, 3] = 100.0
    env = _env([curves])
    env.reset()

    _, first_reward, _, _, _ = env.step(_action(0.0, 0.0, 1.0, -1.0))
    _, second_reward, _, _, second_info = env.step(_action())
    _, third_reward, _, _, third_info = env.step(_action())

    # The original plan injects in month 2 and withdraws in month 3, so it
    # is feasible. Closing both futures at the next quote must not generate
    # an additional rebalancing PnL.
    assert first_reward == pytest.approx(0.0)
    assert second_info[FUTURES_PNL] == pytest.approx(50.0 * (110.0 - 100.0))
    assert second_reward == pytest.approx(500.0)
    assert third_info[FUTURES_PNL] == pytest.approx(0.0)
    assert third_reward == pytest.approx(0.0)


@pytest.mark.parametrize("second_quote", [55.0, 95.0])
def test_fully_hedged_two_month_schedule_locks_twenty_per_unit(
    second_quote: float,
):
    path = _custom_two_month_path(second_quote=second_quote)
    env = _env([path])
    env.reset()

    rewards = []
    for step in range(N_MONTHS):
        action = _action(1.0, -1.0) if step < 2 else _action()
        _, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        if step < N_MONTHS - 1:
            assert not terminated
            assert not truncated

    assert sum(rewards) == pytest.approx(50.0 * (60.0 - 40.0))


def test_zero_action_has_zero_pnl_and_inventory_for_twelve_steps():
    env = _env([_path()])
    env.reset()

    for _ in range(N_MONTHS):
        observation, reward, _, _, info = env.step(_action())
        assert reward == pytest.approx(0.0)
        assert info[PHYSICAL_CASH_FLOW] == pytest.approx(0.0)
        assert info[FUTURES_PNL] == pytest.approx(0.0)
        assert observation[2] == pytest.approx(0.0)


def test_capacity_and_costs_are_applied_in_mwh():
    curves = _path()
    for step in range(N_MONTHS):
        curves[step, step:] = 10.0
    env = _env(
        [curves],
        capacity=100.0,
        rate=0.5,
        injection_costs=2.0,
        withdrawal_costs=3.0,
    )
    env.reset()

    after_injection, injection_reward, _, _, injection_info = env.step(
        _action(1.0, -1.0)
    )
    after_withdrawal, withdrawal_reward, _, _, withdrawal_info = env.step(
        _action(1.0, -1.0)
    )

    assert injection_info[PHYSICAL_FLOW] == pytest.approx(50.0)
    assert injection_info[COSTS] == pytest.approx(50.0 * 2.0)
    assert injection_reward == pytest.approx(-600.0)
    assert after_injection[2] == pytest.approx(50.0)
    assert withdrawal_info[PHYSICAL_FLOW] == pytest.approx(-50.0)
    assert withdrawal_info[COSTS] == pytest.approx(50.0 * 3.0)
    assert withdrawal_reward == pytest.approx(350.0)
    assert after_withdrawal[2] == pytest.approx(0.0)


def test_inventory_dependent_rates_apply_to_each_month_of_the_plan():
    curve = InjectionWithdrawalCurve(
        bin_upper_thresholds=np.array([0.5, 1.0]),
        injection_rates=np.array([0.5, 0.25]),
        withdrawal_rates=np.array([0.25, 0.5]),
    )
    env = GasStorageEnv(
        gas_storage=GasStorage(100.0, curve, "Y"),
        price_buffer=cast(ForwardCurvePathBuffer, StubPriceBuffer([_path()])),
        max_steps=N_MONTHS,
    )
    env.reset()

    plan = _action(1.0, 1.0, -1.0, -1.0)
    for expected_flow, expected_inventory in zip(
        (50.0, 25.0, -50.0, -25.0),
        (50.0, 75.0, 25.0, 0.0),
        strict=True,
    ):
        observation, _, _, _, info = env.step(plan)
        assert info[PHYSICAL_FLOW] == pytest.approx(expected_flow)
        assert observation[2] == pytest.approx(expected_inventory)


@pytest.mark.parametrize("multiplier", [1.0, -1.0])
def test_infeasible_rate_schedule_is_projected_to_feasible_inventory(
    multiplier: float,
):
    env = _env([_path()])
    env.reset()

    observation = None
    for _ in range(N_MONTHS):
        observation, _, _, _, _ = env.step(
            np.full(N_MONTHS, multiplier, dtype=np.float32)
        )
        assert 0.0 <= observation[2] <= 100.0

    assert observation is not None
    # Only the invariant is specified here; the repair/projection algorithm is
    # intentionally undecided and its exact schedule is not asserted.
    assert observation[2] == pytest.approx(0.0)


def test_reset_clears_futures_and_isolates_the_next_path():
    first_curves = _path()
    first_curves[0, 0] = 10.0
    first_curves[0, 1] = 20.0
    second_curves = _path()
    second_curves[0, 0] = 99.0
    second_curves[0, 1] = 10.0
    second_curves[1, 1] = 50.0
    buffer = StubPriceBuffer([first_curves, second_curves])
    env = GasStorageEnv(
        gas_storage=_storage(),
        price_buffer=cast(ForwardCurvePathBuffer, buffer),
        max_steps=N_MONTHS,
    )
    env.reset()
    env.step(_action(1.0, -1.0))

    observation, _ = env.reset()
    assert observation[3] == pytest.approx(99.0)
    _, reward, _, _, info = env.step(_action())
    assert reward == pytest.approx(0.0)
    assert info[FUTURES_PNL] == pytest.approx(0.0)


def test_twelve_steps_naturally_terminate_with_valid_terminal_observation():
    env = _env([_path()])
    observation, _ = env.reset()

    assert env.observation_space.contains(np.asarray(observation))
    for step in range(N_MONTHS):
        observation, _, terminated, truncated, _ = env.step(_action())
        assert env.observation_space.contains(np.asarray(observation))
        assert truncated is False
        if step < N_MONTHS - 1:
            assert terminated is False
        else:
            assert terminated is True


def test_observation_exposes_only_the_current_curve_row():
    first_curves = _path()
    second_curves = _path()
    first_curves[1, 1:] = 111.0
    second_curves[1, 1:] = 999.0
    env_a = _env([first_curves])
    env_b = _env([second_curves])

    observation_a, _ = env_a.reset()
    observation_b, _ = env_b.reset()
    np.testing.assert_array_equal(observation_a, observation_b)
    np.testing.assert_allclose(observation_a[3:15], first_curves[0])

    next_a, *_ = env_a.step(_action())
    next_b, *_ = env_b.step(_action())
    assert not np.array_equal(next_a, next_b)
    np.testing.assert_allclose(next_a[3:15], first_curves[1])


def test_observation_exposes_open_positions_and_cumulative_profit():
    env = _env([_custom_two_month_path(second_quote=55.0)])
    observation, _ = env.reset()
    # Time (2), inventory (1), quotes (12), positions (12), cumulative PnL (1).
    assert observation.shape == (28,)
    np.testing.assert_array_equal(observation[15:27], np.zeros(N_MONTHS))
    assert observation[27] == pytest.approx(0.0)

    observation, reward, _, _, _ = env.step(_action(1.0, -1.0))
    assert observation[15] == pytest.approx(0.0)  # expired contract cleared
    assert observation[16] == pytest.approx(-50.0)  # future withdrawal hedge
    assert observation[27] == pytest.approx(reward)


def test_expired_action_slot_does_not_change_a_later_decision():
    path = _custom_two_month_path(second_quote=55.0)
    env_a, env_b = _env([path]), _env([path])
    env_a.reset()
    env_b.reset()
    env_a.step(_action(1.0, -1.0))
    env_b.step(_action(1.0, -1.0))

    observation_a, reward_a, _, _, info_a = env_a.step(_action(1.0, -1.0))
    observation_b, reward_b, _, _, info_b = env_b.step(_action(-1.0, -1.0))
    np.testing.assert_array_equal(observation_a, observation_b)
    assert reward_a == pytest.approx(reward_b)
    assert info_a[PHYSICAL_FLOW] == pytest.approx(info_b[PHYSICAL_FLOW])


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_action_fails_clearly(bad_value: float):
    env = _env([_path()])
    env.reset()

    with pytest.raises(ValueError, match="finite|nan|inf"):
        env.step(_action(bad_value))


def test_wrong_action_shape_fails_clearly():
    env = _env([_path()])
    env.reset()

    with pytest.raises(ValueError, match="shape|length|12"):
        env.step(np.zeros(N_MONTHS - 1, dtype=np.float32))


@pytest.mark.parametrize("multiplier", [-1.1, 1.1])
def test_action_outside_declared_bounds_fails_clearly(multiplier: float):
    env = _env([_path()])
    env.reset()

    with pytest.raises(ValueError, match="action|bound|range"):
        env.step(_action(multiplier))


@pytest.mark.parametrize(
    "curves", [np.zeros((11, 12)), np.full((12, 12), np.nan)]
)
def test_malformed_price_path_fails_on_reset(curves: np.ndarray):
    env = _env([curves])

    with pytest.raises(ValueError, match="shape|finite|curve"):
        env.reset()


def test_nonfinite_current_settlement_quote_fails_clearly():
    curves = _path()
    curves[0, 0] = np.nan
    env = _env([curves])

    with pytest.raises(ValueError, match="finite"):
        env.reset()


def test_negative_prices_are_processed_arithmetically():
    curves = _path()
    curves[:, :] = 0.0
    curves[0, 0] = -10.0
    env = _env([curves])
    env.reset()

    _, reward, _, _, info = env.step(_action(1.0, -1.0))

    assert np.isfinite(reward)
    assert reward == pytest.approx(50.0 * 10.0)
    assert info[PHYSICAL_CASH_FLOW] == pytest.approx(50.0 * 10.0)


def test_price_history_path_is_not_mutated_by_environment():
    curves = _path()
    original_curves = curves.copy()
    env = _env([curves])
    env.reset()
    env.step(_action(1.0, -1.0))

    np.testing.assert_array_equal(curves, original_curves)
