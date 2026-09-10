from gymnasium.envs.registration import register

from py_gas_storage_valuation.envs.gas_store_env import (  # noqa: F401
    GasStorageEnv,
)

register(
    id="GasStorage-v0",
    entry_point="py_gas_storage_valuation.envs:GasStorageEnv",
    max_episode_steps=12,
)
