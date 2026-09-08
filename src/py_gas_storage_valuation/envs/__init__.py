from py_gas_storage_valuation.envs.gas_store_env import GasStorageEnv
from gymnasium.envs.registration import register


register(
    id="GasStorage-v0",
    entry_point="py_gas_storage_valuation.envs:GasStorageEnv",
    max_episode_steps=12,
)
