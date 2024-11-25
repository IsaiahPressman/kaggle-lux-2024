import functools

import jax
import numpy as np
import pytest
from luxai_s3.state import gen_map

from rux_2024._lowlevel import (
    ParallelEnv,
)
from rux_2024.types import ParallelEnvOut

_N_ENVS = 8
_FLOAT_FLAG = -1_000


@pytest.mark.slow
def test_step() -> None:
    map_width, map_height = (24, 24)
    env = ParallelEnv(_N_ENVS)
    # Reset env
    gen_map_vmapped = jax.vmap(
        functools.partial(
            gen_map,
            params=None,
            map_type=1,
            map_height=map_height,
            map_width=map_width,
            max_energy_nodes=6,
            max_relic_nodes=6,
            relic_config_size=5,
        )
    )
    new_map_dict = gen_map_vmapped(jax.random.split(jax.random.key(42), _N_ENVS))
    env_out = ParallelEnvOut.from_raw_validated(env.get_empty_outputs())
    env.soft_reset(
        obs_arrays=env_out,
        tile_type=np.asarray(new_map_dict["map_features"].tile_type),
        energy_nodes=np.asarray(new_map_dict["energy_nodes"]),
        energy_node_fns=np.asarray(new_map_dict["energy_node_fns"]),
        energy_nodes_mask=np.asarray(new_map_dict["energy_nodes_mask"]),
        relic_nodes=np.asarray(new_map_dict["relic_nodes"]),
        relic_node_configs=np.asarray(new_map_dict["relic_node_configs"]),
        relic_nodes_mask=np.asarray(new_map_dict["relic_nodes_mask"]),
    )

    actions = np.zeros((_N_ENVS, 2, 16, 3), dtype=int)
    for _ in range(303):
        assert not np.any(env_out.done)
        env_out = ParallelEnvOut.from_raw_validated(env.seq_step(actions))

    assert np.all(env_out.done)
    expected_reward = np.zeros((_N_ENVS, 2), dtype=float)
    expected_reward[:] = [1, -1]
    assert np.all(env_out.reward == expected_reward)


@pytest.mark.slow
def test_soft_reset() -> None:
    map_width, map_height = (24, 24)
    env = ParallelEnv(_N_ENVS)
    gen_map_vmapped = jax.vmap(
        functools.partial(
            gen_map,
            params=None,
            map_type=1,
            map_height=map_height,
            map_width=map_width,
            max_energy_nodes=6,
            max_relic_nodes=6,
            relic_config_size=5,
        )
    )

    env_out = ParallelEnvOut.from_raw_validated(env.get_empty_outputs())
    env_out.spatial_obs[:] = _FLOAT_FLAG
    env_out.global_obs[:] = _FLOAT_FLAG
    env_out.action_mask[:] = True
    env_out.sap_mask[:] = True
    env_out.reward[:] = _FLOAT_FLAG
    env_out.done[:] = True

    new_map_dict = gen_map_vmapped(jax.random.split(jax.random.key(42), _N_ENVS))
    env.soft_reset(
        obs_arrays=env_out,
        tile_type=np.asarray(new_map_dict["map_features"].tile_type),
        energy_nodes=np.asarray(new_map_dict["energy_nodes"]),
        energy_node_fns=np.asarray(new_map_dict["energy_node_fns"]),
        energy_nodes_mask=np.asarray(new_map_dict["energy_nodes_mask"]),
        relic_nodes=np.asarray(new_map_dict["relic_nodes"]),
        relic_node_configs=np.asarray(new_map_dict["relic_node_configs"]),
        relic_nodes_mask=np.asarray(new_map_dict["relic_nodes_mask"]),
    )
    assert np.all(env_out.spatial_obs != _FLOAT_FLAG)
    assert np.all(env_out.global_obs != _FLOAT_FLAG)
    assert np.all(np.logical_not(env_out.action_mask))
    assert np.all(np.logical_not(env_out.sap_mask))
    # Reward and done are left as-is after a soft reset
    assert np.all(env_out.reward == _FLOAT_FLAG)
    assert np.all(env_out.done)

    # Now try only resetting some envs
    reset_env_ids = [0, 1, 5]
    not_reset_env_ids = [i for i in range(_N_ENVS) if i not in reset_env_ids]
    env.terminate_envs(reset_env_ids)

    env_out = ParallelEnvOut.from_raw_validated(env.get_empty_outputs())
    env_out.spatial_obs[:] = _FLOAT_FLAG
    env_out.global_obs[:] = _FLOAT_FLAG
    env_out.action_mask[:] = True
    env_out.sap_mask[:] = True
    env_out.reward[not_reset_env_ids] = _FLOAT_FLAG - 1
    env_out.reward[reset_env_ids] = _FLOAT_FLAG
    env_out.done[not_reset_env_ids] = False
    env_out.done[reset_env_ids] = True

    new_map_dict = gen_map_vmapped(jax.random.split(jax.random.key(42), _N_ENVS))
    env.soft_reset(
        obs_arrays=env_out,
        tile_type=np.asarray(new_map_dict["map_features"].tile_type)[
            : len(reset_env_ids)
        ],
        energy_nodes=np.asarray(new_map_dict["energy_nodes"])[: len(reset_env_ids)],
        energy_node_fns=np.asarray(new_map_dict["energy_node_fns"])[
            : len(reset_env_ids)
        ],
        energy_nodes_mask=np.asarray(new_map_dict["energy_nodes_mask"])[
            : len(reset_env_ids)
        ],
        relic_nodes=np.asarray(new_map_dict["relic_nodes"])[: len(reset_env_ids)],
        relic_node_configs=np.asarray(new_map_dict["relic_node_configs"])[
            : len(reset_env_ids)
        ],
        relic_nodes_mask=np.asarray(new_map_dict["relic_nodes_mask"])[
            : len(reset_env_ids)
        ],
    )
    assert np.all(env_out.spatial_obs[reset_env_ids] != _FLOAT_FLAG)
    assert np.all(env_out.spatial_obs[not_reset_env_ids] == _FLOAT_FLAG)
    assert np.all(env_out.global_obs[reset_env_ids] != _FLOAT_FLAG)
    assert np.all(env_out.global_obs[not_reset_env_ids] == _FLOAT_FLAG)
    assert np.all(np.logical_not(env_out.action_mask[reset_env_ids]))
    assert np.all(env_out.action_mask[not_reset_env_ids])
    assert np.all(np.logical_not(env_out.sap_mask[reset_env_ids]))
    assert np.all(env_out.action_mask[not_reset_env_ids])
    # Reward and done are left as-is after a soft reset
    assert np.all(env_out.reward[reset_env_ids] == _FLOAT_FLAG)
    assert np.all(env_out.reward[not_reset_env_ids] == _FLOAT_FLAG - 1)
    assert np.all(env_out.done[reset_env_ids])
    assert np.all(np.logical_not(env_out.done[not_reset_env_ids]))
