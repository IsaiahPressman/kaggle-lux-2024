import functools

import jax
import numpy as np
import pytest
from luxai_s3.state import gen_map

from rux_2024._lowlevel import (
    ParallelEnv,
    get_global_feature_count,
    get_spatial_feature_count,
)
from rux_2024.parallel_env import ParallelEnv as ParallelEnvAPI
from rux_2024.parallel_env import ParallelEnvOut

_FLOAT_FLAG = -1_000


def test_step() -> None:
    n_envs = 8
    env = ParallelEnv(n_envs)
    actions = np.zeros((2, 4, 5), dtype=int)
    env_out = ParallelEnvOut.from_env_out(env.seq_step(actions))
    env_out.validate()

    obs, _, _, _ = env_out.array_out
    assert obs.shape[:2] == (n_envs, 2)
    assert obs.shape[-2:] == (24, 24)
    assert np.all(obs == 1)
    assert env_out.reward == [(1, 2)] * n_envs
    assert env_out.done == [True] * n_envs

    env = ParallelEnv(n_envs)
    for i in range(1, 20):
        (obs, *_) = ParallelEnvOut.from_env_out(env.seq_step(actions)).array_out
        assert np.all(obs == i)


@pytest.mark.slow
def test_soft_reset() -> None:
    n_envs = 8
    map_size = (24, 24)
    map_width, map_height = map_size
    env = ParallelEnv(n_envs)
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

    obs_arrays = ParallelEnvAPI.make_empty_obs_arrays(
        n_envs=n_envs,
        spatial_channels=get_spatial_feature_count(),
        global_channels=get_global_feature_count(),
        map_size=map_size,
    )
    spatial_obs, global_obs, action_mask, sap_mask = obs_arrays
    spatial_obs[:] = _FLOAT_FLAG
    global_obs[:] = _FLOAT_FLAG
    action_mask[:] = True
    sap_mask[:] = True

    new_map_dict = gen_map_vmapped(jax.random.split(jax.random.key(42), n_envs))
    env.soft_reset(
        obs_arrays=obs_arrays,
        tile_type=np.asarray(new_map_dict["map_features"].tile_type),
        energy_nodes=np.asarray(new_map_dict["energy_nodes"]),
        energy_node_fns=np.asarray(new_map_dict["energy_node_fns"]),
        energy_nodes_mask=np.asarray(new_map_dict["energy_nodes_mask"]),
        relic_nodes=np.asarray(new_map_dict["relic_nodes"]),
        relic_node_configs=np.asarray(new_map_dict["relic_node_configs"]),
        relic_nodes_mask=np.asarray(new_map_dict["relic_nodes_mask"]),
    )
    assert np.all(spatial_obs != _FLOAT_FLAG)
    assert np.all(global_obs != _FLOAT_FLAG)
    assert np.all(np.logical_not(action_mask))
    assert np.all(np.logical_not(sap_mask))

    # Now try only resetting some envs
    reset_env_ids = [0, 1, 5]
    not_reset_env_ids = [i for i in range(n_envs) if i not in reset_env_ids]
    env.terminate_envs(reset_env_ids)

    obs_arrays = ParallelEnvAPI.make_empty_obs_arrays(
        n_envs=n_envs,
        spatial_channels=get_spatial_feature_count(),
        global_channels=get_global_feature_count(),
        map_size=map_size,
    )
    spatial_obs, global_obs, action_mask, sap_mask = obs_arrays
    spatial_obs[:] = _FLOAT_FLAG
    global_obs[:] = _FLOAT_FLAG
    action_mask[:] = True
    sap_mask[:] = True

    new_map_dict = gen_map_vmapped(jax.random.split(jax.random.key(42), n_envs))
    env.soft_reset(
        obs_arrays=obs_arrays,
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
    assert np.all(spatial_obs[reset_env_ids] != _FLOAT_FLAG)
    assert np.all(spatial_obs[not_reset_env_ids] == _FLOAT_FLAG)
    assert np.all(global_obs[reset_env_ids] != _FLOAT_FLAG)
    assert np.all(global_obs[not_reset_env_ids] == _FLOAT_FLAG)
    assert np.all(np.logical_not(action_mask[reset_env_ids]))
    assert np.all(action_mask[not_reset_env_ids])
    assert np.all(np.logical_not(sap_mask[reset_env_ids]))
    assert np.all(action_mask[not_reset_env_ids])
