import functools

import jax
import numpy as np
import pytest
from luxai_s3.state import gen_map

from rux_2024._lowlevel import ParallelEnv


def test_step() -> None:
    n_envs = 8
    env = ParallelEnv(n_envs)
    actions = np.zeros((2, 4, 5), dtype=int)
    (obs, reward, done) = env.seq_step(actions)
    print(obs, reward, done)
    assert obs.shape[:2] == (n_envs, 2)
    assert obs.shape[-2:] == (24, 24)
    assert np.all(obs == 1)
    assert reward == [(1, 2)] * n_envs
    assert done == [True] * n_envs

    env = ParallelEnv(n_envs)
    for i in range(1, 20):
        (obs, *_) = env.seq_step(actions)
        assert np.all(obs == i)


@pytest.mark.skip("TODO")
def test_soft_reset() -> None:
    n_envs = 4
    env = ParallelEnv(n_envs)
    keys = jax.random.split(jax.random.key(42), n_envs)
    state = jax.vmap(
        functools.partial(
            gen_map,
            params=None,
            map_type=1,
            map_height=24,
            map_width=24,
            max_energy_nodes=6,
            max_relic_nodes=6,
            relic_config_size=5,
        )
    )(
        keys,
    )

    env.soft_reset(
        np.asarray(state["map_features"].tile_type),
        np.asarray(state["energy_nodes"]),
        np.asarray(state["energy_node_fns"]),
        np.asarray(state["energy_nodes_mask"]),
        np.asarray(state["relic_nodes"]),
        np.asarray(state["relic_node_configs"]),
        np.asarray(state["relic_nodes_mask"]),
    )
