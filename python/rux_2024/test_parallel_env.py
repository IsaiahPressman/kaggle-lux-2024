import numpy as np

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
    for i in range(1, 100):
        (obs, *_) = env.seq_step(actions)
        assert np.all(obs == i)
