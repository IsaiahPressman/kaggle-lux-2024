from rux_2024._lowlevel import ParallelEnv as LowLevelEnv


class ParallelEnv:
    def __init__(self, n_envs: int) -> None:
        self.n_envs = n_envs
        self._env = LowLevelEnv(n_envs)
