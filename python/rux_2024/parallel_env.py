from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from rux_2024._lowlevel import ParallelEnv as LowLevelEnv

EnvArrayOut = tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.bool_],
    npt.NDArray[np.bool_],
]
RewardT = list[tuple[int, int]]
DoneT = list[bool]


class ParallelEnvOut(NamedTuple):
    spatial_obs: npt.NDArray[np.float32]
    global_obs: npt.NDArray[np.float32]
    action_mask: npt.NDArray[np.bool_]
    sap_mask: npt.NDArray[np.bool_]
    reward: RewardT
    done: DoneT

    @property
    def array_out(self) -> EnvArrayOut:
        return self[:-2]

    def validate(self) -> None:
        assert self.spatial_obs.ndim == 5
        assert self.spatial_obs.dtype == np.float32
        assert self.global_obs.ndim == 3
        assert self.global_obs.dtype == np.float32
        assert self.action_mask.ndim == 4
        assert self.action_mask.dtype == np.bool_
        assert self.sap_mask.ndim == 5
        assert self.sap_mask.dtype == np.bool_

    @classmethod
    def from_env_out(
        cls, env_out: tuple[EnvArrayOut, RewardT, DoneT]
    ) -> "ParallelEnvOut":
        return cls(
            *env_out[0],
            *env_out[1:],
        )


class ParallelEnv:
    def __init__(self, n_envs: int) -> None:
        self.n_envs = n_envs
        self._env = LowLevelEnv(n_envs)

    @staticmethod
    def make_empty_obs_arrays(
        n_envs: int,
        spatial_channels: int,
        global_channels: int,
        map_size: tuple[int, int],
    ) -> EnvArrayOut:
        p = 2
        w, h = map_size
        max_units = 16
        n_actions = 6
        return (
            np.zeros((n_envs, p, spatial_channels, w, h), dtype=np.float32),
            np.zeros((n_envs, p, global_channels), dtype=np.float32),
            np.zeros((n_envs, p, max_units, n_actions), dtype=np.bool_),
            np.zeros((n_envs, p, max_units, w, h), dtype=np.bool_),
        )
