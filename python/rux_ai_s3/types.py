from collections.abc import Sequence
from typing import NamedTuple, Union

import numpy as np
import numpy.typing as npt

ObsArrays = tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]
ActionInfoArrays = tuple[
    npt.NDArray[np.bool_],
    npt.NDArray[np.bool_],
    npt.NDArray[np.int64],
    npt.NDArray[np.float32],
    npt.NDArray[np.bool_],
]
StatsArrays = tuple[
    dict[str, float],
    dict[str, npt.NDArray[np.float32]],
]
EnvFullOut = tuple[
    ObsArrays,
    ActionInfoArrays,
    npt.NDArray[np.float32],
    npt.NDArray[np.bool_],
    StatsArrays | None,
]


class Obs(NamedTuple):
    spatial_obs: npt.NDArray[np.float32]
    global_obs: npt.NDArray[np.float32]

    def validate(self, n_envs: int, n_players: int) -> None:
        assert self.spatial_obs.ndim == 5
        assert self.spatial_obs.dtype == np.float32
        assert self.global_obs.ndim == 3
        assert self.global_obs.dtype == np.float32
        for array in self:
            assert array.shape[:2] == (n_envs, n_players)

    @classmethod
    def from_raw(cls, raw: ObsArrays) -> "Obs":
        return Obs(*raw)

    @classmethod
    def concatenate_frame_history(cls, frames: Sequence["Obs"]) -> "Obs":
        raw_concatenated = [
            np.concatenate(obs_frames, axis=3) for obs_frames in zip(*frames)
        ]
        return Obs(*raw_concatenated)


class ActionInfo(NamedTuple):
    main_mask: npt.NDArray[np.bool_]
    sap_mask: npt.NDArray[np.bool_]
    unit_indices: npt.NDArray[np.int64]
    unit_energies: npt.NDArray[np.float32]
    units_mask: npt.NDArray[np.bool_]

    def validate(self, n_envs: int, n_players: int) -> None:
        assert self.main_mask.ndim == 4
        assert self.main_mask.dtype == np.bool_
        assert self.sap_mask.ndim == 5
        assert self.sap_mask.dtype == np.bool_
        assert self.unit_indices.ndim == 4
        assert self.unit_indices.dtype == np.int64
        assert self.unit_energies.ndim == 3
        assert self.unit_energies.dtype == np.float32
        assert self.units_mask.ndim == 3
        assert self.units_mask.dtype == np.bool_
        for array in self:
            assert array.shape[:2] == (n_envs, n_players)

    @classmethod
    def from_raw(cls, raw: ActionInfoArrays) -> "ActionInfo":
        return ActionInfo(*raw)


class Stats(NamedTuple):
    scalar_stats: dict[str, float]
    array_stats: dict[str, npt.NDArray[np.float32]]

    def validate(self) -> None:
        for scalar in self.scalar_stats.values():
            assert isinstance(scalar, float)

        for array in self.array_stats.values():
            assert array.ndim == 1
            assert array.dtype == np.float32

    @classmethod
    def from_raw(cls, raw: StatsArrays | None) -> Union["Stats", None]:
        if raw is None:
            return None

        return Stats(*raw)


class ParallelEnvOut(NamedTuple):
    obs: Obs
    action_info: ActionInfo
    reward: npt.NDArray[np.float32]
    done: npt.NDArray[np.bool_]
    stats: Stats | None

    def validate(self) -> None:
        n_envs, n_players = self.reward.shape
        assert n_players == 2
        self.obs.validate(n_envs, n_players)
        self.action_info.validate(n_envs, n_players)
        if self.stats:
            self.stats.validate()

        assert self.reward.shape == (n_envs, n_players)
        assert self.reward.dtype == np.float32
        assert self.done.shape == (n_envs,)
        assert self.done.dtype == np.bool_

    @classmethod
    def from_raw(cls, raw: EnvFullOut) -> "ParallelEnvOut":
        (raw_obs_arrays, raw_action_info_arrays, reward, done, stats) = raw
        return ParallelEnvOut(
            Obs.from_raw(raw_obs_arrays),
            ActionInfo.from_raw(raw_action_info_arrays),
            reward,
            done,
            Stats.from_raw(stats),
        )

    @classmethod
    def from_raw_validated(cls, raw: EnvFullOut) -> "ParallelEnvOut":
        out = cls.from_raw(raw)
        out.validate()
        return out
