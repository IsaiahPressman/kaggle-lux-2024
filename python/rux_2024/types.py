from enum import Enum
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

ObsArrays = tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.bool_],
    npt.NDArray[np.bool_],
    npt.NDArray[np.int64],
    npt.NDArray[np.float32],
    npt.NDArray[np.bool_],
]
EnvFullOut = tuple[
    ObsArrays,
    npt.NDArray[np.float32],
    npt.NDArray[np.bool_],
]


class Obs(NamedTuple):
    spatial_obs: npt.NDArray[np.float32]
    global_obs: npt.NDArray[np.float32]
    action_mask: npt.NDArray[np.bool_]
    sap_mask: npt.NDArray[np.bool_]
    unit_indices: npt.NDArray[np.int64]
    unit_energies: npt.NDArray[np.float32]
    units_mask: npt.NDArray[np.bool_]

    def validate(self) -> None:
        assert self.spatial_obs.ndim == 5
        assert self.spatial_obs.dtype == np.float32
        assert self.global_obs.ndim == 3
        assert self.global_obs.dtype == np.float32
        assert self.action_mask.ndim == 4
        assert self.action_mask.dtype == np.bool_
        assert self.sap_mask.ndim == 5
        assert self.sap_mask.dtype == np.bool_
        assert self.unit_indices.ndim == 4
        assert self.unit_indices.dtype == np.int64
        assert self.unit_energies.ndim == 3
        assert self.unit_energies.dtype == np.float32
        assert self.units_mask.ndim == 3
        assert self.units_mask.dtype == np.bool_

    @classmethod
    def from_raw(cls, raw: ObsArrays) -> "Obs":
        return Obs(*raw)


class ParallelEnvOut(NamedTuple):
    obs: Obs
    reward: npt.NDArray[np.float32]
    done: npt.NDArray[np.bool_]

    def validate(self) -> None:
        self.obs.validate()
        assert self.reward.ndim == 2
        assert self.reward.dtype == np.float32
        assert self.done.ndim == 1
        assert self.done.dtype == np.bool_

    @classmethod
    def from_raw(cls, raw: EnvFullOut) -> "ParallelEnvOut":
        (raw_obs_arrays, reward, done) = raw
        return ParallelEnvOut(Obs.from_raw(raw_obs_arrays), reward, done)

    @classmethod
    def from_raw_validated(cls, raw: EnvFullOut) -> "ParallelEnvOut":
        out = cls.from_raw(raw)
        out.validate()
        return out


class Action(Enum):
    NO_OP = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4
    SAP = 5
