from typing import NamedTuple

import numpy as np
import numpy.typing as npt

EnvFullOut = tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.bool_],
    npt.NDArray[np.bool_],
    npt.NDArray[np.float32],
    npt.NDArray[np.bool_],
]


class ParallelEnvOut(NamedTuple):
    spatial_obs: npt.NDArray[np.float32]
    global_obs: npt.NDArray[np.float32]
    action_mask: npt.NDArray[np.bool_]
    sap_mask: npt.NDArray[np.bool_]
    reward: npt.NDArray[np.float32]
    done: npt.NDArray[np.bool_]

    def _validate(self) -> None:
        assert self.spatial_obs.ndim == 5
        assert self.spatial_obs.dtype == np.float32
        assert self.global_obs.ndim == 3
        assert self.global_obs.dtype == np.float32
        assert self.action_mask.ndim == 4
        assert self.action_mask.dtype == np.bool_
        assert self.sap_mask.ndim == 5
        assert self.sap_mask.dtype == np.bool_
        assert self.reward.ndim == 2
        assert self.reward.dtype == np.float32
        assert self.done.ndim == 1
        assert self.done.dtype == np.bool_

    @classmethod
    def from_raw(cls, raw: EnvFullOut) -> "ParallelEnvOut":
        return cls(*raw)

    @classmethod
    def from_raw_validated(cls, raw: EnvFullOut) -> "ParallelEnvOut":
        out = cls.from_raw(raw)
        out._validate()
        return out
