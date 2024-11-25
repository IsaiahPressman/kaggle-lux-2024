from enum import Enum

import numpy as np
import numpy.typing as npt

from rux_2024.types import EnvFullOut

def hello_world() -> str: ...
def hello_numpy_world() -> npt.NDArray[np.float32]: ...
def get_spatial_feature_count() -> int: ...
def get_global_feature_count() -> int: ...

class RewardSpace(Enum):
    FINAL_WINNER = ...
    MATCH_WINNER = ...
    POINTS_SCORED = ...

    @staticmethod
    def list() -> list[RewardSpace]: ...
    @staticmethod
    def from_str(s: str) -> RewardSpace: ...

class ParallelEnv:
    def __init__(self, n_envs: int, reward_space: RewardSpace) -> None: ...
    def terminate_envs(self, env_ids: list[int]) -> None: ...
    def get_empty_outputs(self) -> EnvFullOut: ...
    def soft_reset(
        self,
        obs_arrays: EnvFullOut,
        tile_type: npt.NDArray[np.int32],
        energy_nodes: npt.NDArray[np.int16],
        energy_node_fns: npt.NDArray[np.float32],
        energy_nodes_mask: npt.NDArray[np.bool_],
        relic_nodes: npt.NDArray[np.int16],
        relic_node_configs: npt.NDArray[np.bool_],
        relic_nodes_mask: npt.NDArray[np.bool_],
    ) -> None: ...
    def seq_step(self, actions: npt.NDArray[np.int_]) -> EnvFullOut: ...
    def par_step(self, actions: npt.NDArray[np.int_]) -> EnvFullOut: ...
