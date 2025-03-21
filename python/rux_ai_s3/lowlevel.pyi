from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

from rux_ai_s3.types import ActionArray, FeatureEngineeringFullOut, ParallelEnvFullOut

def hello_world() -> str: ...
def hello_numpy_world() -> npt.NDArray[np.float32]: ...
def assert_release_build() -> None: ...
def get_temporal_spatial_feature_count() -> int: ...
def get_nontemporal_spatial_feature_count() -> int: ...
def get_temporal_global_feature_count() -> int: ...
def get_nontemporal_global_feature_count() -> int: ...

class RewardSpace(Enum):
    FINAL_WINNER = ...
    MATCH_WINNER = ...
    POINTS_SCORED = ...

    @staticmethod
    def list() -> list[RewardSpace]: ...
    @staticmethod
    def from_str(s: str) -> RewardSpace: ...

class SapMasking(Enum):
    POINT_TILES = ...
    OPP_UNIT_FRONTIER = ...

    @staticmethod
    def list() -> list[SapMasking]: ...
    @staticmethod
    def from_str(s: str) -> SapMasking: ...

class ParallelEnv:
    def __init__(
        self, n_envs: int, sap_masking: SapMasking, reward_space: RewardSpace
    ) -> None: ...
    def terminate_envs(self, env_ids: list[int]) -> None: ...
    def get_new_match_envs(self) -> list[int]: ...
    def get_new_game_envs(self) -> list[int]: ...
    def get_empty_outputs(self) -> ParallelEnvFullOut: ...
    def soft_reset(
        self,
        output_arrays: ParallelEnvFullOut,
        tile_type: npt.NDArray[np.int32],
        energy_nodes: npt.NDArray[np.int16],
        energy_node_fns: npt.NDArray[np.float32],
        energy_nodes_mask: npt.NDArray[np.bool_],
        relic_nodes: npt.NDArray[np.int16],
        relic_node_configs: npt.NDArray[np.bool_],
        relic_nodes_mask: npt.NDArray[np.bool_],
        relic_spawn_schedule: npt.NDArray[np.int32],
    ) -> None: ...
    def seq_step(self, actions: ActionArray) -> ParallelEnvFullOut: ...
    def par_step(self, actions: ActionArray) -> ParallelEnvFullOut: ...

class FeatureEngineeringEnv:
    def __init__(
        self,
        team_id: int,
        sap_masking: SapMasking,
        env_params: dict[str, Any],
    ) -> None: ...
    def get_empty_outputs(self) -> FeatureEngineeringFullOut: ...
    def step(
        self, lux_obs: str, last_actions: ActionArray
    ) -> FeatureEngineeringFullOut: ...
