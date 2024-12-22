import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rux_ai_s3.feature_engineering_env import FeatureEngineeringEnv
from rux_ai_s3.models.actor_critic import ActorCritic
from rux_ai_s3.rl_agent.agent_config import AgentConfig, AGENT_CONFIG_FILE
from rux_ai_s3.rl_agent.train_config import TRAIN_CONFIG_FILE, TrainConfig
from rux_ai_s3.types import ActionArray, FeatureEngineeringOut
from rux_ai_s3.utils import load_from_yaml, to_json


class Agent:
    def __init__(
        self,
        player: str,
        env_cfg: dict[str, Any],
    ) -> None:
        self.train_config = load_from_yaml(TrainConfig, TRAIN_CONFIG_FILE)
        self.agent_config = load_from_yaml(AgentConfig, AGENT_CONFIG_FILE)
        self.team_id = self.get_team_id(player)
        self.fe_env = FeatureEngineeringEnv(
            frame_stack_len=self.train_config.frame_stack_len,
            team_id=self.team_id,
            env_params=env_cfg,
        )
        self.last_actions: ActionArray = np.zeros(
            (env_cfg["max_units"], 3), dtype=np.int64
        )

        self.device = self.get_device()
        self.model = ActorCritic.from_config(
            # spatial_in_channels=get_spatial_feature_count(),
            # global_in_channels=get_global_feature_count(),
        )
        raise NotImplementedError

    @property
    def opp_team_id(self) -> int:
        return 1 - self.team_id

    def act(
        self, _step: int, obs: dict[str, Any], _remaining_overage_time: int = 60
    ) -> ActionArray:
        raw_obs = json.dumps(to_json(obs))
        env_out = FeatureEngineeringOut.from_raw(
            self.fe_env.step(raw_obs, self.last_actions)
        )

    @staticmethod
    def get_team_id(player: str) -> int:
        if player == "player_0":
            return 0

        if player == "player_1":
            return 1

        raise ValueError(f"Invalid player '{player}'")

    @staticmethod
    def get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda:0")

        return torch.device("cpu")

    @staticmethod
    def get_model_checkpoint_path() -> Path:
        (path,) = list(Path(__file__).parent.glob("*.pt"))
        return path
