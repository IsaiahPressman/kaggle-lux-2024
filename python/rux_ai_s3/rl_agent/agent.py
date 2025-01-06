import json
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import torch
from pydantic import BaseModel, Field

from rux_ai_s3.feature_engineering_env import FeatureEngineeringEnv
from rux_ai_s3.models.actor_critic import (
    ActorCritic,
    FactorizedActorCritic,
)
from rux_ai_s3.models.actor_heads import ActionConfig
from rux_ai_s3.models.build import build_actor_critic
from rux_ai_s3.models.types import TorchActionInfo, TorchObs
from rux_ai_s3.rl_training.constants import TRAIN_CONFIG_FILE_NAME
from rux_ai_s3.rl_training.train_config import TrainConfig
from rux_ai_s3.types import ActionArray
from rux_ai_s3.utils import load_from_yaml, to_json

AGENT_CONFIG_FILE = Path(__file__).parent / "agent_config.yaml"
TRAIN_CONFIG_FILE = Path(__file__).parent / TRAIN_CONFIG_FILE_NAME
ModelTypes = ActorCritic | FactorizedActorCritic


class AgentConfig(BaseModel):
    main_action_temperature: Annotated[float, Field(ge=0.0, le=1.0)]
    sap_action_temperature: Annotated[float, Field(ge=0.0, le=1.0)]


class Agent:
    def __init__(
        self,
        player: str,
        env_cfg: dict[str, Any],
    ) -> None:
        self.agent_config = load_from_yaml(AgentConfig, AGENT_CONFIG_FILE)
        self.train_config = load_from_yaml(TrainConfig, TRAIN_CONFIG_FILE)
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
        self.model = self.build_model()

    @property
    def action_config(self) -> ActionConfig:
        return ActionConfig(
            main_action_temperature=self.agent_config.main_action_temperature,
            sap_action_temperature=self.agent_config.sap_action_temperature,
        )

    def act(
        self, _step: int, obs: dict[str, Any], _remaining_overage_time: int
    ) -> ActionArray:
        raw_obs = json.dumps(to_json(obs))
        is_new_match = obs["match_steps"] == 0
        self.fe_env.step(raw_obs, self.last_actions, is_new_match=is_new_match)
        self.last_actions = self.get_new_actions()
        # TODO: Log memory statuses and estimated value
        return self.last_actions

    def get_new_actions(self) -> ActionArray:
        obs = TorchObs.from_numpy(
            self.fe_env.get_frame_stacked_obs(), device=self.device
        )
        action_info = self.fe_env.last_out.action_info
        torch_action_info = TorchActionInfo.from_numpy(action_info, device=self.device)
        model_out = self.model(
            obs=obs,
            action_info=torch_action_info,
            action_config=self.action_config,
        )
        return model_out.to_env_actions(action_info.unit_indices).squeeze(axis=0)

    def build_model(self) -> ModelTypes:
        example_obs = self.fe_env.get_frame_stacked_obs()
        spatial_in_channels = example_obs.spatial_obs.shape[1]
        global_in_channels = example_obs.global_obs.shape[1]
        model: ModelTypes = build_actor_critic(
            spatial_in_channels=spatial_in_channels,
            global_in_channels=global_in_channels,
            reward_space=self.train_config.env_config.reward_space,
            config=self.train_config.rl_model_config,
        )

        state_dict = torch.load(
            self.get_model_checkpoint_path(),
            map_location=self.device,
            weights_only=True,
        )["model"]
        state_dict = {
            self.remove_compile_prefix(key): value for key, value in state_dict.items()
        }
        model.load_state_dict(state_dict)
        return model.to(self.device).eval()

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
        parent_dir = Path(__file__).parent
        try:
            (path,) = list(parent_dir.glob("*.pt"))
        except ValueError as e:
            raise FileNotFoundError(
                f"Couldn't find weights checkpoint file in {parent_dir}"
            ) from e

        return path

    @staticmethod
    def remove_compile_prefix(key: str) -> str:
        prefix = "_orig_mod."
        if key.startswith(prefix):
            return key[len(prefix) :]

        return key
