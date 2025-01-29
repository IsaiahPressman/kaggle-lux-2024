import json
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator
from typing_extensions import assert_never

from rux_ai_s3.constants import MAP_SIZE
from rux_ai_s3.feature_engineering_env import FeatureEngineeringEnv
from rux_ai_s3.lowlevel import RewardSpace
from rux_ai_s3.models.actor_critic import (
    ActorCritic,
    ActorCriticOut,
    FactorizedActorCritic,
    FactorizedActorCriticOut,
)
from rux_ai_s3.models.actor_critic.out import extract_env_actions
from rux_ai_s3.models.build import build_actor_critic
from rux_ai_s3.models.types import TorchActionInfo, TorchObs
from rux_ai_s3.models.utils import remove_compile_prefix
from rux_ai_s3.rl_agent.data_augmentation import DataAugmenter, Rot180
from rux_ai_s3.rl_training.constants import TRAIN_CONFIG_FILE_NAME
from rux_ai_s3.rl_training.train_config import TrainConfig
from rux_ai_s3.types import ActionArray
from rux_ai_s3.utils import load_from_yaml, to_json

AGENT_CONFIG_FILE = Path(__file__).parent / "agent_config.yaml"
TRAIN_CONFIG_FILE = Path(__file__).parent / TRAIN_CONFIG_FILE_NAME
ModelTypes = ActorCritic | FactorizedActorCritic
ModelOutTypes = ActorCriticOut | FactorizedActorCriticOut
DataAugmentation = Literal["rotate_180", "player_reflect", "drift_reflect"]


class AgentConfig(BaseModel):
    main_action_temperature: Annotated[float, Field(ge=0.0, le=1.0)]
    sap_action_temperature: Annotated[float, Field(ge=0.0, le=1.0)]
    data_augmentations: list[DataAugmentation]

    @field_validator("data_augmentations")
    @classmethod
    def _validate_data_augmentations(
        cls, val: list[DataAugmentation]
    ) -> list[DataAugmentation]:
        if len(val) != len(set(val)):
            raise ValueError("Got duplicate data augmentations")

        return val


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
        self.data_augmenters = self.build_data_augmenters(
            self.agent_config.data_augmentations
        )

    @property
    def model_forward_kwargs(self) -> dict[str, Any]:
        return dict(  # noqa: C408
            main_action_temperature=self.agent_config.main_action_temperature,
            sap_action_temperature=self.agent_config.sap_action_temperature,
            omit_value=self.train_config.env_config.reward_space
            == RewardSpace.FINAL_WINNER,
        )

    def build_model(self) -> ModelTypes:
        example_obs = self.fe_env.get_frame_stacked_obs()
        spatial_in_channels = example_obs.spatial_obs.shape[1]
        global_in_channels = example_obs.global_obs.shape[1]
        n_main_actions = self.fe_env.last_out.action_info.main_mask.shape[-1]
        model: ModelTypes = build_actor_critic(
            spatial_in_channels=spatial_in_channels,
            global_in_channels=global_in_channels,
            n_main_actions=n_main_actions,
            reward_space=self.train_config.env_config.reward_space,
            config=self.train_config.rl_model_config,
        )

        state_dict = torch.load(
            self.get_model_checkpoint_path(),
            map_location=self.device,
            weights_only=True,
        )["model"]
        state_dict = {
            remove_compile_prefix(key): value for key, value in state_dict.items()
        }
        model.load_state_dict(state_dict)
        return model.to(self.device).eval()

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
        augmented_obs, augmented_action_info = self.apply_data_augmentations(
            obs,
            TorchActionInfo.from_numpy(action_info, device=self.device),
        )
        augmented_model_out = self.model(
            obs=augmented_obs,
            action_info=augmented_action_info,
            **self.model_forward_kwargs,
        )
        expanded_main_log_probs, expanded_sap_log_probs = (
            self.invert_data_augmentations(augmented_model_out)
        )
        main_actions = self.model.log_probs_to_actions(
            expanded_main_log_probs.mean(dim=0, keepdim=True),
            self.agent_config.main_action_temperature,
        )
        main_actions = torch.where(
            augmented_action_info.units_mask,
            main_actions,
            torch.zeros_like(main_actions),
        )
        sap_actions = self.model.log_probs_to_actions(
            expanded_sap_log_probs.mean(dim=0, keepdim=True),
            self.agent_config.sap_action_temperature,
        )
        env_actions = extract_env_actions(
            main_actions=main_actions,
            sap_actions=sap_actions,
            unit_indices=action_info.unit_indices,
        )
        return env_actions.squeeze(axis=0)

    def apply_data_augmentations(
        self,
        obs: TorchObs,
        action_info: TorchActionInfo,
    ) -> tuple[TorchObs, TorchActionInfo]:
        augmented_obs = TorchObs(
            spatial_obs=self.repeat_and_augment_spatial(obs.spatial_obs),
            global_obs=self.repeat_for_augmentation(obs.global_obs),
        )
        augmented_action_info = TorchActionInfo(
            main_mask=self.repeat_and_augment_action(action_info.main_mask),
            sap_mask=self.repeat_and_augment_spatial(action_info.sap_mask),
            unit_indices=self.repeat_and_augment_coordinates(action_info.unit_indices),
            unit_energies=self.repeat_for_augmentation(action_info.unit_energies),
            units_mask=self.repeat_for_augmentation(action_info.units_mask),
        )
        return augmented_obs, augmented_action_info

    def repeat_and_augment_spatial(
        self,
        spatial: torch.Tensor,
    ) -> torch.Tensor:
        spatial = self.repeat_for_augmentation(spatial)
        for i, augmenter in enumerate(self.data_augmenters, start=1):
            spatial[i] = augmenter.transform_spatial(spatial[i])

        return spatial

    def repeat_and_augment_action(self, action_space: torch.Tensor) -> torch.Tensor:
        action_space = self.repeat_for_augmentation(action_space)
        for i, augmenter in enumerate(self.data_augmenters, start=1):
            action_space[i] = augmenter.transform_action_space(action_space[i])

        return action_space

    def repeat_and_augment_coordinates(self, coordinates: torch.Tensor) -> torch.Tensor:
        coordinates = self.repeat_for_augmentation(coordinates)
        for i, augmenter in enumerate(self.data_augmenters, start=1):
            coordinates[i] = augmenter.transform_coordinates(coordinates[i], MAP_SIZE)

        return coordinates

    def repeat_for_augmentation(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return t.repeat(len(self.data_augmenters) + 1, *(-1 for _ in range(t.ndim - 1)))

    def invert_data_augmentations(
        self, model_out: ModelOutTypes
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates main_log_probs and sap_log_probs, and removes actions
        themselves, since those must be resampled from the updated log probs.
        """
        if isinstance(model_out, ActorCriticOut):
            new_main_log_probs = model_out.main_log_probs.clone()
            new_sap_log_probs = model_out.sap_log_probs.clone()
            for i, augmenter in enumerate(self.data_augmenters, start=1):
                new_main_log_probs[i] = augmenter.inverse_transform_action_space(
                    model_out.main_log_probs[i]
                )
                new_sap_log_probs[i] = augmenter.inverse_transform_spatial(
                    model_out.sap_log_probs[i]
                )

            return new_main_log_probs, new_sap_log_probs

        if isinstance(model_out, FactorizedActorCriticOut):
            raise NotImplementedError

        assert_never(model_out)

    @staticmethod
    def build_data_augmenters(
        data_augmentations: list[DataAugmentation],
    ) -> list[DataAugmenter]:
        result: list[DataAugmenter] = []
        for augment in data_augmentations:
            match augment:
                case "rotate_180":
                    result.append(Rot180())
                case "player_reflect":
                    raise NotImplementedError
                case "drift_reflect":
                    raise NotImplementedError
                case _:
                    assert_never(augment)

        return result

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
