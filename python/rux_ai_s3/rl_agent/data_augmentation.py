from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch

from rux_ai_s3.constants import NEBULA_TILE_DRIFT_DIRECTION_FEATURE_INDEX
from rux_ai_s3.models.actor_critic import ActorCriticOut
from rux_ai_s3.types import Action

MoveActionMapType = Sequence[tuple[Action, Action]]


class DataAugmenter(ABC):
    @property
    @abstractmethod
    def drift_direction_flipped(self) -> bool:
        pass

    @abstractmethod
    def transform_spatial(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def inverse_transform_spatial(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def transform_coordinates(
        self, coordinates: torch.Tensor, map_size: int
    ) -> torch.Tensor:
        pass

    @staticmethod
    @abstractmethod
    def get_move_action_map() -> MoveActionMapType:
        """
        A list of (original, transformed) action pairs
        """

    def transform_global_obs(
        self,
        global_obs: torch.Tensor,
        frame_stack_len: int,
        temporal_feature_count: int,
    ) -> torch.Tensor:
        if not self.drift_direction_flipped:
            return global_obs

        global_obs = global_obs.clone()
        neg_drift = (
            frame_stack_len * temporal_feature_count
            + NEBULA_TILE_DRIFT_DIRECTION_FEATURE_INDEX
        )
        pos_drift = neg_drift + 1
        neg_drift_value = global_obs[..., neg_drift].item()
        pos_drift_value = global_obs[..., pos_drift].item()
        if neg_drift_value == 0.0:
            assert pos_drift_value == 1.0
        elif neg_drift_value == 0.5:
            assert pos_drift_value == 0.5
        elif neg_drift_value == 1.0:
            assert pos_drift_value == 0.0
        else:
            raise ValueError(f"Unexpected negative drift value: {neg_drift_value}")

        global_obs[..., neg_drift] = pos_drift_value
        global_obs[..., pos_drift] = neg_drift_value
        return global_obs

    def transform_action_space(self, action_space: torch.Tensor) -> torch.Tensor:
        return self.inner_transform_action_space(
            action_space,
            self.get_move_action_map(),
        )

    def inverse_transform_action_space(
        self, action_space: torch.Tensor
    ) -> torch.Tensor:
        return self.inner_inverse_transform_action_space(
            action_space,
            self.get_move_action_map(),
        )

    def inverse_transform_actions(self, actions: ActorCriticOut) -> ActorCriticOut:
        """
        Updates main_log_probs and sap_log_probs, and removes actions
        themselves, since those can be resampled from the updated log probs.
        """
        new_main_log_probs = actions.main_log_probs.clone()
        for orig, transformed in self.get_move_action_map():
            new_main_log_probs[..., orig] = actions.main_log_probs[..., transformed]

        updated_sap_log_probs = self.inverse_transform_spatial(actions.sap_log_probs)
        return ActorCriticOut(
            main_log_probs=new_main_log_probs,
            sap_log_probs=updated_sap_log_probs,
            main_actions=torch.empty(0),
            sap_actions=torch.empty(0),
            value=actions.value,
        )

    @classmethod
    def inner_transform_action_space(
        cls,
        action_space: torch.Tensor,
        move_action_map: MoveActionMapType,
    ) -> torch.Tensor:
        new_action_space = action_space.clone()
        for orig, transformed in move_action_map:
            new_action_space[..., transformed] = action_space[..., orig]

        return new_action_space

    @classmethod
    def inner_inverse_transform_action_space(
        cls,
        action_space: torch.Tensor,
        move_action_map: MoveActionMapType,
    ) -> torch.Tensor:
        inverted_action_map = [(b, a) for a, b in move_action_map]
        return cls.inner_transform_action_space(
            action_space,
            inverted_action_map,
        )

    @staticmethod
    def _validate_spatial(x: torch.Tensor) -> None:
        width, height = x.shape[-2:]
        assert width == height, "input must be square"

    @staticmethod
    def _validate_coordinates(coordinates: torch.Tensor, map_size: int) -> None:
        assert coordinates.shape[-1] == 2
        assert torch.all((coordinates >= 0) & (coordinates < map_size))


class Rot180(DataAugmenter):
    @property
    def drift_direction_flipped(self) -> bool:
        return True

    def transform_spatial(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_spatial(x)
        return x.rot90(2, (-2, -1))

    def inverse_transform_spatial(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform_spatial(x)

    def transform_coordinates(
        self, coordinates: torch.Tensor, map_size: int
    ) -> torch.Tensor:
        self._validate_coordinates(coordinates, map_size)
        return map_size - 1 - coordinates

    @staticmethod
    def get_move_action_map() -> MoveActionMapType:
        return (
            (Action.UP, Action.DOWN),
            (Action.RIGHT, Action.LEFT),
            (Action.DOWN, Action.UP),
            (Action.LEFT, Action.RIGHT),
        )


class PlayerReflect(DataAugmenter):
    @property
    def drift_direction_flipped(self) -> bool:
        return False

    def transform_spatial(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_spatial(x)
        return x.flip((-2, -1)).transpose(-2, -1)

    def inverse_transform_spatial(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform_spatial(x)

    def transform_coordinates(
        self, coordinates: torch.Tensor, map_size: int
    ) -> torch.Tensor:
        self._validate_coordinates(coordinates, map_size)
        return map_size - 1 - coordinates.flip(-1)

    @staticmethod
    def get_move_action_map() -> MoveActionMapType:
        return (
            (Action.UP, Action.RIGHT),
            (Action.RIGHT, Action.UP),
            (Action.DOWN, Action.LEFT),
            (Action.LEFT, Action.DOWN),
        )


class DriftReflect(DataAugmenter):
    @property
    def drift_direction_flipped(self) -> bool:
        return True

    def transform_spatial(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_spatial(x)
        return x.transpose(-2, -1)

    def inverse_transform_spatial(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform_spatial(x)

    def transform_coordinates(
        self, coordinates: torch.Tensor, map_size: int
    ) -> torch.Tensor:
        self._validate_coordinates(coordinates, map_size)
        return coordinates.flip(-1)

    @staticmethod
    def get_move_action_map() -> MoveActionMapType:
        return (
            (Action.UP, Action.LEFT),
            (Action.RIGHT, Action.DOWN),
            (Action.DOWN, Action.RIGHT),
            (Action.LEFT, Action.UP),
        )
