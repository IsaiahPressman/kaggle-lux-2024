from abc import ABC, abstractmethod

import torch

from rux_ai_s3.models.actor_critic import ActorCriticOut
from rux_ai_s3.types import Action

MoveActionMapType = tuple[
    tuple[Action, Action],
    tuple[Action, Action],
    tuple[Action, Action],
    tuple[Action, Action],
]


class DataAugmenter(ABC):
    @abstractmethod
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def transform_coordinates(
        self, coordinates: torch.Tensor, map_size: int
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def get_move_action_map(self) -> MoveActionMapType:
        """
        A list of (original, transformed) action pairs
        """

    def inverse_transform_actions(self, actions: ActorCriticOut) -> ActorCriticOut:
        """
        Updates main_log_probs and sap_log_probs, and removes actions
        themselves, since those can be resampled from the updated log probs.
        """
        new_main_log_probs = actions.main_log_probs.clone()
        for orig, transformed in self.get_move_action_map():
            new_main_log_probs[..., orig] = actions.main_log_probs[..., transformed]

        updated_sap_log_probs = self.inverse_transform(actions.sap_log_probs)
        return ActorCriticOut(
            main_log_probs=new_main_log_probs,
            sap_log_probs=updated_sap_log_probs,
            main_actions=torch.empty(0),
            sap_actions=torch.empty(0),
            value=actions.value,
        )


class Rot180(DataAugmenter):
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        _, width, height = x.shape
        assert width == height, "input must be square"
        return torch.rot90(x, 2, (1, 2))

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(x)

    def transform_coordinates(
        self, coordinates: torch.Tensor, map_size: int
    ) -> torch.Tensor:
        assert coordinates.ndim == 2
        assert coordinates.shape[-1] == 2
        assert torch.all((coordinates >= 0) & (coordinates < map_size))
        return map_size - 1 - coordinates

    def get_move_action_map(self) -> MoveActionMapType:
        return (
            (Action.UP, Action.DOWN),
            (Action.RIGHT, Action.LEFT),
            (Action.DOWN, Action.UP),
            (Action.LEFT, Action.RIGHT),
        )
