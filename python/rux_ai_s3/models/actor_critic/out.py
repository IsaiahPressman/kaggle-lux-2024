from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import torch

from rux_ai_s3.constants import MAP_SIZE
from rux_ai_s3.types import Action


class ActorCriticOut(NamedTuple):
    main_log_probs: torch.Tensor
    """shape (batch, [players,] units, actions)"""
    sap_log_probs: torch.Tensor
    """shape (batch, [players,] units, w * h)"""
    main_actions: torch.Tensor
    """shape (batch, [players,] units)"""
    sap_actions: torch.Tensor
    """shape (batch, [players,] units)"""
    value: torch.Tensor
    """shape (batch, [players,])"""

    def to_device(self, device: torch.device) -> "ActorCriticOut":
        return ActorCriticOut(*(t.to(device, non_blocking=True) for t in self))

    def to_env_actions(
        self,
        unit_indices: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int64]:
        """
        unit_indices shape: (batch, units, 2)
        Converts to actions array of shape (batch, units, 3)
        """
        return extract_env_actions(
            self.main_actions,
            self.sap_actions,
            unit_indices,
        )

    def to_player_env_actions(
        self,
        unit_indices: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int64]:
        """
        unit_indices_shape: (batch / P, P, units, 2)
        Converts to array of per-player actions with shape (batch / P, P, units, 3)
        """
        return extract_env_actions(
            _add_player_dim(self.main_actions),
            _add_player_dim(self.sap_actions),
            unit_indices,
        )

    def add_player_dim(self) -> "ActorCriticOut":
        return ActorCriticOut(*(_add_player_dim(t) for t in self))

    def flatten(self, start_dim: int, end_dim: int) -> "ActorCriticOut":
        return ActorCriticOut(
            *(t.flatten(start_dim=start_dim, end_dim=end_dim) for t in self)
        )

    def compute_joint_log_probs(self) -> torch.Tensor:
        main_log_probs = self.main_log_probs.gather(
            dim=-1, index=self.main_actions.unsqueeze(-1)
        ).squeeze(-1)
        sap_log_probs = self.sap_log_probs.gather(
            dim=-1, index=self.sap_actions.unsqueeze(-1)
        ).squeeze(-1)
        log_probs = main_log_probs + torch.where(
            self.main_actions >= Action.SAP,
            sap_log_probs,
            torch.zeros_like(sap_log_probs),
        )
        assert log_probs.ndim == 2
        return log_probs.sum(dim=-1)


class FactorizedActorCriticOut(NamedTuple):
    main_log_probs: torch.Tensor
    """shape (batch, [players,] units, actions)"""
    sap_log_probs: torch.Tensor
    """shape (batch, [players,] units, w * h)"""
    main_actions: torch.Tensor
    """shape (batch, [players,] units)"""
    sap_actions: torch.Tensor
    """shape (batch, [players,] units)"""
    baseline_value: torch.Tensor
    """shape (batch, [players,])"""
    factorized_value: torch.Tensor
    """shape (batch, [players,] units)"""

    def to_device(self, device: torch.device) -> "FactorizedActorCriticOut":
        return FactorizedActorCriticOut(
            *(t.to(device, non_blocking=True) for t in self)
        )

    def to_env_actions(
        self,
        unit_indices: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int64]:
        """
        unit_indices shape: (batch, units, 2)
        Converts to actions array of shape (batch, units, 3)
        """
        return extract_env_actions(
            self.main_actions,
            self.sap_actions,
            unit_indices,
        )

    def to_player_env_actions(
        self,
        unit_indices: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int64]:
        """
        unit_indices_shape: (batch / P, P, units, 2)
        Converts to array of per-player actions with shape (batch / P, P, units, 3)
        """
        return extract_env_actions(
            _add_player_dim(self.main_actions),
            _add_player_dim(self.sap_actions),
            unit_indices,
        )

    def add_player_dim(self) -> "FactorizedActorCriticOut":
        return FactorizedActorCriticOut(*(_add_player_dim(t) for t in self))

    def flatten(self, start_dim: int, end_dim: int) -> "FactorizedActorCriticOut":
        return FactorizedActorCriticOut(
            *(t.flatten(start_dim=start_dim, end_dim=end_dim) for t in self)
        )

    def compute_unit_value(self, units_mask: torch.Tensor) -> torch.Tensor:
        return torch.where(
            units_mask,
            self.factorized_value,
            self.compute_agent_value(units_mask).unsqueeze(dim=-1),
        )

    def compute_agent_value(self, units_mask: torch.Tensor) -> torch.Tensor:
        assert units_mask.shape == self.factorized_value.shape
        assert units_mask.dtype == torch.bool
        units_mask_float = units_mask.float()
        masked_summed_value = (units_mask_float * self.factorized_value).sum(dim=-1)
        unit_alive_count = units_mask_float.sum(dim=-1)
        return torch.where(
            units_mask.any(dim=-1),
            masked_summed_value / unit_alive_count.clamp_min(1.0),
            self.baseline_value,
        )

    def get_unit_log_probs(self) -> torch.Tensor:
        main_log_probs = self.main_log_probs.gather(
            dim=-1, index=self.main_actions.unsqueeze(-1)
        ).squeeze(-1)
        sap_log_probs = self.sap_log_probs.gather(
            dim=-1, index=self.sap_actions.unsqueeze(-1)
        ).squeeze(-1)
        log_probs = main_log_probs + torch.where(
            self.main_actions >= Action.SAP,
            sap_log_probs,
            torch.zeros_like(sap_log_probs),
        )
        assert log_probs.ndim == 2
        return log_probs


def extract_env_actions(
    main_actions: torch.Tensor,
    sap_actions: torch.Tensor,
    unit_indices: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    main_actions = torch.where(
        main_actions >= Action.SAP,
        Action.SAP,
        main_actions,
    )
    sap_targets = np.divmod(sap_actions.cpu().numpy(), MAP_SIZE)
    actions = np.stack(
        [main_actions.cpu().numpy(), *sap_targets],
        axis=-1,
    )
    actions[..., 1:] -= unit_indices
    return actions


def _add_player_dim(t: torch.Tensor) -> torch.Tensor:
    batch, *shape = t.shape
    return t.view(batch // 2, 2, *shape)
