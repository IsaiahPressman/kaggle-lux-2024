from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from rux_ai_s3._lowlevel import RewardSpace

from ..constants import MAP_SIZE, Action
from .actor_heads import BasicActorHead
from .conv_blocks import ResidualBlock
from .types import ActivationFactory, TorchActionInfo, TorchObs
from .utils import build_critic_head


class ActorCriticOut(NamedTuple):
    # TODO: Is there a better way to decide when to sap? With this setup,
    #  agents might be too conservative about sapping if they don't learn to
    #  appropriately condition the sap action probability on the sap range
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

    def to_env_actions(
        self,
        unit_indices: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int64]:
        """
        unit_indices shape: (batch, units, 2)
        Converts to actions array of shape (batch, units, 3)
        """
        return self._extract_env_actions(
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
        return self._extract_env_actions(
            self._add_player_dim(self.main_actions),
            self._add_player_dim(self.sap_actions),
            unit_indices,
        )

    def add_player_dim(self) -> "ActorCriticOut":
        return ActorCriticOut(*(self._add_player_dim(t) for t in self))

    def player_dim_flattened(self) -> "ActorCriticOut":
        return ActorCriticOut(*(torch.flatten(t, start_dim=0, end_dim=1) for t in self))

    def compute_joint_log_probs(self) -> torch.Tensor:
        main_log_probs = (
            self.main_log_probs.gather(dim=-1, index=self.main_actions.unsqueeze(-1))
            .squeeze(-1)
            .sum(dim=-1)
        )
        sap_log_probs = (
            self.sap_log_probs.gather(dim=-1, index=self.sap_actions.unsqueeze(-1))
            .squeeze(-1)
            .sum(dim=-1)
        )
        log_probs = main_log_probs + torch.where(
            self.main_actions == Action.SAP.value,
            sap_log_probs,
            torch.zeros_like(sap_log_probs),
        )
        return log_probs.sum(dim=-1)

    @staticmethod
    def _extract_env_actions(
        main_actions: torch.Tensor,
        sap_actions: torch.Tensor,
        unit_indices: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int64]:
        sap_targets = np.divmod(sap_actions.cpu().numpy(), MAP_SIZE)
        actions = np.stack(
            [main_actions.cpu().numpy(), *sap_targets],
            axis=-1,
        )
        actions[..., 1:] -= unit_indices
        return actions

    @staticmethod
    def _add_player_dim(t: torch.Tensor) -> torch.Tensor:
        batch, *shape = t.shape
        return t.view(batch // 2, 2, *shape)


class ActorCritic(nn.Module):
    def __init__(
        self,
        spatial_in_channels: int,
        global_in_channels: int,
        d_model: int,
        n_layers: int,
        reward_space: RewardSpace,
        kernel_size: int = 3,
        activation: ActivationFactory = nn.GELU,
    ) -> None:
        super().__init__()
        self.spatial_in = self._build_spatial_in(
            spatial_in_channels,
            d_model,
            activation=activation,
            kernel_size=kernel_size,
        )
        self.global_in = self._build_global_in(
            global_in_channels,
            d_model,
            activation=activation,
        )
        self.base = self._build_base(
            d_model=d_model,
            n_blocks=n_layers,
            activation=activation,
            kernel_size=kernel_size,
        )
        self.actor_head = BasicActorHead(
            d_model,
            activation=activation,
        )
        self.critic_head = build_critic_head(
            reward_space,
            d_model,
            activation=activation,
        )

    def forward(
        self,
        obs: TorchObs,
        action_info: TorchActionInfo,
        random_sample_actions: bool,
    ) -> ActorCriticOut:
        """
        spatial_obs: shape (batch, s_features, w, h) \
        global_obs: shape (batch, g_features) \
        unit_indices: shape (batch, units, 2) \
        unit_energies: shape (batch, units)
        """
        x = (
            self.spatial_in(obs.spatial_obs)
            + self.global_in(obs.global_obs)[..., None, None]
        )
        x = self.base(x)
        main_log_probs, sap_log_probs, main_actions, sap_actions = self.actor_head(
            x, action_info, random_sample_actions=random_sample_actions
        )
        value = self.critic_head(x)
        return ActorCriticOut(
            main_log_probs=main_log_probs,
            sap_log_probs=sap_log_probs,
            main_actions=main_actions,
            sap_actions=sap_actions,
            value=value,
        )

    @classmethod
    def _build_spatial_in(
        cls,
        in_channels: int,
        d_model: int,
        activation: ActivationFactory,
        kernel_size: int,
    ) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=d_model,
                kernel_size=kernel_size,
                padding="same",
            ),
            activation(),
            nn.Conv2d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=kernel_size,
                padding="same",
            ),
        )

    @classmethod
    def _build_global_in(
        cls, in_channels: int, d_model: int, activation: ActivationFactory
    ) -> nn.Module:
        return nn.Sequential(
            nn.Linear(
                in_features=in_channels,
                out_features=d_model,
            ),
            activation(),
            nn.Linear(
                in_features=d_model,
                out_features=d_model,
            ),
        )

    @classmethod
    def _build_base(
        cls,
        d_model: int,
        n_blocks: int,
        activation: ActivationFactory,
        kernel_size: int,
    ) -> nn.Module:
        return nn.Sequential(
            *(
                ResidualBlock(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    activation=activation(),
                    squeeze_excitation=True,
                )
                for _ in range(n_blocks)
            )
        )
