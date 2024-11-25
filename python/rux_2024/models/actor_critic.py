from typing import NamedTuple

import torch
from torch import nn

from rux_2024._lowlevel import RewardSpace

from .actor_heads import BasicActorHead
from .conv_blocks import ResidualBlock
from .types import ActivationFactory
from .utils import build_critic_head


class ActorCriticOut(NamedTuple):
    main_action_logits: torch.Tensor
    """shape (batch, units, actions)"""
    sap_action_logits: torch.Tensor
    """shape (batch, units, w, h)"""
    value: torch.Tensor
    """shape (batch * 2 [n_players],)"""


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
        spatial_obs: torch.Tensor,
        global_obs: torch.Tensor,
        unit_indices: torch.Tensor,
        unit_energies: torch.Tensor,
    ) -> ActorCriticOut:
        """
        spatial_obs: shape (batch, s_features, w, h) \
        global_obs: shape (batch, g_features) \
        unit_indices: shape (batch, units, 2) \
        unit_energies: shape (batch, units)
        """
        x = self.spatial_in(spatial_obs) + self.global_in(global_obs)[..., None, None]
        x = self.base(x)
        main_action_logits, sap_action_logits = self.actor_head(
            x, unit_indices=unit_indices, unit_energies=unit_energies
        )
        value = self.critic_head(x)
        return ActorCriticOut(
            main_action_logits=main_action_logits,
            sap_action_logits=sap_action_logits,
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
