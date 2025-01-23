from collections.abc import Callable

import torch
from torch import nn

from rux_ai_s3.models.actor_heads import BasicActorHead
from rux_ai_s3.models.conv_blocks import ResidualBlock
from rux_ai_s3.models.critic_heads import BaseCriticHead, BaseFactorizedCriticHead
from rux_ai_s3.models.types import ActivationFactory, TorchActionInfo, TorchObs

from ..weight_initialization import orthogonal_initialization_
from .out import ActorCriticOut, FactorizedActorCriticOut


class ActorCriticBase(nn.Module):
    def __init__(
        self,
        spatial_in_channels: int,
        global_in_channels: int,
        d_model: int,
        n_blocks: int,
        kernel_size: int,
        activation: ActivationFactory,
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
            n_blocks=n_blocks,
            activation=activation,
            kernel_size=kernel_size,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        self.apply(orthogonal_initialization_)

    def forward(
        self,
        obs: TorchObs,
    ) -> torch.Tensor:
        x = (
            self.spatial_in(obs.spatial_obs)
            + self.global_in(obs.global_obs)[..., None, None]
        )
        return self.base(x)

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
                    activation=activation,
                    kernel_size=kernel_size,
                    squeeze_excitation=True,
                )
                for _ in range(n_blocks)
            )
        )


class ActorCritic(nn.Module):
    __call__: Callable[..., ActorCriticOut]

    def __init__(
        self,
        base: ActorCriticBase,
        actor_head: BasicActorHead,
        critic_head: BaseCriticHead,
    ) -> None:
        super().__init__()
        self.base = base
        self.actor_head = actor_head
        self.critic_head = critic_head

    def forward(
        self,
        obs: TorchObs,
        action_info: TorchActionInfo,
        main_action_temperature: float | None = None,
        sap_action_temperature: float | None = None,
        omit_value: bool = False,
    ) -> ActorCriticOut:
        x = self.base(obs)
        main_log_probs, sap_log_probs, main_actions, sap_actions = self.actor_head(
            x,
            action_info,
            main_action_temperature=main_action_temperature,
            sap_action_temperature=sap_action_temperature,
        )
        if omit_value:
            assert not self.training
            value = torch.zeros(
                main_log_probs.shape[0],
                device=main_log_probs.device,
                dtype=main_log_probs.dtype,
            )
        else:
            value = self.critic_head(x, action_info)

        return ActorCriticOut(
            main_log_probs=main_log_probs,
            sap_log_probs=sap_log_probs,
            main_actions=main_actions,
            sap_actions=sap_actions,
            value=value,
        )


class FactorizedActorCritic(nn.Module):
    __call__: Callable[..., FactorizedActorCriticOut]

    def __init__(
        self,
        base: ActorCriticBase,
        actor_head: BasicActorHead,
        critic_head: BaseFactorizedCriticHead,
    ) -> None:
        super().__init__()
        self.base = base
        self.actor_head = actor_head
        self.critic_head = critic_head

    def forward(
        self,
        obs: TorchObs,
        action_info: TorchActionInfo,
        main_action_temperature: float | None = None,
        sap_action_temperature: float | None = None,
        omit_value: bool = False,
    ) -> FactorizedActorCriticOut:
        x = self.base(obs)
        main_log_probs, sap_log_probs, main_actions, sap_actions = self.actor_head(
            x,
            action_info,
            main_action_temperature=main_action_temperature,
            sap_action_temperature=sap_action_temperature,
        )
        if omit_value:
            assert not self.training
            baseline_value = torch.zeros(
                main_log_probs.shape[0],
                device=main_log_probs.device,
                dtype=main_log_probs.dtype,
            )
            factorized_value = torch.zeros(
                main_log_probs.shape[:2],
                device=main_log_probs.device,
                dtype=main_log_probs.dtype,
            )
        else:
            baseline_value, factorized_value = self.critic_head(x, action_info)

        return FactorizedActorCriticOut(
            main_log_probs=main_log_probs,
            sap_log_probs=sap_log_probs,
            main_actions=main_actions,
            sap_actions=sap_actions,
            baseline_value=baseline_value,
            factorized_value=factorized_value,
        )
