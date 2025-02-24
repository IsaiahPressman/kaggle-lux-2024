from collections.abc import Callable

import torch
from torch import nn

from rux_ai_s3.models.actor_heads import BasicActorHead
from rux_ai_s3.models.critic_heads import BaseCriticHead, BaseFactorizedCriticHead
from rux_ai_s3.models.types import TorchActionInfo, TorchObs
from .out import ActorCriticOut, FactorizedActorCriticOut


class ActorCritic(nn.Module):
    __call__: Callable[..., ActorCriticOut]

    def __init__(
        self,
        base: nn.Module,
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

    @staticmethod
    @torch.no_grad()
    def log_probs_to_actions(
        log_probs: torch.Tensor, temperature: float | None
    ) -> torch.Tensor:
        return BasicActorHead.log_probs_to_actions(log_probs, temperature)


class FactorizedActorCritic(nn.Module):
    __call__: Callable[..., FactorizedActorCriticOut]

    def __init__(
        self,
        base: nn.Module,
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

    @staticmethod
    @torch.no_grad()
    def log_probs_to_actions(
        log_probs: torch.Tensor, temperature: float | None
    ) -> torch.Tensor:
        return BasicActorHead.log_probs_to_actions(log_probs, temperature)
