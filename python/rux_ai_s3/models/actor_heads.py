from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from rux_ai_s3.types import Action

from .types import ActivationFactory, TorchActionInfo
from .utils import get_unit_slices
from .weight_initialization import orthogonal_initialization_


class ActionConfig(NamedTuple):
    main_action_temperature: float | None
    sap_action_temperature: float | None


class BasicActorHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        activation: ActivationFactory,
    ) -> None:
        super().__init__()
        self.main_actor_linear = nn.Sequential(
            nn.Linear(
                in_features=d_model + 1,
                out_features=d_model,
            ),
            activation(),
            nn.Linear(
                in_features=d_model,
                out_features=len(Action),
            ),
        )
        self.sap_actor_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=1,
            ),
            activation(),
            nn.Conv2d(
                in_channels=d_model,
                out_channels=1,
                kernel_size=1,
            ),
        )
        # self._init_weights()  # noqa: ERA001

    def _init_weights(self) -> None:
        self.apply(orthogonal_initialization_)
        orthogonal_initialization_(
            list(self.main_actor_linear.children())[-1], scale=0.01, strict=True
        )
        orthogonal_initialization_(
            list(self.sap_actor_conv.children())[-1], scale=0.01, strict=True
        )

    def forward(
        self,
        x: torch.Tensor,
        action_info: TorchActionInfo,
        action_config: ActionConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: shape (batch, d_model, w, h)

        Returns: (main_log_probs, sap_log_probs, main_actions, sap_actions)
        main_log_probs: shape (batch, units, actions)
        sap_log_probs: shape (batch, units, w * h)
        main_actions: shape (batch, units)
        sap_actions: shape (batch, units)
        """
        unit_slices = get_unit_slices(x, action_info)
        unit_count = unit_slices.shape[1]
        main_logits = self.main_actor_linear(unit_slices)
        sap_logits = (
            self.sap_actor_conv(x)
            .expand(-1, unit_count, -1, -1)
            .flatten(start_dim=-2, end_dim=-1)
        )

        main_log_probs = F.log_softmax(
            self.safe_mask_logits(
                main_logits,
                action_info.main_mask,
                action_info.units_mask,
            ),
            dim=-1,
        )
        masked_sap_logits = self.safe_mask_logits(
            sap_logits,
            action_info.sap_mask.flatten(start_dim=-2, end_dim=-1),
            action_info.units_mask,
        )
        sap_log_probs = F.log_softmax(masked_sap_logits, dim=-1)
        main_actions = self.log_probs_to_actions(
            main_log_probs, action_config.main_action_temperature
        )
        main_actions = torch.where(
            action_info.units_mask,
            main_actions,
            torch.zeros_like(main_actions),
        )
        sap_actions = self.log_probs_to_actions(
            sap_log_probs, action_config.sap_action_temperature
        )
        return main_log_probs, sap_log_probs, main_actions, sap_actions

    @staticmethod
    def safe_mask_logits(
        logits: torch.Tensor,
        available_actions_mask: torch.Tensor,
        available_units_mask: torch.Tensor,
    ) -> torch.Tensor:
        infs_mask = torch.where(
            available_actions_mask,
            0,
            -torch.inf,
        )
        return torch.where(
            available_units_mask.unsqueeze(-1)
            & available_actions_mask.any(dim=-1, keepdim=True),
            logits + infs_mask,
            torch.zeros_like(logits),
        )

    @staticmethod
    @torch.no_grad()
    def log_probs_to_actions(
        log_probs: torch.Tensor,
        temperature: float | None,
    ) -> torch.Tensor:
        """
        Expects logits to be of shape (*, n_actions).
        Returns action tensor of shape (*,).
        """
        if temperature is None:
            probs = log_probs.exp().view(-1, log_probs.shape[-1])
            actions = torch.multinomial(
                probs,
                num_samples=1,
            )
            return actions.view(*log_probs.shape[:-1])

        if temperature < 0.0:
            raise ValueError(f"Invalid temperature {temperature} < 0")

        if temperature == 0.0:
            return log_probs.argmax(dim=-1)

        probs = F.softmax(log_probs / temperature, dim=-1).view(-1, log_probs.shape[-1])
        actions = torch.multinomial(
            probs,
            num_samples=1,
        )
        return actions.view(*log_probs.shape[:-1])
