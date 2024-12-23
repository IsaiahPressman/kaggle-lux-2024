import torch
import torch.nn.functional as F
from torch import nn

from rux_ai_s3.types import Action

from .types import ActivationFactory, TorchActionInfo


class BasicActorHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        activation: ActivationFactory,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
        )
        self.activation = activation()
        self.main_actor = nn.Linear(
            in_features=d_model + 1,
            out_features=len(Action),
        )
        self.sap_actor = nn.Conv2d(
            in_channels=d_model,
            out_channels=1,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        action_info: TorchActionInfo,
        random_sample_main_actions: bool,
        random_sample_sap_actions: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: shape (batch, d_model, w, h) \
        unit_indices: shape (batch, units, 2) \
        unit_energies: shape (batch, units)

        Returns: (main_log_probs, sap_log_probs)
        main_log_probs: shape (batch, units, actions)
        sap_log_probs: shape (batch, units, w * h)
        main_actions: shape (batch, units)
        sap_actions: shape (batch, units)
        """
        x = self.activation(self.conv(x))
        batch_size, unit_count, _ = action_info.unit_indices.shape
        batch_indices = (
            torch.arange(batch_size).view(batch_size, 1).expand(-1, unit_count)
        )
        unit_slices = x[
            batch_indices,
            :,
            action_info.unit_indices[..., 0],
            action_info.unit_indices[..., 1],
        ]
        # unit_slices has shape (batch, units, d_model)
        unit_slices = torch.cat(
            [unit_slices, action_info.unit_energies.unsqueeze(-1)], dim=-1
        )
        main_logits = self.main_actor(unit_slices)
        sap_logits = torch.flatten(
            self.sap_actor(x).expand(-1, unit_count, -1, -1), start_dim=-2, end_dim=-1
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
            torch.flatten(action_info.sap_mask, start_dim=-2, end_dim=-1),
            action_info.units_mask,
        )
        sap_log_probs = F.log_softmax(masked_sap_logits, dim=-1)
        main_actions = self.log_probs_to_actions(
            main_log_probs, random_sample_main_actions
        )
        main_actions = torch.where(
            action_info.units_mask,
            main_actions,
            torch.zeros_like(main_actions),
        )
        sap_actions = self.log_probs_to_actions(
            sap_log_probs, random_sample_sap_actions
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
        random_sample_actions: bool,
    ) -> torch.Tensor:
        """
        Expects logits to be of shape (*, n_actions).
        Returns action tensor of shape (*,).
        """
        if not random_sample_actions:
            return log_probs.argsort(dim=-1, descending=True)[..., 0]

        probs = log_probs.exp().view(-1, log_probs.shape[-1])
        actions = torch.multinomial(
            probs,
            num_samples=1,
        )
        return actions.view(*log_probs.shape[:-1])
