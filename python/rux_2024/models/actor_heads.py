import torch
from torch import nn

from rux_2024.types import Action

from .types import ActivationFactory


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
        unit_indices: torch.Tensor,
        unit_energies: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: shape (batch, d_model, w, h) \
        unit_indices: shape (batch, units, 2) \
        unit_energies: shape (batch, units)

        Returns: (main_action_logits, sap_action_logits)
        main_action_logits: shape (batch, units, actions)
        sap_action_logits: shape (batch, units, w, h)
        """
        x = self.activation(self.conv(x))
        batch_size, unit_count, _ = unit_indices.shape
        batch_indices = (
            torch.arange(batch_size).view(batch_size, 1).expand(-1, unit_count)
        )
        unit_slices = x[batch_indices, :, unit_indices[..., 0], unit_indices[..., 1]]
        # unit_slices has shape (batch, units, d_model)
        unit_slices = torch.cat([unit_slices, unit_energies.unsqueeze(-1)], dim=-1)
        main_action_logits = self.main_actor(unit_slices)
        sap_action_logits = self.sap_actor(x).expand(-1, unit_count, -1, -1)
        return main_action_logits, sap_action_logits
