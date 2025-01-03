import torch

from .types import TorchActionInfo


def get_unit_slices(x: torch.Tensor, action_info: TorchActionInfo) -> torch.Tensor:
    """
    x: shape (batch, d_model, w, h)

    Returns: unit_slices of shape (batch, units, d_model + 1 [unit energy])
    """
    batch_size, unit_count, _ = action_info.unit_indices.shape
    batch_indices = torch.arange(batch_size).view(batch_size, 1).expand(-1, unit_count)
    unit_slices = x[
        batch_indices,
        :,
        action_info.unit_indices[..., 0],
        action_info.unit_indices[..., 1],
    ]
    return torch.cat([unit_slices, action_info.unit_energies.unsqueeze(-1)], dim=-1)
