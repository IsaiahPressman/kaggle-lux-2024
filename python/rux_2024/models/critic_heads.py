import torch
import torch.nn.functional as F
from torch import nn

from .types import ActivationFactory


class ZeroSumCriticHead(nn.Module):
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
        self.linear = nn.Linear(
            in_features=d_model,
            out_features=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expects input of shape (batch, 2, d_model, width, height)
        Returns output of shape (batch, 2)
        """
        batch, p, _, _, _ = x.shape
        assert p == 2
        x = torch.flatten(x, start_dim=0, end_dim=1)
        x = self.activation(self.conv(x))
        x = torch.flatten(x, start_dim=-2, end_dim=-1).mean(dim=-1)
        x = self.linear(x).view(batch, p)
        return F.softmax(x, dim=-1)
