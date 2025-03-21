import torch
from torch import nn

from .types import ActivationFactory


class SELayer(nn.Module):
    def __init__(self, n_channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_channels, n_channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(n_channels // reduction, n_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = x.flatten(start_dim=-2, end_dim=-1).mean(dim=-1)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: ActivationFactory,
        kernel_size: int,
        dropout: float | None,
        squeeze_excitation: bool,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        self.act1 = activation()

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
        )
        self.act2 = activation()

        if in_channels != out_channels:
            self.change_n_channels: nn.Module = nn.Conv2d(
                in_channels, out_channels, kernel_size=1
            )
        else:
            self.change_n_channels = nn.Identity()

        self.dropout = nn.Dropout2d(dropout or 0.0)
        self.squeeze_excitation = (
            SELayer(out_channels) if squeeze_excitation else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.squeeze_excitation(x)
        x = self.dropout(x) + self.change_n_channels(identity)
        return self.act2(x)
