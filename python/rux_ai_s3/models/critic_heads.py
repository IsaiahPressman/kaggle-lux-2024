import torch
import torch.nn.functional as F
from torch import nn

from .types import ActivationFactory


class CriticBase(nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))


class BoundedCriticHead(nn.Module):
    def __init__(
        self,
        reward_min: float,
        reward_max: float,
        d_model: int,
        activation: ActivationFactory,
    ) -> None:
        super().__init__()
        self.reward_min = reward_min
        self.reward_max = reward_max
        self.critic_base = CriticBase(d_model, activation)
        self.linear = nn.Linear(
            in_features=d_model,
            out_features=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.critic_base(x)
        x = torch.flatten(x, start_dim=-2, end_dim=-1).mean(dim=-1)
        x = self.linear(x)
        return F.sigmoid(x) * (self.reward_max - self.reward_min) + self.reward_min


class PositiveUnboundedCriticHead(nn.Module):
    def __init__(
        self,
        reward_min: float,
        d_model: int,
        activation: ActivationFactory,
    ) -> None:
        super().__init__()
        self.reward_min = reward_min
        self.critic_base = CriticBase(d_model, activation)
        self.linear = nn.Linear(
            in_features=d_model,
            out_features=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.critic_base(x)
        x = torch.flatten(x, start_dim=-2, end_dim=-1).mean(dim=-1)
        x = self.linear(x)
        return F.relu(x) + self.reward_min


class ZeroSumCriticHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        activation: ActivationFactory,
    ) -> None:
        super().__init__()
        self.critic_base = CriticBase(d_model, activation)
        self.linear = nn.Linear(
            in_features=d_model,
            out_features=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Assumes both opposing player observations are passed in at once. \
        Expects input of shape (batch * 2 [n_players], d_model, w, h) \
        Returns output of shape (batch * 2 [n_players],)
        """
        x = self.critic_base(x)
        x = torch.flatten(x, start_dim=-2, end_dim=-1).mean(dim=-1)
        x = self.linear(x).view(-1, 2)
        return F.softmax(x, dim=-1).view(-1) * 2.0 - 1.0
