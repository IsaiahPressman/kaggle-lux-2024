from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn

from .types import ActivationFactory, TorchActionInfo
from .utils import get_unit_slices


class BaseCriticHead(nn.Module, ABC):
    def __init__(
        self,
        d_model: int,
        activation: ActivationFactory,
    ) -> None:
        super().__init__()
        self.conv_base = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
        )
        self.activation = activation()
        self.conv_value = nn.Conv2d(
            in_channels=d_model,
            out_channels=1,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor, _action_info: TorchActionInfo) -> torch.Tensor:
        x = self.activation(self.conv_base(x))
        x = (
            self.conv_value(x)
            .flatten(start_dim=-2, end_dim=-1)
            .mean(dim=-1)
            .squeeze(dim=-1)
        )
        return self.postprocess_value(x)

    @abstractmethod
    def postprocess_value(self, x: torch.Tensor) -> torch.Tensor:
        """Expects and returns a tensor of shape (batch,)"""


class BoundedCriticHead(BaseCriticHead):
    def __init__(
        self,
        reward_min: float,
        reward_max: float,
        d_model: int,
        activation: ActivationFactory,
    ) -> None:
        super().__init__(d_model, activation)
        self.reward_min = reward_min
        self.reward_max = reward_max

    def postprocess_value(self, x: torch.Tensor) -> torch.Tensor:
        return F.sigmoid(x) * (self.reward_max - self.reward_min) + self.reward_min


class PositiveUnboundedCriticHead(BaseCriticHead):
    def __init__(
        self,
        reward_min: float,
        d_model: int,
        activation: ActivationFactory,
    ) -> None:
        super().__init__(d_model, activation)
        self.reward_min = reward_min

    def postprocess_value(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x) + self.reward_min


class ZeroSumCriticHead(BaseCriticHead):
    def __init__(
        self,
        d_model: int,
        activation: ActivationFactory,
    ) -> None:
        super().__init__(d_model, activation)

    def postprocess_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Assumes both opposing player observations are passed in at once. \
        Expects input of shape (batch * 2 [n_players],) \
        Returns output of shape (batch * 2 [n_players],)
        """
        return F.softmax(x.view(-1, 2), dim=-1).view(-1) * 2.0 - 1.0


class BaseFactorizedCriticHead(nn.Module, ABC):
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
        self.baseline_conv = nn.Linear(
            in_features=d_model,
            out_features=1,
        )
        self.units_linear = nn.Linear(
            in_features=d_model + 1,
            out_features=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        action_info: TorchActionInfo,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.activation(self.conv(x))
        # baseline value shape (batch,)
        baseline_value = (
            self.baseline_conv(x)
            .flatten(start_dim=-2, end_dim=-1)
            .mean(dim=-1)
            .squeeze(dim=-1)
        )
        # unit_slices shape (batch, units, d_model + 1)
        unit_slices = get_unit_slices(x, action_info)
        # factorized_value shape (batch, units)
        factorized_value = self.units_linear(unit_slices).squeeze(dim=-1)
        return self.postprocess_value(baseline_value), self.postprocess_value(
            factorized_value
        )

    @abstractmethod
    def postprocess_value(self, x: torch.Tensor) -> torch.Tensor:
        """Expects and returns a tensor of shape (batch,)"""


class BoundedFactorizedCriticHead(BaseFactorizedCriticHead):
    def __init__(
        self,
        reward_min: float,
        reward_max: float,
        d_model: int,
        activation: ActivationFactory,
    ) -> None:
        super().__init__(d_model, activation)
        self.reward_min = reward_min
        self.reward_max = reward_max

    def postprocess_value(self, x: torch.Tensor) -> torch.Tensor:
        return F.sigmoid(x) * (self.reward_max - self.reward_min) + self.reward_min
