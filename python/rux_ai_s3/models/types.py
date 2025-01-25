from collections.abc import Callable
from typing import NamedTuple

import torch
from torch import nn

from rux_ai_s3.types import ActionInfo, FrameStackedObs

ActivationFactory = Callable[[], nn.Module]


class TorchObs(NamedTuple):
    spatial_obs: torch.Tensor
    global_obs: torch.Tensor

    def flatten(self, start_dim: int, end_dim: int) -> "TorchObs":
        return TorchObs(
            *(t.flatten(start_dim=start_dim, end_dim=end_dim) for t in self)
        )

    def to_device(self, device: torch.device) -> "TorchObs":
        return TorchObs(*(t.to(device) for t in self))

    def index_player(self, p: int) -> "TorchObs":
        for t in self:
            assert t.shape[1] == 2

        return TorchObs(*(t[:, p] for t in self))

    @classmethod
    def from_numpy(cls, obs: FrameStackedObs, device: torch.device) -> "TorchObs":
        return TorchObs(
            **{
                key: torch.from_numpy(val).to(device)
                for key, val in obs._asdict().items()
            }
        )


class TorchActionInfo(NamedTuple):
    main_mask: torch.Tensor
    sap_mask: torch.Tensor
    unit_indices: torch.Tensor
    unit_energies: torch.Tensor
    units_mask: torch.Tensor

    def flatten(self, start_dim: int, end_dim: int) -> "TorchActionInfo":
        return TorchActionInfo(
            *(t.flatten(start_dim=start_dim, end_dim=end_dim) for t in self)
        )

    def to_device(self, device: torch.device) -> "TorchActionInfo":
        return TorchActionInfo(*(t.to(device) for t in self))

    def index_player(self, p: int) -> "TorchActionInfo":
        for t in self:
            assert t.shape[1] == 2

        return TorchActionInfo(*(t[:, p] for t in self))

    @classmethod
    def from_numpy(
        cls, action_info: ActionInfo, device: torch.device
    ) -> "TorchActionInfo":
        return TorchActionInfo(
            **{
                key: torch.from_numpy(val).to(device)
                for key, val in action_info._asdict().items()
            }
        )
