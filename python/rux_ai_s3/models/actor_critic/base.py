import torch
from torch import nn

from rux_ai_s3.models.attn_blocks import AttnBlock
from rux_ai_s3.models.conv_blocks import ResidualConvBlock
from rux_ai_s3.models.types import ActivationFactory, TorchObs
from rux_ai_s3.models.weight_initialization import orthogonal_initialization_


class ActorCriticConvBase(nn.Module):
    def __init__(
        self,
        spatial_in_channels: int,
        global_in_channels: int,
        d_model: int,
        n_blocks: int,
        kernel_size: int,
        dropout: float | None,
        activation: ActivationFactory,
    ) -> None:
        super().__init__()
        self.spatial_in = self._build_spatial_in(
            spatial_in_channels,
            d_model,
            activation=activation,
            kernel_size=kernel_size,
        )
        self.global_in = self._build_global_in(
            global_in_channels,
            d_model,
            activation=activation,
        )
        self.base = self._build_base(
            d_model=d_model,
            n_blocks=n_blocks,
            activation=activation,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        self.apply(orthogonal_initialization_)

    def forward(
            self,
            obs: TorchObs,
    ) -> torch.Tensor:
        x = (
                self.spatial_in(obs.spatial_obs)
                + self.global_in(obs.global_obs)[..., None, None]
        )
        return self.base(x)

    @classmethod
    def _build_spatial_in(
            cls,
            in_channels: int,
            d_model: int,
            activation: ActivationFactory,
            kernel_size: int,
    ) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=d_model,
                kernel_size=kernel_size,
                padding="same",
            ),
            activation(),
            nn.Conv2d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=kernel_size,
                padding="same",
            ),
        )

    @classmethod
    def _build_global_in(
            cls, in_channels: int, d_model: int, activation: ActivationFactory
    ) -> nn.Module:
        return nn.Sequential(
            nn.Linear(
                in_features=in_channels,
                out_features=d_model,
            ),
            activation(),
            nn.Linear(
                in_features=d_model,
                out_features=d_model,
            ),
        )

    @classmethod
    def _build_base(
            cls,
            d_model: int,
            n_blocks: int,
            activation: ActivationFactory,
            kernel_size: int,
            dropout: float | None,
    ) -> nn.Module:
        return nn.Sequential(
            *(
                ResidualConvBlock(
                    in_channels=d_model,
                    out_channels=d_model,
                    activation=activation,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    squeeze_excitation=True,
                )
                for _ in range(n_blocks)
            )
        )


class ActorCriticAttnBase(nn.Module):
    def __init__(
        self,
        spatial_in_channels: int,
        global_in_channels: int,
        d_model: int,
        n_blocks: int,
        num_heads: int,
        dropout: float | None,
        activation: ActivationFactory,
    ) -> None:
        super().__init__()
        self.spatial_in = self._build_spatial_in(
            spatial_in_channels,
            d_model,
            activation=activation,
            kernel_size=kernel_size,
        )
        self.global_in = self._build_global_in(
            global_in_channels,
            d_model,
            activation=activation,
        )
        self.base = self._build_base(
            d_model=d_model,
            n_blocks=n_blocks,
            activation=activation,
            num_heads=num_heads,
            dropout=dropout,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        self.apply(orthogonal_initialization_)

    def forward(
            self,
            obs: TorchObs,
    ) -> torch.Tensor:
        # TODO: Reshape to attention format
        x = (
            self.spatial_in(obs.spatial_obs)
            + self.global_in(obs.global_obs)[..., None, None]
        )
        x = self.base(x)
        # TODO: Reshape x back to conv format?
        return x

    @classmethod
    def _build_spatial_in(
        cls,
        in_channels: int,
        d_model: int,
        activation: ActivationFactory,
        kernel_size: int,
    ) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=d_model,
                kernel_size=kernel_size,
                padding="same",
            ),
            activation(),
            nn.Conv2d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=kernel_size,
                padding="same",
            ),
        )

    @classmethod
    def _build_global_in(
            cls, in_channels: int, d_model: int, activation: ActivationFactory
    ) -> nn.Module:
        return nn.Sequential(
            nn.Linear(
                in_features=in_channels,
                out_features=d_model,
            ),
            activation(),
            nn.Linear(
                in_features=d_model,
                out_features=d_model,
            ),
        )

    @classmethod
    def _build_base(
            cls,
            d_model: int,
            n_blocks: int,
            activation: ActivationFactory,
            num_heads: int,
            dropout: float | None,
    ) -> nn.Module:
        return nn.Sequential(
            *(
                AttnBlock(
                    d_model=d_model,
                    activation=activation,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            )
        )
