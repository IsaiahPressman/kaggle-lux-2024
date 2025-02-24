import torch
from torch import nn

from rux_ai_s3.models.attn_blocks import AttnBlock, SpatialAttnIn
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
        self.apply(lambda m: orthogonal_initialization_(m, strict=False))

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
        d_mlp: int,
        num_heads: int,
        n_blocks: int,
        activation: ActivationFactory,
        dropout: float | None,
    ) -> None:
        super().__init__()
        self.spatial_in = SpatialAttnIn(
            in_channels=spatial_in_channels,
            d_model=d_model,
            d_mlp=d_mlp,
            num_heads=num_heads,
            activation=activation,
            dropout=dropout,
        )
        self.global_in = self._build_global_in(
            global_in_channels,
            d_model,
            activation=activation,
        )
        self.base = self._build_base(
            d_model=d_model,
            d_mlp=d_mlp,
            num_heads=num_heads,
            n_blocks=n_blocks,
            activation=activation,
            dropout=dropout,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        in_, _, out_ = self.global_in.children()
        orthogonal_initialization_(in_)
        orthogonal_initialization_(out_)

    def forward(
        self,
        obs: TorchObs,
    ) -> torch.Tensor:
        _, _, width, height = obs.spatial_obs.shape
        x = (
            self.spatial_in(obs.spatial_obs)
            + self.global_in(obs.global_obs)[..., None, None]
        )
        x = self.base(x)
        # x shape is currently (batch_size, seq_len, d_model)
        # Return it to shape (batch_size, d_model, width, height)
        return x.permute(0, 2, 1).unflatten(-1, (width, height))

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
        d_mlp: int,
        num_heads: int,
        n_blocks: int,
        activation: ActivationFactory,
        dropout: float | None,
    ) -> nn.Module:
        return nn.Sequential(
            *(
                AttnBlock(
                    d_model=d_model,
                    d_mlp=d_mlp,
                    num_heads=num_heads,
                    activation=activation,
                    dropout=dropout,
                )
                for _ in range(n_blocks)
            )
        )
