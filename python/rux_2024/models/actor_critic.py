from torch import nn

from .conv_blocks import ResidualBlock
from .types import ActivationFactory


class ActorCritic(nn.Module):
    def __init__(
        self,
        spatial_in_channels: int,
        global_in_channels: int,
        d_model: int,
        n_layers: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        activation: ActivationFactory = nn.GELU
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
            n_blocks=n_layers,
            activation=activation,
            kernel_size=kernel_size,
        )
        self.actor_head = self._build_actor_head()
        self.critic_head = self._build_critic_head()

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
    ) -> nn.Module:
        return nn.Sequential(
            *(
                ResidualBlock(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    activation=activation(),
                    squeeze_excitation=True,
                )
                for _ in range(n_blocks)
            )
        )

    @classmethod
    def _build_actor_head(cls) -> nn.Module:
        # TODO
        raise NotImplementedError
