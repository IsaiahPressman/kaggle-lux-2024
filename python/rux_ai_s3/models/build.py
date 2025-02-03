import math
from typing import Annotated, Literal, TypeVar

from pydantic import BaseModel, Field, field_validator
from torch import nn
from typing_extensions import assert_never

from rux_ai_s3.lowlevel import RewardSpace

from .actor_critic import ActorCritic, ActorCriticBase, FactorizedActorCritic
from .actor_heads import BasicActorHead
from .critic_heads import (
    BaseCriticHead,
    BaseFactorizedCriticHead,
    BoundedCriticHead,
    BoundedFactorizedCriticHead,
    PositiveUnboundedCriticHead,
    ZeroSumCriticHead,
)
from .types import ActivationFactory

CriticMode = Literal["standard", "unit_factorized"]
_ModelT = TypeVar("_ModelT", bound=nn.Module)


class ActorCriticConfig(BaseModel):
    d_model: int
    n_blocks: int
    kernel_size: int = 3
    dropout: Annotated[float | None, Field(gt=0.0, le=0.2)] = None
    critic_mode: CriticMode = "standard"

    @field_validator("d_model")
    @classmethod
    def _validate_d_model(cls, d_model: int) -> int:
        if not math.log2(d_model).is_integer():
            raise ValueError("d_model should be a power of 2")

        return d_model


def build_critic_head(
    reward_space: RewardSpace,
    d_model: int,
    activation: ActivationFactory,
) -> BaseCriticHead:
    if reward_space == RewardSpace.FINAL_WINNER:
        return ZeroSumCriticHead(
            d_model=d_model,
            activation=activation,
        )

    if reward_space == RewardSpace.MATCH_WINNER:
        return BoundedCriticHead(
            reward_min=-5,
            reward_max=5,
            d_model=d_model,
            activation=activation,
        )

    if reward_space == RewardSpace.POINTS_SCORED:
        return PositiveUnboundedCriticHead(
            reward_min=0,
            d_model=d_model,
            activation=activation,
        )

    assert_never(reward_space)


def build_factorized_critic_head(
    reward_space: RewardSpace,
    d_model: int,
    activation: ActivationFactory,
) -> BaseFactorizedCriticHead:
    if reward_space == RewardSpace.FINAL_WINNER:
        return BoundedFactorizedCriticHead(
            reward_min=-1,
            reward_max=1,
            d_model=d_model,
            activation=activation,
        )

    if reward_space == RewardSpace.MATCH_WINNER:
        return BoundedFactorizedCriticHead(
            reward_min=-5,
            reward_max=5,
            d_model=d_model,
            activation=activation,
        )

    if reward_space == RewardSpace.POINTS_SCORED:
        raise NotImplementedError("PositiveUnboundedFactorizedCriticHead")

    assert_never(reward_space)


def build_actor_critic(
    spatial_in_channels: int,
    global_in_channels: int,
    n_main_actions: int,
    reward_space: RewardSpace,
    config: ActorCriticConfig,
    model_type: type[_ModelT] = nn.Module,  # type: ignore[assignment]
) -> _ModelT:
    activation: ActivationFactory = nn.GELU
    base = ActorCriticBase(
        spatial_in_channels=spatial_in_channels,
        global_in_channels=global_in_channels,
        d_model=config.d_model,
        n_blocks=config.n_blocks,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
        activation=activation,
    )
    actor_head = BasicActorHead(
        d_model=config.d_model,
        activation=activation,
        n_main_actions=n_main_actions,
    )
    model: nn.Module
    match config.critic_mode:
        case "standard":
            standard_critic = build_critic_head(
                reward_space=reward_space,
                d_model=config.d_model,
                activation=activation,
            )
            model = ActorCritic(
                base=base,
                actor_head=actor_head,
                critic_head=standard_critic,
            )
        case "unit_factorized":
            factorized_critic = build_factorized_critic_head(
                reward_space=reward_space,
                d_model=config.d_model,
                activation=activation,
            )
            model = FactorizedActorCritic(
                base=base,
                actor_head=actor_head,
                critic_head=factorized_critic,
            )
        case _:
            assert_never(config.critic_mode)

    assert isinstance(model, model_type)
    return model
