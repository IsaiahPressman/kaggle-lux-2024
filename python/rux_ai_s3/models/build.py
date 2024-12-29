from typing import Callable

from torch import nn
from typing_extensions import assert_never

from rux_ai_s3.lowlevel import RewardSpace

from .critic_heads import (
    BoundedCriticHead,
    BoundedFactorizedCriticHead,
    PositiveUnboundedCriticHead,
)
from .types import ActivationFactory

CriticBuilder = Callable[[RewardSpace, int, ActivationFactory], nn.Module]


def build_critic_head(
    reward_space: RewardSpace,
    d_model: int,
    activation: ActivationFactory,
) -> nn.Module:
    if reward_space == RewardSpace.FINAL_WINNER:
        return BoundedCriticHead(
            reward_min=-1,
            reward_max=1,
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
) -> nn.Module:
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
