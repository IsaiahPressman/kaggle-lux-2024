from torch import nn
from typing_extensions import assert_never

from rux_2024._lowlevel import RewardSpace

from .critic_heads import BoundedCriticHead, ZeroSumCriticHead
from .types import ActivationFactory


def build_critic_head(
    reward_space: RewardSpace,
    d_model: int,
    activation: ActivationFactory,
) -> nn.Module:
    if reward_space == RewardSpace.FINAL_WINNER:
        return ZeroSumCriticHead(
            d_model,
            activation=activation,
        )

    if reward_space == RewardSpace.MATCH_WINNER:
        return BoundedCriticHead(
            reward_min=-3,
            reward_max=3,
            d_model=d_model,
            activation=activation,
        )

    if reward_space == RewardSpace.POINTS_SCORED:
        raise NotImplementedError("Positive unbounded critic")

    assert_never(reward_space)
