from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.amp import GradScaler  # type: ignore[attr-defined]

from rux_ai_s3.types import Action

_ModelT = TypeVar("_ModelT", bound=nn.Module)


@dataclass
class TrainState(Generic[_ModelT]):
    model: _ModelT
    optimizer: optim.Optimizer
    scaler: GradScaler
    step: int = 0


def bootstrap_value(
    value_estimate: npt.NDArray[np.float32],
    reward: npt.NDArray[np.float32],
    done: npt.NDArray[np.bool],
    gamma: float,
    gae_lambda: float,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    steps_plus_one, envs, players = reward.shape
    advantages: npt.NDArray[np.float32] = np.zeros(
        (steps_plus_one - 1, envs, players), dtype=np.float32
    )
    last_gae_lambda = np.zeros((envs, players), dtype=np.float32)
    for t, last_value in reversed(list(enumerate(value_estimate))[:-1]):
        step_reward = reward[t + 1]
        step_done = done[t + 1]
        next_value = value_estimate[t + 1]
        delta = step_reward + gamma * next_value * (1.0 - step_done) - last_value
        last_gae_lambda = (
            delta + gamma * gae_lambda * (1.0 - step_done) * last_gae_lambda
        )
        advantages[t] = last_gae_lambda

    returns: npt.NDArray[np.float32] = advantages + value_estimate[:-1]
    return advantages, returns


def compute_pg_loss(
    advantages: torch.Tensor,
    action_probability_ratio: torch.Tensor,
    clip_coefficient: float,
) -> torch.Tensor:
    pg_loss_1 = -advantages * action_probability_ratio
    pg_loss_2 = -advantages * torch.clamp(
        action_probability_ratio,
        1.0 - clip_coefficient,
        1.0 + clip_coefficient,
    )
    return torch.maximum(pg_loss_1, pg_loss_2).mean()


def compute_value_loss(
    new_value_estimate: torch.Tensor,
    returns: torch.Tensor,
) -> torch.Tensor:
    return F.huber_loss(new_value_estimate, returns)


def compute_entropy_loss(
    main_log_probs: torch.Tensor,
    sap_log_probs: torch.Tensor,
) -> torch.Tensor:
    assert main_log_probs.shape[-1] - 1 == Action.SAP.value
    log_policy = torch.cat(
        [main_log_probs[..., :-1], main_log_probs[..., -1:] + sap_log_probs],
        dim=-1,
    )
    policy = log_policy.exp()
    log_policy_masked_zeroed = torch.where(
        log_policy.isneginf(),
        torch.zeros_like(log_policy),
        log_policy,
    )
    entropies = (policy * log_policy_masked_zeroed).sum(dim=-1)
    return entropies.mean()
