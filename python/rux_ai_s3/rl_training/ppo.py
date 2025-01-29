import logging
from enum import Enum, auto

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import wandb
import yaml
from typing_extensions import assert_never

from rux_ai_s3.types import Action


class WandbLogMode(Enum):
    DISABLED = auto()
    SCALAR_ONLY = auto()
    ALL = auto()


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
    assert advantages.shape == action_probability_ratio.shape
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
    assert new_value_estimate.shape == returns.shape
    return F.huber_loss(new_value_estimate, returns)


def merge_log_probs(
    main_log_probs: torch.Tensor,
    sap_log_probs: torch.Tensor,
    units_mask: torch.Tensor,
) -> torch.Tensor:
    r"""
    Combines main_log_probs of shape (\*, units, n_main_actions + n_sap_actions) with
    sap_log_probs of shape (\*, units, 24 * 24) to get combined (joint) log probs
    of shape (\*, units, n_main_actions + 24 * 24).

    Results where units_mask == True will behave as expected, but results where
    units_mask == False will be ill-formed log probs and should be ignored.
    """
    main_sap_log_probs = main_log_probs[..., Action.SAP :]
    main_sap_log_probs = torch.where(
        units_mask.unsqueeze(-1),
        main_sap_log_probs,
        -torch.inf,
    )
    if torch.any(main_sap_log_probs.isneginf().logical_not().sum(dim=-1) > 1):
        raise ValueError("Got multiple unmasked sap actions")

    can_sap = main_sap_log_probs.isneginf().logical_not().any(dim=-1)
    main_sap_log_probs_masked_zeroed = torch.where(
        main_sap_log_probs.isneginf(),
        torch.zeros_like(main_sap_log_probs),
        main_sap_log_probs,
    ).sum(dim=-1, keepdim=True)
    merged_sap_log_probs = torch.where(
        can_sap.unsqueeze(-1),
        main_sap_log_probs_masked_zeroed + sap_log_probs,
        torch.zeros_like(sap_log_probs) - torch.inf,
    )
    return torch.cat(
        [
            main_log_probs[..., : Action.SAP],
            merged_sap_log_probs,
        ],
        dim=-1,
    )


def compute_entropy_loss(
    main_log_probs: torch.Tensor,
    sap_log_probs: torch.Tensor,
    units_mask: torch.Tensor,
) -> torch.Tensor:
    log_policy = merge_log_probs(main_log_probs, sap_log_probs, units_mask)
    policy = log_policy.exp()
    log_policy_masked_zeroed = torch.where(
        log_policy.isneginf(),
        torch.zeros_like(log_policy),
        log_policy,
    )
    entropies = (policy * log_policy_masked_zeroed).sum(dim=-1) * units_mask.float()
    return entropies.sum(dim=-1).mean()


def compute_teacher_kl_loss(
    learner_main_log_probs: torch.Tensor,
    learner_sap_log_probs: torch.Tensor,
    teacher_main_log_probs: torch.Tensor,
    teacher_sap_log_probs: torch.Tensor,
    units_mask: torch.Tensor,
) -> torch.Tensor:
    learner_log_probs = merge_log_probs(
        learner_main_log_probs, learner_sap_log_probs, units_mask
    )
    teacher_log_probs = merge_log_probs(
        teacher_main_log_probs, teacher_sap_log_probs, units_mask
    )
    kl_div = F.kl_div(
        learner_log_probs,
        teacher_log_probs,
        reduction="none",
        log_target=True,
    )
    kl_div = (
        torch.where(
            learner_log_probs.isneginf(),
            torch.zeros_like(kl_div),
            kl_div,
        ).sum(dim=-1)
        * units_mask.float()
    )
    return kl_div.sum(dim=-1).mean()


def compute_teacher_value_loss(
    learner_value_estimate: torch.Tensor,
    teacher_value_estimate: torch.Tensor,
) -> torch.Tensor:
    assert learner_value_estimate.shape == teacher_value_estimate.shape
    return F.huber_loss(learner_value_estimate, teacher_value_estimate)


def get_wandb_log_mode(
    wandb_enabled: bool,
    histograms_enabled: bool,
) -> WandbLogMode:
    if not wandb_enabled:
        return WandbLogMode.DISABLED

    if not histograms_enabled:
        return WandbLogMode.SCALAR_ONLY

    return WandbLogMode.ALL


def log_results(
    step: int,
    scalar_stats: dict[str, float],
    array_stats: dict[str, npt.NDArray[np.float32]],
    wandb_log_mode: WandbLogMode,
    logger: logging.Logger,
) -> None:
    logger.info("Completed step %d\n%s\n", step, yaml.dump(scalar_stats))
    if wandb_log_mode == WandbLogMode.DISABLED:
        return

    if wandb_log_mode == WandbLogMode.SCALAR_ONLY:
        wandb.log(scalar_stats, step=step)
        return

    if wandb_log_mode == WandbLogMode.ALL:
        histograms = {k: wandb.Histogram(v) for k, v in array_stats.items()}  # type: ignore[arg-type]
        combined_stats = dict(**scalar_stats, **histograms)
        wandb.log(combined_stats, step=step)
        return

    assert_never(wandb_log_mode)
