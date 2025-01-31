from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from rux_ai_s3.lowlevel import RewardSpace
from rux_ai_s3.models.actor_critic import ActorCritic
from rux_ai_s3.models.types import TorchActionInfo, TorchObs
from rux_ai_s3.parallel_env import ParallelEnv


@torch.no_grad()
def run_match(
    env: ParallelEnv,
    models: tuple[ActorCritic, ActorCritic],
    min_games: int,
    device: torch.device,
    prog_bar: Any | None = None,
) -> tuple[int, int]:
    assert env.reward_space == RewardSpace.FINAL_WINNER
    env.hard_reset()
    p1_model, p2_model = models

    scores: npt.NDArray[np.float32] = np.array([0.0, 0.0])
    while sum(scores) < min_games:
        stacked_obs = TorchObs.from_numpy(env.get_frame_stacked_obs(), device)
        action_info = TorchActionInfo.from_numpy(env.last_out.action_info, device)
        unit_indices = env.last_out.action_info.unit_indices
        p1_actions = p1_model(
            obs=stacked_obs.index_player(0),
            action_info=action_info.index_player(0),
        ).to_env_actions(unit_indices[:, 0])
        p2_actions = p2_model(
            obs=stacked_obs.index_player(1),
            action_info=action_info.index_player(1),
        ).to_env_actions(unit_indices[:, 1])

        env.step(np.stack([p1_actions, p2_actions], axis=1))
        scores += np.maximum(env.last_out.reward, 0.0).sum(axis=0)
        games_completed: int = np.sum(env.last_out.done).item()
        if games_completed and prog_bar is not None:
            prog_bar.update(games_completed)

    p1_score, p2_score = scores
    return int(p1_score), int(p2_score)
