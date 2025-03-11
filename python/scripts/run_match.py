import argparse
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import torch
import tqdm
from pydantic import BaseModel
from rux_ai_s3.lowlevel import RewardSpace
from rux_ai_s3.models.actor_critic import ActorCritic
from rux_ai_s3.models.build import build_actor_critic
from rux_ai_s3.models.utils import remove_compile_prefix
from rux_ai_s3.parallel_env import ParallelEnv
from rux_ai_s3.rl_training.evaluation import run_match
from rux_ai_s3.rl_training.train_config import TrainConfig
from rux_ai_s3.rl_training.utils import get_config_path_from_checkpoint
from rux_ai_s3.utils import load_from_yaml

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class UserArgs(BaseModel):
    p1: Path
    p2: Path
    min_games: int
    batch_size: int

    def get_p1_config(self) -> TrainConfig:
        return load_from_yaml(TrainConfig, get_config_path_from_checkpoint(self.p1))

    def get_p2_config(self) -> TrainConfig:
        return load_from_yaml(TrainConfig, get_config_path_from_checkpoint(self.p2))

    def get_batch_size(self) -> int:
        return min(self.batch_size, self.min_games // 2)

    @classmethod
    def from_argparse(cls) -> "UserArgs":
        parser = argparse.ArgumentParser(
            description="Plays a multi-game match between two trained models"
        )
        parser.add_argument("p1", type=Path)
        parser.add_argument("p2", type=Path)
        parser.add_argument("-g", "--min_games", type=int, default=512)
        parser.add_argument("-b", "--batch_size", type=int, default=256)
        args = parser.parse_args()
        return UserArgs(**vars(args))


def main() -> None:
    args = UserArgs.from_argparse()
    p1_config = args.get_p1_config()
    p2_config = args.get_p2_config()
    assert p1_config.frame_stack_len == p2_config.frame_stack_len
    assert p1_config.sap_masking == p2_config.sap_masking
    env = ParallelEnv(
        n_envs=args.get_batch_size(),
        frame_stack_len=p1_config.env_config.frame_stack_len,
        sap_masking=p1_config.sap_masking,
        reward_space=RewardSpace.FINAL_WINNER,
        jax_device=jax.devices("cpu")[0],
    )
    p1 = build_model(env, p1_config, args.p1, DEVICE)
    p2 = build_model(env, p2_config, args.p2, DEVICE)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        prog_bar = tqdm.tqdm(total=args.min_games)
        (p1_score_1, p2_score_2) = run_match(
            env,
            (p1, p2),
            args.min_games // 2,
            DEVICE,
            prog_bar,
        )
        (p2_score_1, p1_score_2) = run_match(
            env,
            (p2, p1),
            args.min_games // 2,
            DEVICE,
            prog_bar,
        )

    raw_results = np.array([[p1_score_1, p2_score_1], [p1_score_2, p2_score_2]])
    raw_results_normalized = np.round(100.0 * raw_results / raw_results.sum(), 2)
    index = ["(0, 0)", "(23, 23)"]
    columns = ["p1", "p2"]
    results = pd.DataFrame(
        raw_results,
        index=index,
        columns=columns,
    )
    results_normalized = pd.DataFrame(
        raw_results_normalized,
        index=index,
        columns=columns,
    )
    print(f"\nRaw results:\n{results}\n\nNormalized results:\n{results_normalized}")

    (p1_score, p2_score) = results.sum(axis="rows").tolist()
    (p1_pct, p2_pct) = results_normalized.sum(axis="rows").tolist()
    print(
        f"\nFinal scores after {p1_score + p2_score} games: "
        f"({p1_score}, {p2_score}) / ({p1_pct:.2f}%, {p2_pct:.2f}%)"
    )


def build_model(
    env: ParallelEnv, config: TrainConfig, weights_path: Path, device: torch.device
) -> ActorCritic:
    example_obs = env.get_frame_stacked_obs()
    spatial_in_channels = example_obs.spatial_obs.shape[2]
    global_in_channels = example_obs.global_obs.shape[2]
    n_main_actions = env.last_out.action_info.main_mask.shape[-1]
    model = build_actor_critic(
        spatial_in_channels=spatial_in_channels,
        global_in_channels=global_in_channels,
        n_main_actions=n_main_actions,
        # NB: This may not be the same as the env reward space
        reward_space=config.env_config.reward_space,
        config=config.rl_model_config,
        model_type=ActorCritic,
    )

    state_dict = torch.load(
        weights_path,
        map_location=torch.device("cpu"),
        weights_only=True,
    )["model"]
    state_dict = {
        remove_compile_prefix(key): value for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict)
    return model.to(device).eval()


if __name__ == "__main__":
    main()
