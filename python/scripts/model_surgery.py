from pathlib import Path
from typing import Any, Final

import torch
from rux_ai_s3.models.actor_critic import ActorCritic
from rux_ai_s3.models.build import ActorCriticConfig, build_actor_critic
from rux_ai_s3.models.utils import remove_compile_prefix
from rux_ai_s3.parallel_env import ParallelEnv
from rux_ai_s3.rl_training.train_config import TrainConfig
from rux_ai_s3.rl_training.utils import get_config_path_from_checkpoint
from rux_ai_s3.utils import load_from_yaml
from torch import nn

MODEL_CHECKPOINT: Final[Path] = Path(
    "/home/isaiah/GitHub/rux-ai-s3/hall_of_fame/jolly-yogurt-134_031067/rux_ai_s3/rl_agent/checkpoint_031067_weights.pt"
)


def main() -> None:
    """
    This script is used to manually add an input feature channel to an existing
    RL model without changing its weights or behavior. Adapt it as needed on a
    case by case basis
    """
    config = load_from_yaml(
        TrainConfig, get_config_path_from_checkpoint(MODEL_CHECKPOINT)
    )
    env = ParallelEnv.from_config(config.env_config)
    original_state_dict = torch.load(
        MODEL_CHECKPOINT,
        map_location=torch.device("cpu"),
        weights_only=True,
    )["model"]
    model = load_model_from_state_dict(config, env, original_state_dict)
    print({k: v.shape for k, v in model.base.spatial_in.named_parameters()})
    layer = model.base.spatial_in[0]
    print(layer.weight.shape)
    new_channel = 6
    assert env.last_out.obs.temporal_spatial_obs.shape[-3] == new_channel
    assert (
        layer.weight.shape[1]
        == new_channel * config.env_config.frame_stack_len
        + env.last_out.obs.nontemporal_spatial_obs.shape[-3]
    )
    layer.weight = modify_weights(layer.weight, new_channel, env)
    print(layer.weight.shape)
    print(layer.weight[:2, :, 0, 0])
    torch.save({"model": model.state_dict()}, "weights_surgery.pt")


def load_model_from_state_dict(
    config: TrainConfig, env: ParallelEnv, state_dict: Any
) -> ActorCritic:
    model = build_model(config.rl_model_config, env)
    state_dict = {
        remove_compile_prefix(key): value for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict)
    return model.eval()


def build_model(
    config: ActorCriticConfig,
    env: ParallelEnv,
) -> ActorCritic:
    example_obs = env.get_frame_stacked_obs()
    spatial_in_channels = example_obs.spatial_obs.shape[2]
    global_in_channels = example_obs.global_obs.shape[2]
    n_main_actions = env.last_out.action_info.main_mask.shape[-1]
    return build_actor_critic(
        spatial_in_channels=spatial_in_channels,
        global_in_channels=global_in_channels,
        n_main_actions=n_main_actions,
        reward_space=env.reward_space,
        config=config,
        model_type=ActorCritic,
    )


def modify_weights(
    weights: torch.Tensor, new_channel: int, env: ParallelEnv
) -> torch.Tensor:
    new_w_shape = (weights.shape[0], 1, *weights.shape[-2:])
    splits = [new_channel] * env.frame_stack_len
    splits.append(env.last_out.obs.nontemporal_spatial_obs.shape[-3])
    new_weights = []
    weights_split = torch.split(weights, splits, dim=1)
    for w in weights_split[:-1]:
        new_weights.append(
            torch.cat([w, torch.zeros(new_w_shape, dtype=w.dtype)], dim=1)
        )

    new_weights.append(weights_split[-1])
    return nn.Parameter(torch.cat(new_weights, dim=1))


if __name__ == "__main__":
    main()
