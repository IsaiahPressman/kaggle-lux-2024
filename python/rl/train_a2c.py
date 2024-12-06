# ruff: noqa
# TODO: Re-enable linting
import itertools
from collections.abc import Generator
from pathlib import Path
from typing import Final

import torch
import yaml
from pydantic import BaseModel, ConfigDict, field_validator
from rux_2024.models.actor_critic import ActorCritic, ActorCriticOut
from rux_2024.parallel_env import ParallelEnv, RewardSpace
from torch import optim
from torch.cuda.amp import GradScaler

FILE: Final[Path] = Path(__file__)
CONFIG_FILE: Final[Path] = FILE.parent / "config" / "a2c.yaml"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class A2CConfig(BaseModel):
    # Training config
    max_batches: int | None
    optimizer_kwargs: dict[str, float]
    steps_per_batch: int
    n_envs: int
    frame_stack_len: int
    reward_space: RewardSpace

    # Model config
    d_model: int
    n_layers: int

    # Miscellaneous config
    device: torch.device

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    def iter_batches(self) -> Generator[int, None, None]:
        if self.max_batches is None:
            return (i for i in itertools.count(1))

        return (i for i in range(1, self.max_batches + 1))

    @field_validator("reward_space", mode="before")
    @classmethod
    def _validate_reward_space(cls, reward_space: RewardSpace | str) -> RewardSpace:
        if isinstance(reward_space, RewardSpace):
            return reward_space

        return RewardSpace.from_str(reward_space)

    @field_validator("device", mode="before")
    @classmethod
    def _validate_device(cls, device: torch.device | str) -> torch.device:
        if isinstance(device, torch.device):
            return device

        return torch.device(device)

    @classmethod
    def from_file(cls, path: Path) -> "A2CConfig":
        with open(path) as f:
            data = yaml.safe_load(f)

        return A2CConfig(**data)


def main() -> None:
    cfg = A2CConfig.from_file(CONFIG_FILE)
    env = ParallelEnv(
        n_envs=cfg.n_envs,
        reward_space=cfg.reward_space,
        frame_stack_len=cfg.frame_stack_len,
    )
    model: ActorCritic = build_model(env, cfg).to(cfg.device).train()
    model = torch.compile(model)  # type: ignore
    optimizer = optim.Adam(model.parameters(), **cfg.optimizer_kwargs)  # type: ignore
    scaler = GradScaler()
    for b in cfg.iter_batches():
        env_out = env.last_out
        stacked_obs = env.get_frame_stacked_obs()
        model_out: ActorCriticOut = model(
            spatial_obs=stacked_obs.spatial_obs,
            global_obs=stacked_obs.global_obs,
            unit_indices=env_out.action_info.unit_indices,
            unit_energies=env_out.action_info.unit_energies,
        )
        # TODO: Left off here

    raise NotImplementedError


def build_model(
    env: ParallelEnv,
    cfg: A2CConfig,
) -> ActorCritic:
    example_obs = env.get_frame_stacked_obs()
    spatial_in_channels = example_obs.spatial_obs.shape[3]
    global_in_channels = example_obs.global_obs.shape[3]
    return ActorCritic(
        spatial_in_channels=spatial_in_channels,
        global_in_channels=global_in_channels,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        reward_space=env.reward_space,
    )


if __name__ == "__main__":
    main()
