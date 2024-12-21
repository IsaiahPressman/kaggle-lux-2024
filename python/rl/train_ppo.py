import argparse
import collections
import datetime
import itertools
import math
import os
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import wandb
import yaml
from pydantic import BaseModel, ConfigDict, field_validator
from rux_ai_s3.constants import PROJECT_NAME, TRAIN_OUTPUTS_DIR
from rux_ai_s3.models.actor_critic import ActorCritic, ActorCriticOut
from rux_ai_s3.models.types import TorchActionInfo, TorchObs
from rux_ai_s3.parallel_env import EnvConfig, ParallelEnv
from rux_ai_s3.types import Action, Stats
from torch import optim
from torch.amp import GradScaler  # type: ignore[attr-defined]

FILE: Final[Path] = Path(__file__)
NAME: Final[str] = "ppo"
CONFIG_FILE: Final[Path] = FILE.parent / "config" / f"{NAME}.yaml"
CHECKPOINT_FREQ: Final[datetime.timedelta] = datetime.timedelta(minutes=10)
CPU: Final[torch.device] = torch.device("cpu")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class UserArgs(BaseModel):
    debug: bool

    @property
    def release(self) -> bool:
        return not self.debug

    @classmethod
    def from_argparse(cls) -> "UserArgs":
        parser = argparse.ArgumentParser()
        parser.add_argument("--debug", action="store_true")
        args = parser.parse_args()
        return UserArgs(**vars(args))


class LossCoefficients(BaseModel):
    policy: float
    value: float
    entropy: float

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )


class PPOConfig(BaseModel):
    # Training config
    max_updates: int | None
    optimizer_kwargs: dict[str, float]
    steps_per_update: int
    epochs_per_update: int
    train_batch_size: int
    use_mixed_precision: bool

    gamma: float
    gae_lambda: float
    clip_coefficient: float
    loss_coefficients: LossCoefficients

    # Environment config
    env_config: EnvConfig

    # Model config
    d_model: int
    n_blocks: int

    # Miscellaneous config
    device: torch.device

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    @property
    def full_batch_size(self) -> int:
        return self.env_config.n_envs * self.steps_per_update * 2

    def iter_updates(self) -> Generator[int, None, None]:
        if self.max_updates is None:
            return (i for i in itertools.count(1))

        return (i for i in range(1, self.max_updates + 1))

    @field_validator("device", mode="before")
    @classmethod
    def _validate_device(cls, device: torch.device | str) -> torch.device:
        if isinstance(device, torch.device):
            return device

        return torch.device(device)

    @classmethod
    def from_file(cls, path: Path) -> "PPOConfig":
        with open(path) as f:
            data = yaml.safe_load(f)

        return PPOConfig(**data)


@dataclass
class TrainState:
    model: ActorCritic
    optimizer: optim.Optimizer
    scaler: GradScaler


@dataclass(frozen=True)
class ExperienceBatch:
    obs: TorchObs
    action_info: TorchActionInfo
    model_out: ActorCriticOut
    reward: npt.NDArray[np.float32]
    done: npt.NDArray[np.bool_]

    def validate(self) -> None:
        expected = self.reward.shape
        for t in itertools.chain(self.obs, self.action_info, self.model_out):
            assert t.shape[: self.reward.ndim] == expected

        assert self.done.shape == expected

    def trim(self) -> "ExperienceBatch":
        obs = TorchObs(*(t[:-1] for t in self.obs))
        action_info = TorchActionInfo(*(t[:-1] for t in self.action_info))
        model_out = ActorCriticOut(*(t[:-1] for t in self.model_out))
        return ExperienceBatch(
            obs=obs,
            action_info=action_info,
            model_out=model_out,
            reward=self.reward[1:],
            done=self.done[1:],
        )

    def flatten(self, end_dim: int) -> "ExperienceBatch":
        return ExperienceBatch(
            obs=self.obs.flatten(0, end_dim),
            action_info=self.action_info.flatten(0, end_dim),
            model_out=self.model_out.flatten(0, end_dim),
            reward=self.reward.reshape((-1, *self.reward.shape[(end_dim + 1) :])),
            done=self.done.reshape((-1, *self.done.shape[(end_dim + 1) :])),
        )

    def index(self, ix: npt.NDArray[np.int_]) -> "ExperienceBatch":
        obs = TorchObs(*(t[ix] for t in self.obs))
        action_info = TorchActionInfo(*(t[ix] for t in self.action_info))
        model_out = ActorCriticOut(*(t[ix] for t in self.model_out))
        return ExperienceBatch(
            obs=obs,
            action_info=action_info,
            model_out=model_out,
            reward=self.reward[ix],
            done=self.done[ix],
        )

    def to_device(self, device: torch.device) -> "ExperienceBatch":
        return ExperienceBatch(
            obs=self.obs.to_device(device),
            action_info=self.action_info.to_device(device),
            model_out=self.model_out.to_device(device),
            reward=self.reward,
            done=self.done,
        )

    @classmethod
    def from_lists(
        cls,
        obs: list[TorchObs],
        action_info: list[TorchActionInfo],
        model_out: list[ActorCriticOut],
        reward: list[npt.NDArray[np.float32]],
        done: list[npt.NDArray[np.bool_]],
    ) -> "ExperienceBatch":
        return ExperienceBatch(
            obs=TorchObs(*(torch.stack(o_batch) for o_batch in zip(*obs))),
            action_info=TorchActionInfo(
                *(torch.stack(ai_batch) for ai_batch in zip(*action_info))
            ),
            model_out=ActorCriticOut(
                *(torch.stack(m_out) for m_out in zip(*model_out))
            ),
            reward=np.stack(reward),
            done=np.stack(done),
        )


def main() -> None:
    args = UserArgs.from_argparse()
    cfg = PPOConfig.from_file(CONFIG_FILE)
    init_train_dir(cfg)
    env = ParallelEnv(
        n_envs=cfg.env_config.n_envs,
        reward_space=cfg.env_config.reward_space,
        frame_stack_len=cfg.env_config.frame_stack_len,
    )
    model: ActorCritic = build_model(env, cfg).to(cfg.device).train()
    if args.release:
        model = torch.compile(model)  # type: ignore[assignment]

    optimizer = optim.Adam(model.parameters(), **cfg.optimizer_kwargs)  # type: ignore[arg-type]
    scaler = GradScaler("cuda")
    train_state = TrainState(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
    )
    if args.release:
        wandb.init(
            project=PROJECT_NAME,
            group=NAME,
            config=cfg.model_dump(),
        )

    last_checkpoint = datetime.datetime.now()
    try:
        for step in cfg.iter_updates():
            train_step(
                step=step,
                env=env,
                train_state=train_state,
                args=args,
                cfg=cfg,
            )
            if (now := datetime.datetime.now()) - last_checkpoint >= CHECKPOINT_FREQ:
                last_checkpoint = now
                checkpoint(step, train_state)
    finally:
        checkpoint(step + 1, train_state)


def init_train_dir(cfg: PPOConfig) -> None:
    start_time = datetime.datetime.now()
    train_dir = (
        TRAIN_OUTPUTS_DIR
        / NAME
        / start_time.strftime("%Y_%m_%d")
        / start_time.strftime("%H_%M")
    )
    train_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(train_dir)
    with open("config.yaml", "w") as f:
        yaml.dump(cfg.model_dump(), f)


def build_model(
    env: ParallelEnv,
    cfg: PPOConfig,
) -> ActorCritic:
    example_obs = env.get_frame_stacked_obs()
    spatial_in_channels = example_obs.spatial_obs.shape[2]
    global_in_channels = example_obs.global_obs.shape[2]
    return ActorCritic(
        spatial_in_channels=spatial_in_channels,
        global_in_channels=global_in_channels,
        d_model=cfg.d_model,
        n_blocks=cfg.n_blocks,
        reward_space=env.reward_space,
    )


def train_step(
    step: int,
    env: ParallelEnv,
    train_state: TrainState,
    args: UserArgs,
    cfg: PPOConfig,
) -> None:
    step_start_time = time.perf_counter()
    experience, stats = collect_trajectories(env, train_state.model, cfg)
    scalar_stats = update_model(experience, train_state, cfg)
    array_stats = {}
    if stats:
        scalar_stats.update(stats.scalar_stats)
        array_stats.update(stats.array_stats)

    time_elapsed = time.perf_counter() - step_start_time
    batches_per_epoch = math.ceil(cfg.full_batch_size / cfg.train_batch_size)
    scalar_stats["updates_per_second"] = (
        batches_per_epoch * cfg.epochs_per_update / time_elapsed
    )
    scalar_stats["env_steps_per_second"] = (
        cfg.env_config.n_envs * cfg.steps_per_update / time_elapsed
    )
    log_results(
        step,
        scalar_stats,
        array_stats,
        wandb_log=args.release,
    )


@torch.no_grad()
def collect_trajectories(
    env: ParallelEnv,
    model: ActorCritic,
    cfg: PPOConfig,
) -> tuple[ExperienceBatch, Stats | None]:
    batch_obs = []
    batch_action_info = []
    batch_model_out = []
    batch_reward = []
    batch_done = []
    stats: Stats | None = None
    for step in range(cfg.steps_per_update + 1):
        last_out = env.last_out
        stacked_obs = TorchObs.from_numpy(env.get_frame_stacked_obs(), cfg.device)
        action_info = TorchActionInfo.from_numpy(last_out.action_info, cfg.device)
        model_out: ActorCriticOut = model(
            obs=stacked_obs.flatten(start_dim=0, end_dim=1),
            action_info=action_info.flatten(start_dim=0, end_dim=1),
            random_sample_actions=True,
        )

        batch_obs.append(stacked_obs.to_device(CPU))
        batch_action_info.append(action_info.to_device(CPU))
        batch_model_out.append(model_out.add_player_dim().to_device(CPU))
        batch_reward.append(last_out.reward)
        batch_done.append(last_out.done[:, None].repeat(2, axis=1))
        if step < cfg.steps_per_update:
            env.step(model_out.to_player_env_actions(last_out.action_info.unit_indices))
            stats = env.last_out.stats or stats

    experience = ExperienceBatch.from_lists(
        obs=batch_obs,
        action_info=batch_action_info,
        model_out=batch_model_out,
        reward=batch_reward,
        done=batch_done,
    )
    return experience, stats


def update_model(
    experience: ExperienceBatch,
    train_state: TrainState,
    cfg: PPOConfig,
) -> dict[str, float]:
    advantages_np, returns_np = bootstrap_value(experience, cfg)
    advantages = torch.from_numpy(advantages_np)
    returns = torch.from_numpy(returns_np)
    experience.validate()
    # Combine batch/player dims for experience batch, advantages, and returns
    experience = experience.trim().flatten(end_dim=2)
    advantages = torch.flatten(advantages, start_dim=0, end_dim=2)
    returns = torch.flatten(returns, start_dim=0, end_dim=2)
    aggregated_stats = collections.defaultdict(list)
    for _ in range(cfg.epochs_per_update):
        assert experience.done.shape[0] == cfg.full_batch_size
        batch_indices = np.random.permutation(cfg.full_batch_size)
        for minibatch_start in range(0, cfg.full_batch_size, cfg.train_batch_size):
            minibatch_end = minibatch_start + cfg.train_batch_size
            minibatch_indices = batch_indices[minibatch_start:minibatch_end]
            batch_stats = update_model_on_batch(
                train_state,
                experience.index(minibatch_indices).to_device(cfg.device),
                advantages[minibatch_indices].to(cfg.device),
                returns[minibatch_indices].to(cfg.device),
                cfg,
            )
            for k, v in batch_stats.items():
                aggregated_stats[k].append(v)

    return {key: np.mean(val).item() for key, val in aggregated_stats.items()}


def bootstrap_value(
    experience: ExperienceBatch,
    cfg: PPOConfig,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    steps_plus_one, envs, players = experience.reward.shape
    advantages: npt.NDArray[np.float32] = np.zeros(
        (steps_plus_one - 1, envs, players), dtype=np.float32
    )
    last_gae_lambda = np.zeros((envs, players), dtype=np.float32)
    agent_value_out: npt.NDArray[np.float32] = experience.model_out.value.cpu().numpy()
    for t, last_value in reversed(list(enumerate(agent_value_out))[:-1]):
        reward = experience.reward[t + 1]
        done = experience.done[t + 1]
        next_value = agent_value_out[t + 1]
        delta = reward + cfg.gamma * next_value * (1.0 - done) - last_value
        last_gae_lambda = (
            delta + cfg.gamma * cfg.gae_lambda * (1.0 - done) * last_gae_lambda
        )
        advantages[t] = last_gae_lambda

    returns: npt.NDArray[np.float32] = advantages + agent_value_out[:-1]
    return advantages, returns


def update_model_on_batch(
    train_state: TrainState,
    experience: ExperienceBatch,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    cfg: PPOConfig,
) -> dict[str, float]:
    with torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=cfg.use_mixed_precision
    ):
        new_out: ActorCriticOut = train_state.model(
            obs=experience.obs,
            action_info=experience.action_info,
            random_sample_actions=True,
        )._replace(
            main_actions=experience.model_out.main_actions,
            sap_actions=experience.model_out.sap_actions,
        )
        log_action_probability_ratio = (
            new_out.compute_joint_log_probs()
            - experience.model_out.compute_joint_log_probs()
        )
        action_probability_ratio = log_action_probability_ratio.exp()
        clip_fraction: float = (
            ((action_probability_ratio - 1.0).abs() > cfg.clip_coefficient)
            .float()
            .mean()
            .item()
        )
        policy_loss = (
            compute_pg_loss(
                advantages,
                action_probability_ratio,
                cfg.clip_coefficient,
            )
            * cfg.loss_coefficients.policy
        )
        value_loss = (
            compute_value_loss(
                new_out.value,
                returns,
            )
            * cfg.loss_coefficients.value
        )
        entropy_loss = (
            compute_entropy_loss(
                new_out.main_log_probs,
                new_out.sap_log_probs,
            )
            * cfg.loss_coefficients.entropy
        )
        total_loss = policy_loss + value_loss + entropy_loss

    step_optimizer(train_state, total_loss, cfg)
    return {
        "total_loss": total_loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy_loss": entropy_loss.item(),
        "clip_fraction": clip_fraction,
    }


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


def step_optimizer(
    train_state: TrainState,
    total_loss: torch.Tensor,
    cfg: PPOConfig,
) -> None:
    train_state.optimizer.zero_grad()
    if not cfg.use_mixed_precision:
        total_loss.backward()
        train_state.optimizer.step()
        return

    train_state.scaler.scale(total_loss).backward()
    train_state.scaler.step(train_state.optimizer)
    train_state.scaler.update()


def log_results(
    step: int,
    scalar_stats: dict[str, float],
    array_stats: dict[str, npt.NDArray[np.float32]],
    wandb_log: bool,
) -> None:
    print(f"Completed step {step}\n" f"{yaml.dump(scalar_stats)}\n")
    if not wandb_log:
        return

    histograms = {k: wandb.Histogram(v) for k, v in array_stats.items()}  # type: ignore[arg-type]
    combined_stats = dict(**scalar_stats, **histograms)
    wandb.log(combined_stats)


def checkpoint(step: int, train_state: TrainState) -> None:
    checkpoint_name = f"checkpoint_{str(step).zfill(6)}"
    full_path = f"{checkpoint_name}.pt"
    weights_path = f"{checkpoint_name}_weights.pt"
    torch.save(
        {
            "step": step,
            "model": train_state.model.state_dict(),
            "optimizer": train_state.optimizer.state_dict(),
            "scaler": train_state.scaler.state_dict(),
        },
        full_path,
    )
    torch.save(
        {
            "model": train_state.model.state_dict(),
        },
        weights_path,
    )
    print(f"Full checkpoint saved to {full_path} and weights saved to {weights_path}")


if __name__ == "__main__":
    main()
