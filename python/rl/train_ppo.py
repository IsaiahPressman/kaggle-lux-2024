import argparse
import collections
import itertools
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
from rux_ai_s3.constants import PROJECT_NAME, Action
from rux_ai_s3.models.actor_critic import ActorCritic, ActorCriticOut
from rux_ai_s3.models.types import TorchActionInfo, TorchObs
from rux_ai_s3.parallel_env import EnvConfig, ParallelEnv
from rux_ai_s3.types import Stats
from torch import optim
from torch.cuda.amp import GradScaler

FILE: Final[Path] = Path(__file__)
CONFIG_FILE: Final[Path] = FILE.parent / "config" / "ppo.yaml"
NAME: Final[str] = FILE.stem
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
    n_layers: int

    # Miscellaneous config
    device: torch.device

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
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

    def player_dim_flattened(self) -> "ExperienceBatch":
        return ExperienceBatch(
            obs=self.obs.player_dim_flattened(),
            action_info=self.action_info.player_dim_flattened(),
            model_out=self.model_out.player_dim_flattened(),
            reward=self.reward.reshape((-1, *self.reward.shape[2:])),
            done=self.done.reshape((-1, *self.done.shape[2:])),
        )

    def index(self, ix: npt.NDArray[np.int_]) -> "ExperienceBatch":
        obs = TorchObs(*(t[ix] for t in self.obs))
        action_info = TorchActionInfo(*(t[ix] for t in self.action_info))
        model_out = ActorCriticOut(*(t[ix] for t in self.action_info))
        return ExperienceBatch(
            obs=obs,
            action_info=action_info,
            model_out=model_out,
            reward=self.reward[ix],
            done=self.done[ix],
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
    env = ParallelEnv(
        n_envs=cfg.env_config.n_envs,
        reward_space=cfg.env_config.reward_space,
        frame_stack_len=cfg.env_config.frame_stack_len,
    )
    model: ActorCritic = build_model(env, cfg).to(cfg.device).train()
    if args.release:
        model = torch.compile(model)  # type: ignore[assignment]

    optimizer = optim.Adam(model.parameters(), **cfg.optimizer_kwargs)  # type: ignore[arg-type]
    scaler = GradScaler()
    train_state = TrainState(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
    )
    if args.release:
        wandb.init(
            project=f"{PROJECT_NAME}_{NAME}",
            config=cfg.model_dump(),
        )

    for step in cfg.iter_updates():
        step_start_time = time.perf_counter()
        experience, stats = collect_trajectories(env, train_state.model, cfg)
        scalar_stats = update_model(experience, train_state, cfg)
        array_stats = {}
        if stats:
            scalar_stats.update(stats.scalar_stats)
            array_stats.update(stats.array_stats)

        time_elapsed = time.perf_counter() - step_start_time
        scalar_stats["updates_per_second"] = 1.0 / time_elapsed
        scalar_stats["env_steps_per_second"] = cfg.steps_per_update / time_elapsed
        log_results(
            step,
            scalar_stats,
            array_stats,
            args.debug,
        )


def build_model(
    env: ParallelEnv,
    cfg: PPOConfig,
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
            obs=stacked_obs.player_dim_flattened(),
            action_info=action_info.player_dim_flattened(),
            random_sample_actions=True,
        )

        batch_obs.append(stacked_obs)
        batch_action_info.append(action_info)
        batch_model_out.append(model_out.add_player_dim())
        batch_reward.append(last_out.reward)
        batch_done.append(last_out.done[:, None].repeat(2, axis=1))
        if step < cfg.steps_per_update:
            env.step(model_out.to_player_env_actions(last_out.action_info.unit_indices))
            stats = stats or env.last_out.stats

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
    advantages = torch.from_numpy(advantages_np).to(cfg.device)
    returns = torch.from_numpy(returns_np).to(cfg.device)
    # Combine batch/player dims for experience batch, advantages, and returns
    experience = experience.player_dim_flattened()
    advantages = torch.flatten(advantages, start_dim=0, end_dim=1)
    returns = torch.flatten(returns, start_dim=0, end_dim=1)
    aggregated_stats = collections.defaultdict(list)
    for _ in range(cfg.epochs_per_update):
        batch_indices = np.random.permutation(cfg.full_batch_size)
        for minibatch_start in range(0, cfg.full_batch_size, cfg.train_batch_size):
            minibatch_end = minibatch_start + cfg.train_batch_size
            minibatch_indices = batch_indices[minibatch_start:minibatch_end]
            batch_stats = update_model_on_batch(
                train_state,
                experience.index(minibatch_indices),
                advantages[minibatch_indices],
                returns[minibatch_indices],
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
            obs=experience.obs.player_dim_flattened(),
            action_info=experience.action_info.player_dim_flattened(),
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
    debug: bool,
) -> None:
    if debug:
        print(f"Completed step {step}\n" f"{yaml.dump(scalar_stats)}\n")
        return

    histograms = {k: wandb.Histogram(v) for k, v in array_stats.items()}  # type: ignore[arg-type]
    combined_stats = dict(**scalar_stats, **histograms)
    wandb.log(combined_stats)


if __name__ == "__main__":
    main()
