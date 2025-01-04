import argparse
import collections
import datetime
import itertools
import logging
import math
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final

import numpy as np
import numpy.typing as npt
import torch
import wandb
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from rux_ai_s3.lowlevel import assert_release_build
from rux_ai_s3.models.actor_critic import (
    FactorizedActorCritic,
    FactorizedActorCriticOut,
)
from rux_ai_s3.models.build import ActorCriticConfig, build_actor_critic
from rux_ai_s3.models.types import TorchActionInfo, TorchObs
from rux_ai_s3.parallel_env import EnvConfig, ParallelEnv
from rux_ai_s3.rl_training.constants import MAX_UNITS, PROJECT_NAME
from rux_ai_s3.rl_training.ppo import (
    bootstrap_value,
    compute_entropy_loss,
    compute_pg_loss,
    compute_value_loss,
    get_wandb_log_mode,
    log_results,
)
from rux_ai_s3.rl_training.train_config import TrainConfig
from rux_ai_s3.rl_training.utils import (
    TrainState,
    WandbInitConfig,
    count_trainable_params,
    get_config_path_from_checkpoint,
    init_logger,
    init_train_dir,
    load_checkpoint,
    save_checkpoint,
    validate_full_checkpoint_path,
)
from rux_ai_s3.types import Stats
from rux_ai_s3.utils import load_from_yaml
from torch import optim
from torch.amp import GradScaler  # type: ignore[attr-defined]
from torch.optim.lr_scheduler import LambdaLR

TrainStateT = TrainState[FactorizedActorCritic]
FILE: Final[Path] = Path(__file__)
NAME: Final[str] = "unit_factorized_ppo"
CONFIG_FILE: Final[Path] = FILE.parent / "config" / f"{NAME}.yaml"
CHECKPOINT_FREQ: Final[datetime.timedelta] = datetime.timedelta(minutes=10)
CPU: Final[torch.device] = torch.device("cpu")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logger = logging.getLogger(NAME.upper())


class UserArgs(BaseModel):
    debug: bool
    checkpoint: Path | None

    _validate_checkpoint = field_validator("checkpoint")(validate_full_checkpoint_path)

    @property
    def release(self) -> bool:
        return not self.debug

    @property
    def config_file(self) -> Path:
        if self.checkpoint is None:
            return CONFIG_FILE

        return get_config_path_from_checkpoint(self.checkpoint)

    @classmethod
    def from_argparse(cls) -> "UserArgs":
        parser = argparse.ArgumentParser()
        parser.add_argument("--debug", action="store_true")
        parser.add_argument(
            "--checkpoint",
            type=str,
            default=None,
            help="Checkpoint to resume training from",
        )
        args = parser.parse_args()
        return UserArgs(**vars(args))


class LossCoefficients(BaseModel):
    policy: float
    unit_value: float
    baseline_value: float
    entropy: float

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )


class UnitFactorizedPPOConfig(TrainConfig):
    # Training config
    max_updates: int | None
    optimizer_kwargs: dict[str, float]
    lr_schedule_steps: Annotated[int, Field(ge=10_000, lt=1_000_000)]
    lr_schedule_min_factor: Annotated[float, Field(gt=0.0, lt=1.0)]
    steps_per_update: int
    epochs_per_update: int
    train_batch_size: int
    use_mixed_precision: bool

    gamma: float
    gae_lambda: float
    clip_coefficient: float
    loss_coefficients: LossCoefficients

    # Config objects
    env_config: EnvConfig
    rl_model_config: ActorCriticConfig

    # Miscellaneous config
    device: torch.device
    log_histograms: bool

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    @property
    def full_batch_size(self) -> int:
        return self.env_config.n_envs * self.steps_per_update * 2

    @property
    def game_steps_per_update(self) -> int:
        return self.env_config.n_envs * self.steps_per_update

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

    @field_serializer("device")
    def serialize_device(self, device: torch.device) -> str:
        return str(device)


@dataclass(frozen=True)
class ExperienceBatch:
    obs: TorchObs
    action_info: TorchActionInfo
    model_out: FactorizedActorCriticOut
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
        model_out = FactorizedActorCriticOut(*(t[:-1] for t in self.model_out))
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

    def index(self, ix: slice | npt.NDArray[np.int_]) -> "ExperienceBatch":
        obs = TorchObs(*(t[ix] for t in self.obs))
        action_info = TorchActionInfo(*(t[ix] for t in self.action_info))
        model_out = FactorizedActorCriticOut(*(t[ix] for t in self.model_out))
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
        model_out: list[FactorizedActorCriticOut],
        reward: list[npt.NDArray[np.float32]],
        done: list[npt.NDArray[np.bool_]],
    ) -> "ExperienceBatch":
        return ExperienceBatch(
            obs=TorchObs(*(torch.stack(o_batch) for o_batch in zip(*obs))),
            action_info=TorchActionInfo(
                *(torch.stack(ai_batch) for ai_batch in zip(*action_info))
            ),
            model_out=FactorizedActorCriticOut(
                *(torch.stack(m_out) for m_out in zip(*model_out))
            ),
            reward=np.stack(reward),
            done=np.stack(done),
        )


def main() -> None:
    args = UserArgs.from_argparse()
    if args.release:
        assert_release_build()

    init_logger(logger=logger)
    logger.info("Loading config from %s", args.config_file)
    cfg = load_from_yaml(UnitFactorizedPPOConfig, args.config_file)
    init_train_dir(NAME, cfg.model_dump())
    env = ParallelEnv.from_config(cfg.env_config)
    model = build_model(env, cfg).to(cfg.device).train()
    logger.info(
        "Training model with %s parameters", f"{count_trainable_params(model):,d}"
    )
    if args.release:
        model = torch.compile(model)  # type: ignore[assignment]

    optimizer = optim.Adam(model.parameters(), **cfg.optimizer_kwargs)  # type: ignore[arg-type]
    lr_scheduler = build_lr_scheduler(cfg, optimizer)
    train_state = TrainState(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=GradScaler("cuda"),
    )
    if args.checkpoint:
        wandb_init_config = (
            WandbInitConfig(
                project=PROJECT_NAME,
                group=NAME,
            )
            if args.release
            else None
        )
        load_checkpoint(
            train_state,
            args.checkpoint,
            wandb_init_config=wandb_init_config,
            logger=logger,
        )

    if args.release and wandb.run is None:
        wandb.init(
            project=PROJECT_NAME,
            group=NAME,
            config=cfg.model_dump(),
        )

    last_checkpoint = datetime.datetime.now()
    try:
        for _ in cfg.iter_updates():
            train_step(
                env=env,
                train_state=train_state,
                cfg=cfg,
            )
            if (now := datetime.datetime.now()) - last_checkpoint >= CHECKPOINT_FREQ:
                last_checkpoint = now
                save_checkpoint(train_state, logger)
    finally:
        train_state.step += 1
        save_checkpoint(train_state, logger)


def build_model(
    env: ParallelEnv,
    cfg: UnitFactorizedPPOConfig,
) -> FactorizedActorCritic:
    example_obs = env.get_frame_stacked_obs()
    spatial_in_channels = example_obs.spatial_obs.shape[2]
    global_in_channels = example_obs.global_obs.shape[2]
    return build_actor_critic(
        spatial_in_channels=spatial_in_channels,
        global_in_channels=global_in_channels,
        reward_space=env.reward_space,
        config=cfg.rl_model_config,
        model_type=FactorizedActorCritic,
    )


def build_lr_scheduler(
    cfg: UnitFactorizedPPOConfig,
    optimizer: optim.Optimizer,
) -> LambdaLR:
    def lr_lambda(epoch: int) -> float:
        pct_remaining = max(1.0 - epoch / cfg.lr_schedule_steps, 0.0)
        return cfg.lr_schedule_min_factor + pct_remaining * (
            1.0 - cfg.lr_schedule_min_factor
        )

    return LambdaLR(
        optimizer,
        lr_lambda,
    )


def train_step(
    env: ParallelEnv,
    train_state: TrainStateT,
    cfg: UnitFactorizedPPOConfig,
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
    scalar_stats["game_steps"] = train_state.step * cfg.game_steps_per_update
    scalar_stats["updates_per_second"] = (
        batches_per_epoch * cfg.epochs_per_update / time_elapsed
    )
    scalar_stats["env_steps_per_second"] = (
        cfg.env_config.n_envs * cfg.steps_per_update / time_elapsed
    )
    log_results(
        train_state.step,
        scalar_stats,
        array_stats,
        wandb_log_mode=get_wandb_log_mode(
            wandb_enabled=wandb.run is not None,
            histograms_enabled=cfg.log_histograms,
        ),
        logger=logger,
    )
    train_state.lr_scheduler.step()
    train_state.step += 1


@torch.no_grad()
def collect_trajectories(
    env: ParallelEnv,
    model: FactorizedActorCritic,
    cfg: UnitFactorizedPPOConfig,
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
        model_out = model(
            obs=stacked_obs.flatten(start_dim=0, end_dim=1),
            action_info=action_info.flatten(start_dim=0, end_dim=1),
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
    train_state: TrainStateT,
    cfg: UnitFactorizedPPOConfig,
) -> dict[str, float]:
    unit_value_estimate = experience.model_out.compute_unit_value(
        experience.action_info.units_mask
    )
    unit_advantages_np, unit_returns_np = boostrap_unit_factorized_value(
        value_estimate=unit_value_estimate.cpu().numpy(),
        reward=experience.reward,
        done=experience.done,
        units_died=torch.logical_not(experience.action_info.units_mask).cpu().numpy(),
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
    )
    agent_value_estimate = experience.model_out.compute_agent_value(
        experience.action_info.units_mask
    )
    _, agent_returns_np = bootstrap_value(
        value_estimate=agent_value_estimate.cpu().numpy(),
        reward=experience.reward,
        done=experience.done,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
    )
    unit_advantages = torch.from_numpy(unit_advantages_np)
    unit_returns = torch.from_numpy(unit_returns_np)
    agent_returns = torch.from_numpy(agent_returns_np)
    experience.validate()
    # Combine batch/player dims for experience batch, advantages, and returns
    experience = experience.trim().flatten(end_dim=2)
    unit_advantages = unit_advantages.flatten(start_dim=0, end_dim=2)
    unit_returns = unit_returns.flatten(start_dim=0, end_dim=2)
    agent_returns = agent_returns.flatten(start_dim=0, end_dim=2)
    aggregated_stats = collections.defaultdict(list)
    for _ in range(cfg.epochs_per_update):
        assert experience.done.shape[0] == cfg.full_batch_size
        for minibatch_start in range(0, cfg.full_batch_size, cfg.train_batch_size):
            minibatch_slice = slice(
                minibatch_start, minibatch_start + cfg.train_batch_size
            )
            batch_stats = update_model_on_batch(
                train_state=train_state,
                experience=experience.index(minibatch_slice).to_device(cfg.device),
                unit_advantages=unit_advantages[minibatch_slice].to(cfg.device),
                unit_returns=unit_returns[minibatch_slice].to(cfg.device),
                agent_returns=agent_returns[minibatch_slice].to(cfg.device),
                cfg=cfg,
            )
            for k, v in batch_stats.items():
                aggregated_stats[k].append(v)

    return {key: np.mean(val).item() for key, val in aggregated_stats.items()}


def boostrap_unit_factorized_value(
    value_estimate: npt.NDArray[np.float32],
    reward: npt.NDArray[np.float32],
    done: npt.NDArray[np.bool],
    units_died: npt.NDArray[np.bool],
    gamma: float,
    gae_lambda: float,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    steps_plus_one, envs, players = reward.shape
    unit_advantages: npt.NDArray[np.float32] = np.zeros(
        (steps_plus_one - 1, envs, players, MAX_UNITS), dtype=np.float32
    )
    last_unit_gae_lambda = np.zeros((envs, players, MAX_UNITS), dtype=np.float32)
    for t, last_value in reversed(list(enumerate(value_estimate))[:-1]):
        step_reward = reward[t + 1, ..., None]
        step_done = done[t + 1, ..., None] | units_died[t + 1]
        next_value = value_estimate[t + 1]
        delta = step_reward + gamma * next_value * (1.0 - step_done) - last_value
        last_unit_gae_lambda = (
            delta + gamma * gae_lambda * (1.0 - step_done) * last_unit_gae_lambda
        )
        unit_advantages[t] = last_unit_gae_lambda

    unit_returns: npt.NDArray[np.float32] = unit_advantages + value_estimate[:-1]
    return unit_advantages, unit_returns


def update_model_on_batch(
    train_state: TrainStateT,
    experience: ExperienceBatch,
    unit_advantages: torch.Tensor,
    unit_returns: torch.Tensor,
    agent_returns: torch.Tensor,
    cfg: UnitFactorizedPPOConfig,
) -> dict[str, float]:
    # NB: Batch of observations is random, with no association between
    # trajectories, games, and players
    with torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=cfg.use_mixed_precision
    ):
        new_out = train_state.model(
            obs=experience.obs,
            action_info=experience.action_info,
        )._replace(
            main_actions=experience.model_out.main_actions,
            sap_actions=experience.model_out.sap_actions,
        )
        action_probability_ratio = (
            new_out.get_unit_log_probs() - experience.model_out.get_unit_log_probs()
        ).exp()
        clip_fraction: float = (
            ((action_probability_ratio - 1.0).abs() > cfg.clip_coefficient)
            .float()
            .mean()
            .item()
        )
        policy_loss = (
            compute_pg_loss(
                unit_advantages,
                action_probability_ratio,
                cfg.clip_coefficient,
            )
            * cfg.loss_coefficients.policy
        )
        unit_value_loss = (
            compute_value_loss(
                new_out.factorized_value[experience.action_info.units_mask],
                unit_returns[experience.action_info.units_mask],
            )
            * cfg.loss_coefficients.unit_value
        )
        baseline_value_loss = (
            compute_value_loss(
                new_out.compute_agent_value(
                    units_mask=experience.action_info.units_mask
                ),
                agent_returns,
            )
            * cfg.loss_coefficients.baseline_value
        )
        negative_entropy = compute_entropy_loss(
            new_out.main_log_probs,
            new_out.sap_log_probs,
        )
        entropy_loss = negative_entropy * cfg.loss_coefficients.entropy
        total_loss = policy_loss + unit_value_loss + baseline_value_loss + entropy_loss

    step_optimizer(train_state, total_loss, cfg)
    return {
        "total_loss": total_loss.item(),
        "policy_loss": policy_loss.item(),
        "unit_value_loss": unit_value_loss.item(),
        "baseline_value_loss": baseline_value_loss.item(),
        "entropy_loss": entropy_loss.item(),
        "clip_fraction": clip_fraction,
        "entropy": -negative_entropy.item(),
    }


def step_optimizer(
    train_state: TrainStateT,
    total_loss: torch.Tensor,
    cfg: UnitFactorizedPPOConfig,
) -> None:
    train_state.optimizer.zero_grad()
    if not cfg.use_mixed_precision:
        total_loss.backward()
        train_state.optimizer.step()
        return

    train_state.scaler.scale(total_loss).backward()
    train_state.scaler.step(train_state.optimizer)
    train_state.scaler.update()


if __name__ == "__main__":
    main()
