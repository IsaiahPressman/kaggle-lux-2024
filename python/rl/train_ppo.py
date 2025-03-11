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
import torch.multiprocessing as mp
import wandb
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)
from rux_ai_s3.lowlevel import RewardSpace, assert_release_build
from rux_ai_s3.models.actor_critic import ActorCritic, ActorCriticOut
from rux_ai_s3.models.build import (
    ActorCriticConfigT,
    ActorCriticConfigWrapper,
    build_actor_critic,
)
from rux_ai_s3.models.types import TorchActionInfo, TorchObs
from rux_ai_s3.models.utils import remove_compile_prefix
from rux_ai_s3.parallel_env import EnvConfig, ParallelEnv
from rux_ai_s3.rl_training.constants import PROJECT_NAME
from rux_ai_s3.rl_training.evaluation import run_match
from rux_ai_s3.rl_training.ppo import (
    bootstrap_value,
    compute_entropy_loss,
    compute_pg_loss,
    compute_teacher_kl_loss,
    compute_teacher_value_loss,
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
    load_model_weights,
    save_checkpoint,
    validate_checkpoint_with_config_path,
    validate_file_path,
)
from rux_ai_s3.types import Stats
from rux_ai_s3.utils import load_from_yaml
from torch import optim
from torch.amp import GradScaler  # type: ignore[attr-defined]
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR

TrainStateT = TrainState[ActorCritic]
FILE: Final[Path] = Path(__file__)
NAME: Final[str] = "ppo"
DEFAULT_CONFIG_FILE: Final[Path] = FILE.parent / "config" / f"{NAME}_default.yaml"
CHECKPOINT_FREQ: Final[datetime.timedelta] = datetime.timedelta(minutes=20)
WIN_RATE_THRESHOLD: Final[float] = 0.8
CPU: Final[torch.device] = torch.device("cpu")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

logger = logging.getLogger()


class UserArgs(BaseModel):
    debug: bool
    config: Path | None
    model_weights: Path | None
    checkpoint: Path | None

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    _validate_config = field_validator("config")(validate_file_path)
    _validate_model_weights = field_validator("model_weights")(validate_file_path)

    @property
    def release(self) -> bool:
        return not self.debug

    @property
    def config_file(self) -> Path:
        if self.checkpoint:
            return get_config_path_from_checkpoint(self.checkpoint)

        if self.config:
            return self.config

        return DEFAULT_CONFIG_FILE

    @property
    def checkpoint_dir(self) -> Path | None:
        if self.checkpoint is None:
            return None

        return self.checkpoint.parent

    @field_validator("checkpoint")
    @classmethod
    def _validate_checkpoint(
        cls, checkpoint: Path | None, info: ValidationInfo
    ) -> Path | None:
        if checkpoint is None:
            return None

        for field in ("config", "model_weights"):
            if info.data[field] is not None:
                raise ValueError(f"checkpoint and {field} are mutually exclusive")

        return validate_checkpoint_with_config_path(checkpoint)

    @classmethod
    def from_argparse(cls) -> "UserArgs":
        parser = argparse.ArgumentParser()
        parser.add_argument("--debug", action="store_true")
        parser.add_argument(
            "--config",
            type=Path,
            default=None,
            help="Path to load training config from",
        )
        parser.add_argument(
            "--model_weights",
            type=Path,
            default=None,
            help="Path to load model weights from",
        )
        parser.add_argument(
            "--checkpoint",
            type=Path,
            default=None,
            help="Checkpoint to resume training from",
        )
        args = parser.parse_args()
        return UserArgs(**vars(args))


class LRScheduleConfig(BaseModel):
    steps: Annotated[int, Field(ge=10_000, lt=1_000_000)]
    min_factor: Annotated[float, Field(gt=0.0, lt=1.0)]

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


class TeacherLossSchedule(BaseModel):
    start: Annotated[float, Field(ge=0.0, le=1.0)]
    end: Annotated[float, Field(ge=0.0, le=1.0)]
    steps: Annotated[int, Field(ge=100, lt=1_000_000)]

    @field_validator("end")
    @classmethod
    def _validate_end(cls, end: float, info: ValidationInfo) -> float:
        start = info.data["start"]
        if end > start:
            raise ValueError(f"end {end} > start {start}")

        return end

    def get_coefficient(self, step: int) -> float:
        decay_progress = 1.0 - min(step / self.steps, 1.0)
        decayed_coeff = (self.start - self.end) * decay_progress
        return self.end + decayed_coeff

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


class LossCoefficients(BaseModel):
    policy: Annotated[float, Field(ge=0.0, le=1.0)]
    value: Annotated[float, Field(ge=0.0, le=1.0)]
    entropy: Annotated[float, Field(ge=0.0, le=1e-3)]
    teacher_policy: TeacherLossSchedule
    teacher_value: TeacherLossSchedule

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    def get_weighted_loss_coefficients(
        self, step: int
    ) -> tuple[float, float, float, float, float]:
        teacher_policy = self.teacher_policy.get_coefficient(step)
        teacher_value = self.teacher_value.get_coefficient(step)
        total = self.policy + self.value + self.entropy + teacher_policy + teacher_value
        return (
            self.policy / total,
            self.value / total,
            self.entropy / total,
            teacher_policy / total,
            teacher_value / total,
        )


class PPOConfig(TrainConfig):
    # Training config
    max_updates: int | None
    optimizer_kwargs: dict[str, float]
    lr_schedule: LRScheduleConfig
    clip_gradients: float | None
    steps_per_update: int
    epochs_per_update: int
    train_batch_size: int
    use_mixed_precision: bool
    teacher_path: Path | None

    gamma: float
    gae_lambda: float
    clip_coefficient: float
    loss_coefficients: LossCoefficients

    # Config objects
    env_config: EnvConfig
    rl_model_config: ActorCriticConfigT

    # Miscellaneous config
    device: torch.device
    eval_games: Annotated[int, Field(ge=0)]
    eval_device: torch.device
    log_histograms: bool

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    @field_validator("teacher_path")
    @classmethod
    def _validate_teacher_path(cls, path: Path | None) -> Path | None:
        if path is None:
            return None

        return path.absolute()

    @field_serializer("teacher_path")
    def serialize_teacher_path(self, teacher_path: Path | None) -> str | None:
        if teacher_path is None:
            return None

        return str(teacher_path)

    @field_validator("device", "eval_device", mode="before")
    @classmethod
    def _validate_device(cls, device: torch.device | str) -> torch.device:
        if isinstance(device, torch.device):
            return device

        return torch.device(device)

    @model_validator(mode="after")
    def _validate_model(self) -> "PPOConfig":
        if self.full_batch_size % self.train_batch_size != 0:
            raise ValueError(
                f"full_batch_size {self.full_batch_size} % train_batch_size "
                f"{self.train_batch_size} != 0"
            )

        return self

    @field_serializer("device", "eval_device")
    def serialize_device(self, device: torch.device) -> str:
        return str(device)

    @property
    def full_batch_size(self) -> int:
        return self.env_config.n_envs * self.steps_per_update * 2

    @property
    def game_steps_per_update(self) -> int:
        return self.env_config.n_envs * self.steps_per_update

    @property
    def batches_per_epoch(self) -> int:
        return math.ceil(self.full_batch_size / self.train_batch_size)

    @property
    def eval_env_config(self) -> EnvConfig:
        return EnvConfig(
            n_envs=min(self.eval_games // 2, self.env_config.n_envs),
            frame_stack_len=self.env_config.frame_stack_len,
            sap_masking=self.env_config.sap_masking,
            reward_space=RewardSpace.FINAL_WINNER,
            jax_device=self.env_config.jax_device,
        )

    def iter_updates(self) -> Generator[int, None, None]:
        if self.max_updates is None:
            return (i for i in itertools.count(1))

        return (i for i in range(1, self.max_updates + 1))

    def load_teacher_config(self) -> tuple[Path, ActorCriticConfigT] | None:
        if self.teacher_path is None:
            return None

        teacher_config_path = get_config_path_from_checkpoint(self.teacher_path)
        with open(teacher_config_path) as f:
            data = yaml.safe_load(f)

        parsed_config = ActorCriticConfigWrapper(config=data["rl_model_config"])
        return self.teacher_path, parsed_config.config


class EvalProcess:
    def __init__(
        self,
        current_model: ActorCritic,
        last_best_model: ActorCritic,
        cfg: PPOConfig,
    ) -> None:
        self.current_model = current_model
        self.start: mp.Queue[int] = mp.Queue(maxsize=1)
        self.results: mp.Queue[tuple[float | None, int]] = mp.Queue(maxsize=1)
        self.in_progress: list[int] = []
        self.process = mp.Process(
            target=model_evaluation_process,
            args=(
                cfg.model_dump_json(),
                self.current_model,
                last_best_model,
                self.start,
                self.results,
            ),
            daemon=True,
        )
        self.process.start()

    def get_result_if_available(self) -> tuple[float, int] | None:
        if self.results.empty():
            return None

        win_rate, step = self.results.get_nowait()
        self.in_progress.remove(step)
        if win_rate is None:
            return None

        return win_rate, step

    def start_evaluation(self, step: int, model: ActorCritic) -> None:
        if self.in_progress:
            logger.warning("Interrupting training to wait on evaluation thread")
            wait_time = self._wait_for_result()
            logger.warning(
                "Waited %d seconds for evaluation to complete",
                round(wait_time.total_seconds()),
            )

        self.current_model.load_state_dict(
            {
                remove_compile_prefix(key): value
                for key, value in model.state_dict().items()
            }
        )
        self.in_progress.append(step)
        self.start.put_nowait(step)

    def _wait_for_result(self) -> datetime.timedelta:
        wait_start = datetime.datetime.now()
        while self.results.empty():
            if not self.process.is_alive():
                raise RuntimeError("Evaluation process failed")

            time.sleep(1.0)

        return datetime.datetime.now() - wait_start


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

    def index(self, ix: slice | npt.NDArray[np.int_]) -> "ExperienceBatch":
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


def main() -> None:  # noqa: C901
    args = UserArgs.from_argparse()
    if args.release:
        assert_release_build()

    cfg = load_from_yaml(PPOConfig, args.config_file)
    init_train_dir(NAME, cfg.model_dump(), checkpoint_dir=args.checkpoint_dir)
    init_logger(logger=logger)
    logger.info("Loaded config from %s", args.config_file)
    env = ParallelEnv.from_config(cfg.env_config)
    model = build_model(env, cfg.rl_model_config).to(cfg.device).train()
    last_best_model = (
        build_model(env, cfg.rl_model_config).to(cfg.eval_device).eval().share_memory()
    )
    eval_model = (
        build_model(env, cfg.rl_model_config).to(cfg.eval_device).eval().share_memory()
    )
    if args.model_weights:
        load_model_weights(
            model,
            args.model_weights,
            logger=logger,
        )

    last_best_model.load_state_dict(
        {remove_compile_prefix(key): value for key, value in model.state_dict().items()}
    )
    logger.info(
        "Training model with %s parameters", f"{count_trainable_params(model):,d}"
    )
    # TODO: Use last_best as teacher model
    teacher_model = load_teacher_model(env, cfg)
    if args.release:
        model = torch.compile(model)  # type: ignore[assignment]
        if teacher_model is not None:
            teacher_model = torch.compile(teacher_model)  # type: ignore[assignment]

    optimizer = optim.Adam(model.parameters(), **cfg.optimizer_kwargs)  # type: ignore[arg-type]
    lr_scheduler = build_lr_scheduler(cfg, optimizer)
    train_state = TrainState(
        model=model,
        teacher_model=teacher_model,
        last_best_model=last_best_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        scaler=GradScaler("cuda"),
    )
    eval_process = EvalProcess(
        current_model=eval_model,
        last_best_model=last_best_model,
        cfg=cfg,
    )
    if args.checkpoint:
        wandb_init_config = (
            WandbInitConfig(
                project=PROJECT_NAME,
                group=NAME,
                config_dict=cfg.model_dump(),
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

    if wandb.run:
        wandb.watch(
            train_state.model,
            log="all",
            log_freq=200 * cfg.batches_per_epoch,
        )

    last_checkpoint = datetime.datetime.now()
    try:
        for _ in cfg.iter_updates():
            eval_result = eval_process.get_result_if_available()
            if eval_result is None:
                extra_log_info = {}
            else:
                (win_rate, eval_step) = eval_result
                extra_log_info = {"win_rate_vs_last_best": win_rate}
                logger.info(
                    "Win rate vs last best on step %d: %.4f", eval_step, win_rate
                )
                if win_rate > WIN_RATE_THRESHOLD:
                    logger.info("Updated last best on step %d", eval_step)
                    train_state.last_best_model.load_state_dict(
                        eval_process.current_model.state_dict()
                    )

            train_step(
                env=env,
                train_state=train_state,
                cfg=cfg,
                extra_log_info=extra_log_info,
            )
            now = datetime.datetime.now()
            if now - last_checkpoint < CHECKPOINT_FREQ:
                continue

            last_checkpoint = now
            save_checkpoint(
                train_state,
                logger,
            )
            eval_process.start_evaluation(train_state.step, train_state.model)
    finally:
        save_checkpoint(train_state, logger)


def build_model(
    env: ParallelEnv,
    config: ActorCriticConfigT,
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


def load_teacher_model(
    env: ParallelEnv,
    cfg: PPOConfig,
) -> ActorCritic | None:
    wp_tc = cfg.load_teacher_config()
    if not wp_tc:
        return None

    (weights_path, teacher_cfg) = wp_tc
    model = build_model(
        env=env,
        config=teacher_cfg,
    )
    load_model_weights(model, weights_path, logger=logger, model_name="teacher")
    return model.to(cfg.device).eval()


def build_lr_scheduler(
    cfg: PPOConfig,
    optimizer: optim.Optimizer,
) -> LambdaLR:
    def lr_lambda(epoch: int) -> float:
        pct_remaining = max(1.0 - epoch / cfg.lr_schedule.steps, 0.0)
        return cfg.lr_schedule.min_factor + pct_remaining * (
            1.0 - cfg.lr_schedule.min_factor
        )

    return LambdaLR(
        optimizer,
        lr_lambda,
    )


def train_step(
    env: ParallelEnv,
    train_state: TrainStateT,
    cfg: PPOConfig,
    extra_log_info: dict[str, float],
) -> None:
    step_start_time = time.perf_counter()
    experience, stats = collect_trajectories(env, train_state.model, cfg)
    data_collection_time = time.perf_counter() - step_start_time
    scalar_stats = update_model(experience, train_state, cfg) | extra_log_info
    train_time = time.perf_counter() - step_start_time - data_collection_time
    array_stats = {}
    if stats:
        scalar_stats.update(stats.scalar_stats)
        array_stats.update(stats.array_stats)

    (scalar_stats["lr"],) = train_state.lr_scheduler.get_last_lr()
    scalar_stats["game_steps"] = train_state.step * cfg.game_steps_per_update
    assert experience.reward.shape[-1] == 2
    nonzero_reward = experience.reward[experience.reward != 0.0]
    if len(nonzero_reward) > 0:
        nonzero_reward = nonzero_reward.reshape(-1, 2)
        scalar_stats["mean_nonzero_reward"] = nonzero_reward.mean().item()
        scalar_stats["p1_mean_nonzero_reward"] = nonzero_reward[..., 0].mean().item()
        scalar_stats["p2_mean_nonzero_reward"] = nonzero_reward[..., 1].mean().item()

    if train_state.finished_warmup:
        total_time = time.perf_counter() - step_start_time
        scalar_stats["updates_per_second"] = (
            cfg.batches_per_epoch * cfg.epochs_per_update / total_time
        )
        scalar_stats["env_steps_per_second"] = (
            cfg.env_config.n_envs * cfg.steps_per_update / total_time
        )
        scalar_stats["update_data_collection_time"] = data_collection_time
        scalar_stats["update_train_time"] = train_time
        scalar_stats["update_total_time"] = total_time

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
    train_state.increment_step()


def model_evaluation_process(
    cfg_json: str,
    current_model: ActorCritic,
    last_best_model: ActorCritic,
    start: "mp.Queue[int]",
    results: "mp.Queue[tuple[float | None, int]]",
) -> None:
    cfg = PPOConfig.model_validate_json(cfg_json)
    env = ParallelEnv.from_config(cfg.eval_env_config)
    while True:
        step = start.get()
        if cfg.eval_games <= 0:
            results.put((None, step))
            continue

        win_rate = evaluate_model(
            env,
            model=current_model,
            last_best=last_best_model,
            cfg=cfg,
        )
        results.put((win_rate, step))


@torch.no_grad()
def evaluate_model(
    eval_env: ParallelEnv,
    model: ActorCritic,
    last_best: ActorCritic,
    cfg: PPOConfig,
) -> float:
    # Could use different spda kernel if eval GPU supports flash attention
    with torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=cfg.use_mixed_precision
    ), sdpa_kernel(SDPBackend.MATH):
        w1, l1 = run_match(
            eval_env,
            (model, last_best),
            cfg.eval_games // 2,
            cfg.eval_device,
        )
        l2, w2 = run_match(
            eval_env,
            (last_best, model),
            cfg.eval_games // 2,
            cfg.eval_device,
        )

    return (w1 + w2) / (w1 + w2 + l1 + l2)


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
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=cfg.use_mixed_precision
        ):
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
            stats = Stats.merge(
                stats, env.last_out.stats, include_array_stats=cfg.log_histograms
            )

    experience = ExperienceBatch.from_lists(
        obs=batch_obs,
        action_info=batch_action_info,
        model_out=batch_model_out,
        reward=batch_reward,
        done=batch_done,
    )
    return experience, stats


def update_model(
    full_experience: ExperienceBatch,
    train_state: TrainStateT,
    cfg: PPOConfig,
) -> dict[str, float]:
    full_experience.validate()
    aggregated_stats = collections.defaultdict(list)
    for _ in range(cfg.epochs_per_update):
        experience, advantages, returns = prepare_epoch_data(full_experience, cfg)
        for minibatch_start in range(0, cfg.full_batch_size, cfg.train_batch_size):
            minibatch_slice = slice(
                minibatch_start, minibatch_start + cfg.train_batch_size
            )
            batch_stats = update_model_on_batch(
                train_state=train_state,
                experience=experience.index(minibatch_slice).to_device(cfg.device),
                advantages=advantages[minibatch_slice].to(
                    cfg.device, non_blocking=True
                ),
                returns=returns[minibatch_slice].to(cfg.device, non_blocking=True),
                cfg=cfg,
            )
            for k, v in batch_stats.items():
                aggregated_stats[k].append(v)

    return {key: np.mean(val).item() for key, val in aggregated_stats.items()}


def prepare_epoch_data(
    full_experience: ExperienceBatch,
    cfg: PPOConfig,
) -> tuple[ExperienceBatch, torch.Tensor, torch.Tensor]:
    advantages_np, returns_np = bootstrap_value(
        value_estimate=full_experience.model_out.value.cpu().numpy(),
        reward=full_experience.reward,
        done=full_experience.done,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
    )
    advantages = torch.from_numpy(advantages_np)
    returns = torch.from_numpy(returns_np)
    # Combine step/batch/player dims for experience, advantages, and returns
    experience = full_experience.trim().flatten(end_dim=2)
    advantages = advantages.flatten(start_dim=0, end_dim=2)
    returns = returns.flatten(start_dim=0, end_dim=2)
    assert experience.done.shape[0] == cfg.full_batch_size
    return experience, advantages, returns


def update_model_on_batch(
    train_state: TrainStateT,
    experience: ExperienceBatch,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    cfg: PPOConfig,
) -> dict[str, float]:
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
        if train_state.teacher_model is not None:
            with torch.no_grad():
                teacher_out = train_state.teacher_model(
                    obs=experience.obs,
                    action_info=experience.action_info,
                )

        action_probability_ratio = (
            new_out.compute_joint_log_probs()
            - experience.model_out.compute_joint_log_probs()
        ).exp()
        clip_fraction: float = (
            ((action_probability_ratio - 1.0).abs() > cfg.clip_coefficient)
            .float()
            .mean()
            .item()
        )
        (
            policy_coefficient,
            value_coefficient,
            entropy_coefficient,
            teacher_policy_coefficient,
            teacher_value_coefficient,
        ) = cfg.loss_coefficients.get_weighted_loss_coefficients(train_state.step)
        policy_loss = (
            compute_pg_loss(
                advantages,
                action_probability_ratio,
                cfg.clip_coefficient,
            )
            * policy_coefficient
        )
        value_loss = (
            compute_value_loss(
                new_out.value,
                returns,
            )
            * value_coefficient
        )
        negative_entropy = compute_entropy_loss(
            new_out.main_log_probs,
            new_out.sap_log_probs,
            units_mask=experience.action_info.units_mask,
        )
        entropy_loss = negative_entropy * entropy_coefficient
        teacher_policy_loss = (
            (
                compute_teacher_kl_loss(
                    learner_main_log_probs=new_out.main_log_probs,
                    learner_sap_log_probs=new_out.sap_log_probs,
                    teacher_main_log_probs=teacher_out.main_log_probs,
                    teacher_sap_log_probs=teacher_out.sap_log_probs,
                    units_mask=experience.action_info.units_mask,
                )
                * teacher_policy_coefficient
            )
            if train_state.teacher_model
            else torch.zeros_like(policy_loss)
        )
        teacher_value_loss = (
            (
                compute_teacher_value_loss(
                    learner_value_estimate=new_out.value,
                    teacher_value_estimate=teacher_out.value,
                )
                * teacher_value_coefficient
            )
            if train_state.teacher_model
            else torch.zeros_like(value_loss)
        )
        total_loss = (
            policy_loss
            + value_loss
            + entropy_loss
            + teacher_policy_loss
            + teacher_value_loss
        )

    step_optimizer(train_state, total_loss, cfg)
    return {
        "total_loss": total_loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy_loss": entropy_loss.item(),
        "teacher_policy_loss": teacher_policy_loss.item(),
        "teacher_value_loss": teacher_value_loss.item(),
        "clip_fraction": clip_fraction,
        "entropy": -negative_entropy.item(),
    }


def step_optimizer(
    train_state: TrainStateT,
    total_loss: torch.Tensor,
    cfg: PPOConfig,
) -> None:
    train_state.optimizer.zero_grad()
    if not cfg.use_mixed_precision:
        total_loss.backward()
        if cfg.clip_gradients:
            clip_grad_norm_(train_state.model.parameters(), cfg.clip_gradients)

        train_state.optimizer.step()
        return

    train_state.scaler.scale(total_loss).backward()
    if cfg.clip_gradients:
        train_state.scaler.unscale_(train_state.optimizer)
        clip_grad_norm_(train_state.model.parameters(), cfg.clip_gradients)

    train_state.scaler.step(train_state.optimizer)
    train_state.scaler.update()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
