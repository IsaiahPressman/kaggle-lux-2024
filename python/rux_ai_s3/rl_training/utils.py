import datetime
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Generic, TypeVar

import coloredlogs
import torch
import wandb
import yaml
from torch import GradScaler, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from rux_ai_s3.models.utils import remove_compile_prefix
from rux_ai_s3.rl_training.constants import TRAIN_CONFIG_FILE_NAME, TRAIN_OUTPUTS_DIR

_ModelT = TypeVar("_ModelT", bound=nn.Module)
WARMUP_STEPS: Final[int] = 10


@dataclass
class TrainState(Generic[_ModelT]):
    model: _ModelT
    teacher_model: _ModelT | None
    last_best_model: _ModelT
    optimizer: Optimizer
    lr_scheduler: LRScheduler
    scaler: GradScaler
    step: int = 0
    steps_this_run: int = 0

    @property
    def finished_warmup(self) -> bool:
        return self.steps_this_run >= WARMUP_STEPS

    def increment_step(self) -> None:
        self.step += 1
        self.steps_this_run += 1


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_logger(logger: logging.Logger) -> None:
    info_file_handler = logging.FileHandler("info.log", mode="a")
    info_file_handler.setLevel(logging.INFO)
    logger.addHandler(info_file_handler)

    coloredlogs.install(
        level=logging.INFO,
        logger=logger,
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )


def init_train_dir(
    name: str,
    cfg_dict: dict[str, Any],
    checkpoint_dir: Path | None,
) -> None:
    if checkpoint_dir:
        os.chdir(checkpoint_dir)
        return

    start_time = datetime.datetime.now()
    train_dir = (
        TRAIN_OUTPUTS_DIR
        / name
        / start_time.strftime("%Y_%m_%d")
        / start_time.strftime("%H_%M_%S")
    )
    train_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(train_dir)
    with open(TRAIN_CONFIG_FILE_NAME, "w") as f:
        yaml.dump(cfg_dict, f, sort_keys=False)


def save_checkpoint(
    train_state: TrainState[Any],
    logger: logging.Logger,
) -> None:
    checkpoint_name = f"checkpoint_{str(train_state.step).zfill(6)}"
    base_path = Path(os.getcwd())
    full_path = base_path / f"{checkpoint_name}.pt"
    weights_path = base_path / f"{checkpoint_name}_weights.pt"
    torch.save(
        {
            "step": train_state.step,
            "run_id": wandb.run.id if wandb.run else None,
            "model": train_state.model.state_dict(),
            "last_best_model": train_state.last_best_model.state_dict(),
            "optimizer": train_state.optimizer.state_dict(),
            "lr_scheduler": train_state.lr_scheduler.state_dict(),
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
    logger.info(
        "Full checkpoint saved to %s and weights saved to %s",
        full_path,
        weights_path,
    )


@dataclass(frozen=True)
class WandbInitConfig:
    project: str
    group: str
    config_dict: dict[str, Any]


def load_checkpoint(
    train_state: TrainState[Any],
    checkpoint: Path,
    wandb_init_config: WandbInitConfig | None,
    logger: logging.Logger,
) -> None:
    logger.info("Loading checkpoint from %s", checkpoint)
    checkpoint_state = torch.load(
        checkpoint, map_location=torch.device("cpu"), weights_only=True
    )
    train_state.model.load_state_dict(checkpoint_state["model"])
    train_state.optimizer.load_state_dict(checkpoint_state["optimizer"])
    train_state.last_best_model.load_state_dict(checkpoint_state["last_best_model"])
    train_state.scaler.load_state_dict(checkpoint_state["scaler"])
    train_state.lr_scheduler.load_state_dict(checkpoint_state["lr_scheduler"])
    train_state.step = checkpoint_state["step"]
    run_id: str | None = checkpoint_state["run_id"]
    if wandb_init_config and run_id:
        wandb.init(
            project=wandb_init_config.project,
            group=wandb_init_config.group,
            config=wandb_init_config.config_dict,
            id=run_id,
            resume="must",
        )


def load_model_weights(
    model: nn.Module,
    weights_path: Path,
    logger: logging.Logger,
    model_name: str = "",
) -> None:
    logger.info(
        "Loading %s weights from %s",
        f"{model_name} model" if model_name else "model",
        weights_path,
    )
    checkpoint_state = torch.load(
        weights_path, map_location=torch.device("cpu"), weights_only=True
    )
    state_dict = checkpoint_state["model"]
    state_dict = {
        remove_compile_prefix(key): value for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict)


def get_config_path_from_checkpoint(checkpoint: Path) -> Path:
    return checkpoint.parent / TRAIN_CONFIG_FILE_NAME


def validate_file_path(path: Path | None) -> Path | None:
    if path is None:
        return None

    path = path.absolute()
    if not path.is_file():
        raise ValueError(f"Invalid path: {path}")

    return path


def validate_checkpoint_with_config_path(checkpoint: Path | None) -> Path | None:
    checkpoint = validate_file_path(checkpoint)
    if checkpoint is None:
        return None

    config_path = get_config_path_from_checkpoint(checkpoint)
    if not config_path.is_file():
        raise ValueError(f"Invalid checkpoint config path: {config_path}")

    return checkpoint
