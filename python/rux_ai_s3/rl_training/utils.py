import datetime
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import coloredlogs
import torch
import wandb
import yaml
from torch import nn

from rux_ai_s3.rl_training.constants import TRAIN_OUTPUTS_DIR
from rux_ai_s3.rl_training.ppo import TrainState


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_logger(logger: logging.Logger) -> None:
    coloredlogs.install(
        level=logging.INFO,
        logger=logger,
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


def init_train_dir(
    name: str,
    cfg_dict: dict[str, Any],
) -> None:
    start_time = datetime.datetime.now()
    train_dir = (
        TRAIN_OUTPUTS_DIR
        / name
        / start_time.strftime("%Y_%m_%d")
        / start_time.strftime("%H_%M")
    )
    train_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(train_dir)
    with open("train_config.yaml", "w") as f:
        yaml.dump(cfg_dict, f, sort_keys=False)


def save_checkpoint(
    train_state: TrainState[Any],
    logger: logging.Logger,
) -> None:
    checkpoint_name = f"checkpoint_{str(train_state.step).zfill(6)}"
    full_path = f"{checkpoint_name}.pt"
    weights_path = f"{checkpoint_name}_weights.pt"
    torch.save(
        {
            "step": train_state.step,
            "run_id": wandb.run.id if wandb.run else None,
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
    logger.info(
        "Full checkpoint saved to %s and weights saved to %s",
        full_path,
        weights_path,
    )


@dataclass(frozen=True)
class WandbInitConfig:
    project: str
    group: str


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
    train_state.scaler.load_state_dict(checkpoint_state["scaler"])
    train_state.step = checkpoint_state["step"]
    run_id: str | None = checkpoint_state["run_id"]
    if wandb_init_config and run_id:
        wandb.init(
            project=wandb_init_config.project,
            group=wandb_init_config.group,
            id=run_id,
        )
