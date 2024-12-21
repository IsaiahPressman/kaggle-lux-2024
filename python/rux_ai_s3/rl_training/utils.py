import logging

import coloredlogs
from torch import nn


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_logger(logger: logging.Logger) -> None:
    coloredlogs.install(
        level=logging.INFO,
        logger=logger,
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
