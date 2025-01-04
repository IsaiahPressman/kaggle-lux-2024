import math

from torch import nn


def orthogonal_initialization_(
    layer: nn.Module,
    scale: float = math.sqrt(2.0),
    bias: float = 0.0,
    strict: bool = False,
) -> None:
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, scale)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, bias)

        return

    if strict:
        raise ValueError(f"Unexpected layer type: {type(layer)}")
