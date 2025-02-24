import math

from torch import nn
from torch.nn.init import constant_, orthogonal_


def orthogonal_initialization_(
    layer: nn.Module,
    scale: float = math.sqrt(2.0),
    bias: float = 0.0,
    strict: bool = True,
) -> None:
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        orthogonal_(layer.weight, scale)
        if layer.bias is not None:
            constant_(layer.bias, bias)

        return

    if isinstance(layer, nn.MultiheadAttention):
        if layer.in_proj_weight is not None:
            orthogonal_(layer.in_proj_weight, scale)
        else:
            orthogonal_(layer.q_proj_weight, scale)
            orthogonal_(layer.k_proj_weight, scale)
            orthogonal_(layer.v_proj_weight, scale)

        if layer.in_proj_bias is not None:
            constant_(layer.in_proj_bias, bias)
            constant_(layer.out_proj.bias, bias)

        if layer.bias_k is not None:
            constant_(layer.bias_k, bias)

        if layer.bias_v is not None:
            constant_(layer.bias_v, bias)

        return

    if strict:
        raise ValueError(f"Unexpected layer type: {type(layer)}")
