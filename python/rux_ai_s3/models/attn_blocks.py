import torch
from torch import nn

from .types import ActivationFactory


class AttnBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        activation: ActivationFactory,
        num_heads: int,
        dropout: float | None = None,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout or 0.0,
            batch_first=True,
        )
        self.act1 = activation()

        self.linear = nn.Linear(d_model, d_model)
        self.act2 = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x should have shape (batch_size, seq_len [map_width * map_height], d_model)
        """
        identity1 = x
        x = self.mha(x, x, x, need_weights=False)
        x = self.act1(x) + identity1

        identity2 = x
        x = self.linear(x) + identity2
        return self.act2(x)
