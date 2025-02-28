import torch
import torch.nn.functional as F
from torch import nn

from rux_ai_s3.constants import MAP_SIZE

from .rotary_embedding_torch import (  # type: ignore[attr-defined]
    RotaryEmbedding,
    apply_rotary_emb,
)
from .types import ActivationFactory
from .weight_initialization import orthogonal_initialization_


class MLPBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        activation: ActivationFactory,
        dropout: float | None,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_mlp)
        self.activation = activation()
        self.lin_dropout1 = nn.Dropout(dropout or 0.0)
        self.lin2 = nn.Linear(d_mlp, d_model)
        self.lin_dropout2 = nn.Dropout(dropout or 0.0)
        self._init_weights()

    def _init_weights(self) -> None:
        orthogonal_initialization_(self.lin1)
        orthogonal_initialization_(self.lin2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = self.activation(x)
        x = self.lin_dropout1(x)
        x = self.lin2(x)
        return self.lin_dropout2(x)


class AttnBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        num_heads: int,
        activation: ActivationFactory,
        dropout: float | None,
    ) -> None:
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout or 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout or 0.0)
        self.mlp = MLPBlock(
            d_model=d_model,
            d_mlp=d_mlp,
            activation=activation,
            dropout=dropout,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        orthogonal_initialization_(self.mha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x should have shape (batch_size, seq_len [map_width * map_height], d_model)
        """
        x = x + self._mha_block(x)
        return x + self.mlp(x)

    def _mha_block(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.mha(x, x, x, need_weights=False)
        return self.dropout(x)


class SpatialAttnIn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        d_mlp: int,
        num_heads: int,
        activation: ActivationFactory,
        dropout: float | None,
    ) -> None:
        super().__init__()
        self.initial_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=1,
        )
        self.activation = activation()
        self.qkv_proj = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model * 3,
            kernel_size=1,
        )
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_head = d_model // num_heads
        assert self.d_head % 2 == 0
        assert self.d_head / 2 >= 8
        self.num_heads = num_heads
        self.dropout_p = dropout or 0.0
        self.rotary_emb = RotaryEmbedding(
            dim=self.d_head // 2,
            freqs_for="pixel",
            max_freq=MAP_SIZE,
        )
        self.cached_freqs: torch.Tensor = torch.empty(())
        self.mlp = MLPBlock(
            d_model=d_model,
            d_mlp=d_mlp,
            activation=activation,
            dropout=dropout,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        orthogonal_initialization_(self.initial_proj)
        orthogonal_initialization_(self.qkv_proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x should have shape (batch_size, in_channels, map_width, map_height)
        returns shape (batch_size, seq_len, d_model)
        """
        batch_size, _, width, height = x.shape
        seq_len = width * height
        freqs = self._get_freqs((width, height))

        x = self.initial_proj(x)
        x = self.activation(x)
        q, k, v = self.qkv_proj(x).permute(0, 2, 3, 1).chunk(3, dim=-1)
        # q/k/v shape is now (batch_size, w, h, d_model)
        q = q.unflatten(-1, (self.num_heads, self.d_head)).permute(0, 3, 1, 2, 4)
        k = k.unflatten(-1, (self.num_heads, self.d_head)).permute(0, 3, 1, 2, 4)
        # q/k shape is now (batch_size, num_heads, w, h, d_head)
        assert q.shape[-1] == k.shape[-1] == freqs.shape[-1]
        q = apply_rotary_emb(freqs, q).flatten(2, 3)
        k = apply_rotary_emb(freqs, k).flatten(2, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        # v shape is now also (batch_size, num_heads, seq_len, d_head), matching q/k
        assert (
            q.shape
            == k.shape
            == v.shape
            == (batch_size, self.num_heads, seq_len, self.d_head)
        )
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout_p if self.training else 0.0
        )
        # x shape is now (batch_size, num_heads, seq_len, d_head)
        x = x.permute(0, 2, 1, 3).flatten(2, 3)
        # x shape is now (batch_size, seq_len, d_model)
        return x + self.mlp(x)

    def _get_freqs(self, dim: tuple[int, int]) -> torch.Tensor:
        if self.cached_freqs.shape[:-1] != dim:
            if self.cached_freqs.shape != ():
                # This could be a warning for a more generalized implementation, but
                # for this season of Lux the map size will never change, so if it does
                # something has gone wrong
                raise ValueError(
                    f"Incorrect shape of cached_freqs: "
                    f"{self.cached_freqs.shape}[:-1] != {dim}"
                )

            self.cached_freqs = self.rotary_emb.get_axial_freqs(*dim)

        return self.cached_freqs
