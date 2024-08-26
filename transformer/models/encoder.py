import torch
import torch.nn as nn

from transformer.layers.multi_head_attention import MultiHeadAttention
from transformer.layers.feed_forward import FeedForward
from transformer.layers.residual_connection import ResidualConnection
from transformer.layers.layer_normalization import LayerNormalization


class EncoderLayer(nn.Module):
    """
    Args:
            self_attention
            feed_forward
            dropout
    """

    def __init__(
        self,
        features: int,
        self_attention: MultiHeadAttention,
        feed_forward: FeedForward,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features=features, dropout=dropout) for _ in range(2)]
        )

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.residual_connections[0](
            x=x,
            sublayer=(lambda x: self.self_attention(q=x, k=x, v=x, mask=src_mask)),
        )
        x = self.residual_connections[1](x=x, sublayer=self.feed_forward)
        return x


class Encoder(nn.Module):
    """
    Args:
            layers: list of residual connections
    """

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features=features)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x=x, src_mask=mask)
        return self.norm(x)
