import torch
import torch.nn as nn

from transformer.layers.multi_head_attention import MultiHeadAttention
from transformer.layers.feed_forward import FeedForward
from transformer.layers.residual_connection import ResidualConnection
from transformer.layers.layer_normalization import LayerNormalization


class DecoderLayer(nn.Module):
    """
    Args:
            self_attention: Masked Multi-Head Attention
            cross_attention: Encoder-Decoder Attention
            feed_forward
            dropout
    """

    def __init__(
        self,
        self_attention: MultiHeadAttention,
        cross_attention: MultiHeadAttention,
        feed_forward: FeedForward,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout) for _ in range(3)]
        )

    # src_mask, tgt_mask ??
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.residual_connections[0](
            x=x,
            sublayer=lambda x: self.self_attention(q=x, k=x, v=x, mask=tgt_mask),
        )
        x = self.residual_connections[1](
            x=x,
            sublayer=lambda x: self.cross_attention(
                q=x, k=encoder_output, v=encoder_output, mask=src_mask
            ),
        )
        x = self.residual_connections[2](x=x, sublayer=self.feed_forward)
        return x


class Decoder(nn.Module):
    """
    Args:
            layers: number of residual connections in Decoder block
    """

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    # src_mask, tgt_mask ??
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x=x,
                encoder_output=encoder_output,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
            )
        return self.norm(x)
