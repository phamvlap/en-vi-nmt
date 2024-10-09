import torch.nn as nn

from torch import Tensor

from transformer.layers.multi_head_attention import MultiHeadAttention
from transformer.layers.feed_forward import FeedForward
from transformer.layers.residual_connection import ResidualConnection
from transformer.layers.layer_normalization import LayerNormalization


class EncoderLayer(nn.Module):
    def __init__(
        self,
        features: int,
        self_attention: MultiHeadAttention,
        feed_forward: FeedForward,
        dropout: float,
    ) -> None:
        """
        Args
            features: number of features (hidden dimension - d_model)
            self_attention: Multi-Head Attention
            feed_forward: Feed Forward Neural Network
            dropout: probability number of elements to zero during training
        """
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features=features, dropout=dropout) for _ in range(2)]
        )

    def forward(self, x: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args
            x: input tensor, shape `(batch_size, seq_length, d_model)`
            src_mask: mask tensor, shape `(batch_size, 1, 1, seq_length)`
        Returns
            Tensor with shape `(batch_size, seq_length, d_model)`
        """
        x = self.residual_connections[0](
            x=x,
            sublayer=(lambda x: self.self_attention(q=x, k=x, v=x, mask=src_mask)),
        )
        x = self.residual_connections[1](x=x, sublayer=self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        """
        Args
            features: number of features (hidden dimension - d_model)
            layers: list of residual connections in encoder
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features=features)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Args
            x: input tensor, shape `(batch_size, seq_length, d_model)`
            mask: mask tensor, shape `(batch_size, 1, 1, seq_length)`
        Returns
            Tensor with shape `(batch_size, seq_length, d_model)`
        """
        for layer in self.layers:
            x = layer(x=x, src_mask=mask)
        output = self.norm(x)
        return output
