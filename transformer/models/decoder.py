import torch.nn as nn

from torch import Tensor

from ..layers import (
    MultiHeadAttention,
    FeedForward,
    ResidualConnection,
    LayerNormalization,
)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        features: int,
        self_attention: MultiHeadAttention,
        cross_attention: MultiHeadAttention,
        feed_forward: FeedForward,
        dropout: float,
    ) -> None:
        """
        Args
            features: number of features (hidden dimension - d_model)
            self_attention: Masked Multi-Head Attention
            cross_attention: Encoder-Decoder Attention
            feed_forward: Feed Forward Neural Network
            dropout: probability number of elements to zero during traininga
        """
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features=features, dropout=dropout) for _ in range(3)]
        )

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args
            x: input tensor, shape `(batch_size, seq_length, d_model)`
            encoder_output: output tensor from encoder, shape `(batch_size, seq_length, d_model)`
            src_mask: mask tensor of encoder, shape `(batch_size, 1, 1, seq_length)`
            tgt_mask: mask tensor of decoder, shape `(batch_size, 1, seq_length, seq_length)`
        Returns
            Tensor with shape `(batch_size, seq_length, d_model)`
        """
        x = self.residual_connections[0](
            x=x,
            sublayer=lambda x: self.self_attention(q=x, k=x, v=x, mask=tgt_mask),
        )
        x = self.residual_connections[1](
            x=x,
            sublayer=lambda x: self.cross_attention(
                q=x,
                k=encoder_output,
                v=encoder_output,
                mask=src_mask,
            ),
        )
        x = self.residual_connections[2](x=x, sublayer=self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        """
        Args
            features: number of features (hidden dimension - d_model)
            layers: residual connections in Decoder block
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features=features)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
    ) -> Tensor:
        """
        Args
            x: input tensor, shape `(batch_size, seq_length, d_model)`
            encoder_output: output tensor from encoder, shape `(batch_size, seq_length, d_model)`
            src_mask: mask tensor of encoder, shape `(batch_size, 1, 1, seq_length)`
            tgt_mask: mask tensor of decoder, shape `(batch_size, 1, seq_length, seq_length)`
        Returns
            Tensor with shape `(batch_size, seq_length, d_model)`
        """
        for layer in self.layers:
            x = layer(
                x=x,
                encoder_output=encoder_output,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
            )
        output = self.norm(x)
        return output
