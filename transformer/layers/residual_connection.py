import torch.nn as nn

from torch import Tensor
from transformer.layers.layer_normalization import LayerNormalization


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        """
        Args
            features: number of features (hidden dimension - d_model)
            dropout: probability number of elements to zero during training
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNormalization(features=features)

    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        """
        Args
            x: input tensor, shape `(batch_size, seq_length, d_model)`
            sublayer: sublayer nn.Module
        Returns
            Tensor with shape `(batch_size, seq_length, d_model)`
        """
        output = x + self.dropout(sublayer(self.norm(x)))
        return output
