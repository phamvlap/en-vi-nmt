import torch.nn as nn

from torch import Tensor
from transformer.layers.layer_normalization import LayerNormalization


class ResidualConnection(nn.Module):
    """
    Args:
        features
        dropout
    """

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNormalization(features=features)

    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        return x + self.dropout(sublayer(self.norm(x)))
