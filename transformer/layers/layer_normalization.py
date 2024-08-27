import torch
import torch.nn as nn

from torch import Tensor


class LayerNormalization(nn.Module):
    """
    Args:
        features: dimension of the input
        epsilon: small number to avoid division by zero when std is zero or close to zero
    x(l) = (x - mean) / (std + epsilon) * alpha + bias
    alpha, bias: learnable parameters
    """

    def __init__(self, features: int, epsilon: float = 10**-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(features))  # alpha (multiplicative)
        self.bias = nn.Parameter(torch.zeros(features))  # bias (additive)

    def forward(self, x: Tensor) -> Tensor:
        # x (batch_size, seq_length, d_model)
        mean = x.mean(dim=-1, keepdim=True)  # (batch_size, seq_length, 1)
        std = x.std(dim=-1, keepdim=True)  # (batch_size, seq_length, 1)
        return (x - mean) / (std + self.epsilon) * self.alpha + self.bias
