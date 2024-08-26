import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    """
    Args:
            epsilon: ??
    x(l) =(x - mean) / (std + epsilon) * alpha + bias
    """

    def __init__(self, epsilon: float = 10**-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1))  # multiplicative
        self.bias = nn.Parameter(torch.zeros(1))  # additive

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) / (std + self.epsilon) * self.alpha + self.bias
