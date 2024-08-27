import torch
import torch.nn as nn

from torch import Tensor


class ProjectionLayer(nn.Module):
    """
    ProjectionLayer = Linear + Softmax
    Args:
            d_model
            vocab_size
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(
            in_features=d_model,
            out_features=vocab_size,
            dtype=torch.float32,
        )

    # x.log_softmaxt(x_i) = ln(softmax(x_i)) = ln(exp(x_i)  / sum(exp(x_j), j=[1, 2, ..., n]))
    def forward(self, x: Tensor) -> Tensor:
        # (batch_size, seq_length, d_model) -> (batch_size, seq_length, vocab_size)
        return torch.log_softmax(input=self.proj(x), dim=-1)
