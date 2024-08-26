import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Args:
            d_model: dimension
            d_ff: dimension of feed forward layer
            dropout:
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(
            in_features=d_model, out_features=d_ff, dtype=torch.float32
        )  # w1 and b1
        self.linear2 = nn.Linear(
            in_features=d_ff, out_features=d_model, dtype=torch.float32
        )  # w2 and b2
        self.dropout = nn.Dropout(p=dropout)

    # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_ff) --> (batch_size, seq_length, d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: torch.Tensor = self.linear1(x)
        y = torch.relu(y)  # relu(x) = max(0, x)
        y = self.dropout(y)
        y = self.linear2(y)
        return y
