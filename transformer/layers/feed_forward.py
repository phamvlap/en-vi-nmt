import torch
import torch.nn as nn

from torch import Tensor


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """
        Args
            d_model: hidden dimension
            d_ff: dimension of feed forward layer
            dropout: probability number of elements to zero during training
        """
        super().__init__()
        self.linear1 = nn.Linear(
            in_features=d_model,
            out_features=d_ff,
            dtype=torch.float32,
        )  # w1 and b1
        self.linear2 = nn.Linear(
            in_features=d_ff,
            out_features=d_model,
            dtype=torch.float32,
        )  # w2 and b2
        # Activation function: ReLU(x) = max(0, x)
        self.act_fn = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args
            x: input tensor, shape `(batch_size, seq_length, d_model)`
        Returns
            Tensor with shape `(batch_size, seq_length, d_model)`
        """
        y: Tensor = self.linear1(x)
        y = self.act_fn(y)
        y = self.dropout(y)
        y = self.linear2(y)
        return y
