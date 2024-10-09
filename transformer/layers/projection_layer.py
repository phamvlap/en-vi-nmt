import torch
import torch.nn as nn

from torch import Tensor


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        Args
            d_model: hidden dimension
            vocab_size: size of vocabulary
        ProjectionLayer = Linear + Softmax
        """
        super().__init__()
        self.proj = nn.Linear(
            in_features=d_model,
            out_features=vocab_size,
            dtype=torch.float32,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args
            x: input tensor, shape `(batch_size, seq_length, d_model)`
        Returns
            Tensor with shape `(batch_size, seq_length, vocab_size)`
        """
        # x.log_softmaxt(x_i) = ln(softmax(x_i)) = ln(exp(x_i)  / sum(exp(x_j), j=[1, 2, ..., n]))
        output = torch.log_softmax(input=self.proj(x), dim=-1)
        return output
