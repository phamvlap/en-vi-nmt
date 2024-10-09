import math
import torch
import torch.nn as nn

from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_length: int, dropout: float) -> None:
        """
        Args
            d_model: hidden dimension
            seq_length: length of sequence
            dropout: probability number of elemnets to zero during training
        """
        super().__init__()
        self.d_model = d_model
        self.seq_length = seq_length
        self.dropout = nn.Dropout(p=dropout)

        # Create matrix store positional encoding of sequence
        # pe(seq_length, d_model)
        pe = torch.zeros(seq_length, d_model)

        # Define matrix store position of each word in sequence
        # position(seq_length, 1)
        position = torch.arange(0, seq_length, dtype=torch.float32).unsqueeze(dim=1)

        # p = 1 / 10000 ** (2i / d_model) = exp(-log(10000) * 2i / d_model)
        # div_term(1, d_model/2)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        # pe(pos, 2i) = sin(pos / 10000 ** (2i / d_model))
        # pe(pos, 2i + 1) = cos(pos / 10000 ** (2i / d_model))
        # i: position in embedding vector [0, 2, ..., d_model-1]
        # position(seq_length, 1) * div_term(1, d_model/2) = (seq_length, d_model/2)

        pe[:, ::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(dim=0)  # pe(1, seq_length, d_model)

        # Register pe as buffer
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args
            x: input tensor, shape `(batch_size, seq_length, d_model)`
        Returns
            Tensor with shape `(batch_size, seq_length, d_model)`
        """
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
