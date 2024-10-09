import math
import torch.nn as nn

from torch import Tensor


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        Args
            d_model: hidden dimension
            vocab_size: size of vocabulary
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args
            x: input tensor, shape `(batch_size, seq_length)`
        Returns
            Tensor with shape `(batch_size, seq_length, d_model)`
        """
        output = self.embedding(x) * math.sqrt(self.d_model)
        return output
