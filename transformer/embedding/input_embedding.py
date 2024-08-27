import math
import torch.nn as nn

from torch import Tensor


class InputEmbedding(nn.Module):
    """
    Args:
            d_model: dimension
            vocab_size: size of vocabulary
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x (batch_size, seq_length) -> x (batch_size, seq_length, d_model)
        return self.embedding(x) * math.sqrt(self.d_model)
