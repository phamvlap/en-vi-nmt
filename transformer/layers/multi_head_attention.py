import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Args:
            d_model
            h: numebr of heads
            dropout
    """

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h

        self.w_query = nn.Linear(
            in_features=d_model, out_features=d_model, dtype=torch.float32
        )
        self.w_key = nn.Linear(
            in_features=d_model, out_features=d_model, dtype=torch.float32
        )
        self.w_value = nn.Linear(
            in_features=d_model, out_features=d_model, dtype=torch.float32
        )
        self.w_o = nn.Linear(
            in_features=d_model, out_features=d_model, dtype=torch.float32
        )  # h * self.d_k == d_model

        self.dropout = nn.Dropout(p=dropout)

    @staticmethod
    def attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        dropout: nn.Dropout | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d_k: int = query.shape[-1]

        # (batch_size, h, seq_length, d_k) @ (batch_size, h, d_k, seq_length) = (batch_size, h, seq_length, seq_length)
        attention_scores: torch.Tensor = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        # Set the masked positions (mask == 0) to -1e9
        if mask is not None:
            attention_scores = attention_scores.masked_fill_(
                mask=(mask == 0), value=-1e9
            )

        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (batch_size, h, seq_length, seq_length) @ (batch_size, h, seq_length, d_k) = (batch_size, h, seq_length, d_k)
        return torch.matmul(attention_scores, value), attention_scores

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)
        query: torch.Tensor = self.w_query(q)
        key: torch.Tensor = self.w_key(k)
        value: torch.Tensor = self.w_value(v)

        # h * d_k == d_model
        # [query | key | value](batch_size, seq_length, d_model)
        # --> (batch_size, seq_length, h, d_k)
        # --> (batch_size, h, seq_length, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        x, attention_scores = MultiHeadAttention.attention(
            query=query,
            key=key,
            value=value,
            mask=mask,
            dropout=self.dropout,
        )

        # (batch_size, h, seq_length, d_k) --> (batch_size, seq_length, h, d_k) -> (bathch_size, seq_length, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)
        return self.w_o(x)
