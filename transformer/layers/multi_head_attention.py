import math
import torch
import torch.nn as nn

from torch import Tensor


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        """
        Args
            d_model: hidden dimension
            num_heads: number of heads
            dropout: probability number of elements to zero during training
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0, "d_model is not divisible by h"

        self.d_k = d_model // num_heads

        self.w_query = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            dtype=torch.float32,
        )
        self.w_key = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            dtype=torch.float32,
        )
        self.w_value = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            dtype=torch.float32,
        )
        self.w_o = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            dtype=torch.float32,
        )  # h * self.d_k == d_model

        self.dropout = nn.Dropout(p=dropout)

    @staticmethod
    def attention(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor,
        dropout: nn.Dropout | None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args
            query: query tensor, shape `(batch_size, num_heads, seq_length, d_k)`
            key: key tensor, shape `(batch_size, num_heads, seq_length, d_k)`
            value: value tensor, shape `(batch_size, num_heads, seq_length, d_k)`
            mask: mask tensor, shape `(batch_size, 1, 1, seq_length)`
            dropout: nn.Dropout
        Returns
            output: Tensor with shape `(batch_size, num_heads, seq_length, d_k)`
            attention_scores: Tensor with shape `(batch_size, num_heads, seq_length, seq_length)`
        """
        d_k: int = query.shape[-1]

        # (batch_size, h, seq_length, d_k) @ (batch_size, h, d_k, seq_length) = (batch_size, h, seq_length, seq_length)
        attention_scores: Tensor = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        # Set the masked positions (mask == 0) to -1e9
        if mask is not None:
            attention_scores = attention_scores.masked_fill_(
                mask=(mask == 0),
                value=float("-inf"),
            )

        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (batch_size, h, seq_length, seq_length) @ (batch_size, h, seq_length, d_k) = (batch_size, h, seq_length, d_k)
        output = torch.matmul(attention_scores, value)
        return output, attention_scores

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor) -> Tensor:
        """
        Args
            q: query tensor, shape `(batch_size, seq_length, d_model)`
            k: query tensor, shape `(batch_size, seq_length, d_model)`
            v: query tensor, shape `(batch_size, seq_length, d_model)`
            mask: mask tensor, shape `(batch_size, 1, 1, seq_length)`
        Returns
            Tensor with shape `(batch_size, seq_length, d_model)`
        """
        # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)
        query: Tensor = self.w_query(q)
        key: Tensor = self.w_key(k)
        value: Tensor = self.w_value(v)

        # h * d_k == d_model
        # [query | key | value](batch_size, seq_length, d_model)
        # --> (batch_size, seq_length, h, d_k)
        # --> (batch_size, h, seq_length, d_k)
        batch_size, seq_length = query.shape[0], query.shape[1]
        query = query.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.d_k,
        ).transpose(1, 2)
        key = key.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.d_k,
        ).transpose(1, 2)
        value = value.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.d_k,
        ).transpose(1, 2)

        attn_weights, attn_scores = MultiHeadAttention.attention(
            query=query,
            key=key,
            value=value,
            mask=mask,
            dropout=self.dropout,
        )

        # (batch_size, h, seq_length, d_k) --> (batch_size, seq_length, h, d_k) -> (bathch_size, seq_length, d_model)
        attn_weights = (
            attn_weights.transpose(1, 2)
            .contiguous()
            .view(attn_weights.shape[0], -1, self.num_heads * self.d_k)
        )

        # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)
        output = self.w_o(attn_weights)
        return output
