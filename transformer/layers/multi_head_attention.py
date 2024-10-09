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
        mask: Tensor | None = None,
        dropout: nn.Dropout | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args
            query: query tensor, shape `(batch_size, num_heads, seq_length, d_k)`
            key: key tensor, shape `(batch_size, num_heads, seq_length, d_k)`
            value: value tensor, shape `(batch_size, num_heads, seq_length, d_k)`
            mask: mask tensor, shape `(batch_size, 1, 1, seq_length)` or `(batch_size, 1, seq_length, seq_length)`
            dropout: nn.Dropout
        Returns
            output: Tensor with shape `(batch_size, num_heads, seq_length, d_k)`
            attn_scores: Tensor with shape `(batch_size, num_heads, seq_length, seq_length)`
        """
        d_k: int = query.shape[-1]

        # (batch_size, h, seq_length, d_k) @ (batch_size, h, d_k, seq_length) = (batch_size, h, seq_length, seq_length)
        attn_scores: Tensor = torch.matmul(
            query,
            key.transpose(-2, -1),
        ) / math.sqrt(d_k)

        # Set the masked positions (mask == 0) to -1e9
        if mask is not None:
            attn_scores.masked_fill_(
                mask=(mask == 0),
                value=float("-inf"),
            )

        attn_scores = attn_scores.softmax(dim=-1)
        if dropout is not None:
            attn_scores = dropout(attn_scores)

        # (batch_size, h, seq_length, seq_length) @ (batch_size, h, seq_length, d_k) = (batch_size, h, seq_length, d_k)
        attn_weights = torch.matmul(attn_scores, value)
        return attn_weights, attn_scores

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args
            q: query tensor, shape `(batch_size, seq_length, d_model)`
            k: query tensor, shape `(batch_size, seq_length, d_model)`
            v: query tensor, shape `(batch_size, seq_length, d_model)`
            mask: mask tensor, shape `(batch_size, 1, 1, seq_length)` or `(batch_size, 1, seq_length, seq_length)`
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

        query = self._reshape_to_4d(query, batch_size, seq_length)
        key = self._reshape_to_4d(key, batch_size, seq_length)
        value = self._reshape_to_4d(value, batch_size, seq_length)

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
            .view(batch_size, -1, self.num_heads * self.d_k)
        )

        # (batch_size, seq_length, d_model) --> (batch_size, seq_length, d_model)
        output = self.w_o(attn_weights)
        return output

    def _reshape_to_4d(self, x: Tensor, batch_size: int, seq_length: int) -> Tensor:
        """
        Args
            x: input tensor, shape `(batch_size, seq_length, d_model)`
            batch_size: batch size
            seq_length: sequence length
        Returns
            Tensor with shape `(batch_size, num_heads, seq_length, d_k)`
        """
        reshaped = x.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.d_k,
        ).transpose(1, 2)
        return reshaped
