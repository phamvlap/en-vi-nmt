import torch.nn as nn

from torch import Tensor
from dataclasses import dataclass

from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer
from ..embedding import InputEmbedding, PositionalEncoding
from ..layers import ProjectionLayer, MultiHeadAttention, FeedForward
from ..functional import create_encoder_mask, create_decoder_mask


@dataclass
class TransformerConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    src_seq_length: int
    tgt_seq_length: int
    d_model: int
    h: int
    num_encoder_layers: int
    num_decoder_layers: int
    dropout: float
    d_ff: int
    src_pad_token_id: int
    tgt_pad_token_id: int


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embedding: InputEmbedding,
        tgt_embedding: InputEmbedding,
        src_position: PositionalEncoding,
        tgt_position: PositionalEncoding,
        projection_layer: ProjectionLayer,
        src_pad_token_id: int,
        tgt_pad_token_id: int,
    ) -> None:
        """
        Args
            encoder: Encoder
            decoder: Decoder
            src_embedding: InputEmbedding for Encoder
            tgt_embedding: InputEmbedding for Decoder
            src_position: PositionalEncoding for Encoder
            tgt_position: PositionalEncoding for Decoder
            projection_layer: ProjectionLayer
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_position = src_position
        self.tgt_position = tgt_position
        self.projection_layer = projection_layer
        self.src_pad_token_id = src_pad_token_id
        self.tgt_pad_token_id = tgt_pad_token_id

    def encode(self, src: Tensor, src_mask: Tensor | None = None) -> Tensor:
        """
        Args
            src: input tensor, shape `(batch_size, seq_length)`
            src_mask: mask tensor, shape `(batch_size, 1, 1, seq_length)`
        Returns
            Tensor with shape `(batch_size, seq_length, d_model)`
        """
        if src_mask is None:
            src_mask = create_encoder_mask(src, self.src_pad_token_id)
        src = self.src_embedding(src)
        src = self.src_position(src)
        output = self.encoder(x=src, src_mask=src_mask)
        return output

    def decode(
        self,
        encoder_output: Tensor,
        tgt: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args
            encoder_output: output tensor from encoder, shape `(batch_size, seq_length, d_model)`
            tgt: input tensor, shape `(batch_size, seq_length)`
            src_mask: mask tensor, shape `(batch_size, 1, 1, seq_length)`
            tgt_mask: mask tensor, shape `(batch_size, 1, seq_length, seq_length)`
        Returns
            Tensor with shape `(batch_size, seq_length, d_model)`
        """
        if tgt_mask is None:
            tgt_mask = create_decoder_mask(tgt, self.tgt_pad_token_id)
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_position(tgt)
        output = self.decoder(
            x=tgt,
            encoder_output=encoder_output,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
        )
        return output

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args
            src: input tensor, shape `(batch_size, seq_length)`
            tgt: input tensor, shape `(batch_size, seq_length)`
            src_mask: mask tensor, shape `(batch_size, 1, 1, seq_length)`
            tgt_mask: mask tensor, shape `(batch_size, 1, seq_length, seq_length)`
        Returns
            Tensor with shape `(batch_size, seq_length, d_model)`
        """
        if src_mask is None:
            src_mask = create_encoder_mask(
                encoder_input=src,
                pad_token_id=self.src_pad_token_id,
            )
        encoder_output = self.encode(src=src, src_mask=src_mask)
        decoder_output = self.decode(
            encoder_output=encoder_output,
            tgt=tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
        )
        return decoder_output

    def project(self, x: Tensor) -> Tensor:
        """
        Args
            x: input tensor, shape `(batch_size, seq_length, d_model)`
        Returns
            Tensor with shape `(batch_size, seq_length, vocab_size)`
        """
        return self.projection_layer(x)


def build_transformer(config: TransformerConfig) -> Transformer:
    src_embedding = InputEmbedding(
        d_model=config.d_model,
        vocab_size=config.src_vocab_size,
    )
    tgt_embedding = InputEmbedding(
        d_model=config.d_model,
        vocab_size=config.tgt_vocab_size,
    )

    src_position = PositionalEncoding(
        d_model=config.d_model,
        seq_length=config.src_seq_length,
        dropout=config.dropout,
    )
    tgt_position = PositionalEncoding(
        d_model=config.d_model,
        seq_length=config.tgt_seq_length,
        dropout=config.dropout,
    )

    encoder_blocks: list[EncoderLayer] = []
    for _ in range(config.num_encoder_layers):
        self_attention = MultiHeadAttention(
            d_model=config.d_model,
            num_heads=config.h,
            dropout=config.dropout,
        )
        feed_forward = FeedForward(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )
        encoder_layer = EncoderLayer(
            features=config.d_model,
            self_attention=self_attention,
            feed_forward=feed_forward,
            dropout=config.dropout,
        )
        encoder_blocks.append(encoder_layer)

    decoder_blocks: list[DecoderLayer] = []
    for _ in range(config.num_decoder_layers):
        self_attention = MultiHeadAttention(
            d_model=config.d_model,
            num_heads=config.h,
            dropout=config.dropout,
        )
        cross_attention = MultiHeadAttention(
            d_model=config.d_model,
            num_heads=config.h,
            dropout=config.dropout,
        )
        feed_forward = FeedForward(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )
        decoder_layer = DecoderLayer(
            features=config.d_model,
            self_attention=self_attention,
            cross_attention=cross_attention,
            feed_forward=feed_forward,
            dropout=config.dropout,
        )
        decoder_blocks.append(decoder_layer)

    encoder = Encoder(features=config.d_model, layers=nn.ModuleList(encoder_blocks))
    decoder = Decoder(features=config.d_model, layers=nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(
        d_model=config.d_model,
        vocab_size=config.tgt_vocab_size,
    )

    transformer = Transformer(
        encoder=encoder,
        decoder=decoder,
        src_embedding=src_embedding,
        tgt_embedding=tgt_embedding,
        src_position=src_position,
        tgt_position=tgt_position,
        projection_layer=projection_layer,
        src_pad_token_id=config.src_pad_token_id,
        tgt_pad_token_id=config.tgt_pad_token_id,
    )

    return transformer
