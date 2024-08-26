import torch
import torch.nn as nn

from transformer.models.encoder import Encoder, EncoderLayer
from transformer.models.decoder import Decoder, DecoderLayer
from transformer.embedding.input_embedding import InputEmbedding
from transformer.embedding.positional_encoding import PositionalEncoding
from transformer.layers.projection_layer import ProjectionLayer
from transformer.layers.multi_head_attention import MultiHeadAttention
from transformer.layers.feed_forward import FeedForward


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
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_position = src_position
        self.tgt_position = tgt_position
        self.projection_layer = projection_layer

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        src = self.src_embedding(src)
        src = self.src_position(src)
        return self.encoder(x=src, mask=src_mask)

    def decode(
        self,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_position(tgt)
        return self.decoder(
            x=tgt,
            encoder_output=encoder_output,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
        )

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_layer(x)


def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_length: int,
    tgt_seq_length: int,
    d_model: int = 512,
    h: int = 8,  # number of heads
    num_encoders: int = 6,
    num_decoders: int = 6,
    dropout: float = 0.1,
    d_ff: int = 2048,  # dimension of feed forward
) -> Transformer:
    src_embedding = InputEmbedding(d_model=d_model, vocab_size=src_vocab_size)
    tgt_embedding = InputEmbedding(d_model=d_model, vocab_size=tgt_vocab_size)

    src_position = PositionalEncoding(
        d_model=d_model,
        seq_length=src_seq_length,
        dropout=dropout,
    )
    tgt_position = PositionalEncoding(
        d_model=d_model,
        seq_length=tgt_seq_length,
        dropout=dropout,
    )

    encoder_blocks: list[EncoderLayer] = []
    for _ in range(num_encoders):
        self_attention = MultiHeadAttention(d_model=d_model, h=h, dropout=dropout)
        feed_forward = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        encoder_layer = EncoderLayer(
            features=d_model,
            self_attention=self_attention,
            feed_forward=feed_forward,
            dropout=dropout,
        )
        encoder_blocks.append(encoder_layer)

    decoder_blocks: list[DecoderLayer] = []
    for _ in range(num_decoders):
        self_attention = MultiHeadAttention(d_model=d_model, h=h, dropout=dropout)
        cross_attention = MultiHeadAttention(d_model=d_model, h=h, dropout=dropout)
        feed_forward = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        decoder_layer = DecoderLayer(
            features=d_model,
            self_attention=self_attention,
            cross_attention=cross_attention,
            feed_forward=feed_forward,
            dropout=dropout,
        )
        decoder_blocks.append(decoder_layer)

    encoder = Encoder(layers=nn.ModuleList(encoder_blocks))
    decoder = Decoder(layers=nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model=d_model, vocab_size=tgt_vocab_size)

    transformer = Transformer(
        encoder=encoder,
        decoder=decoder,
        src_embedding=src_embedding,
        tgt_embedding=tgt_embedding,
        src_position=src_position,
        tgt_position=tgt_position,
        projection_layer=projection_layer,
    )

    return transformer
