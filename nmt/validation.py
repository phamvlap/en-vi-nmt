import torch
import torch.nn as nn

from torch import Tensor
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from .constants import SpecialToken
from transformer.models import Transformer
from transformer.functional import create_encoder_mask, create_decoder_mask


def greedy_search_decode(
    model: Transformer,
    source: Tensor,
    tokenizer_tgt: Tensor,
    seq_length: int,
    device: torch.device,
) -> Tensor:
    """
    Args
        model: Transformer model
        source: source tensor `(seq_length, )`
        tokenizer_tgt: target tokenizer
        seq_length: maximum sequence length
        device: torch.device
    Returns
        Tensor with shape `(seq_length, )`
    """
    # Get the <SOS> and <EOS> token ids
    sos_token_id = tokenizer_tgt.token_to_id(SpecialToken.SOS)
    eos_token_id = tokenizer_tgt.token_to_id(SpecialToken.EOS)
    pad_token_id = tokenizer_tgt.token_to_id(SpecialToken.PAD)

    source = source.unsqueeze(0).to(device)

    encoder_mask = create_encoder_mask(
        encoder_input=source,
        pad_token_id=pad_token_id,
    )

    # encoder_output (1, seq_length, d_model)
    encoder_output = model.encode(src=source, src_mask=encoder_mask)

    # Initialize the decoder input with the <SOS> token: decoder_input (1, 1)
    decoder_input = (
        torch.empty(1, 1).fill_(value=sos_token_id).type_as(source).to(device)
    )
    for _ in range(seq_length):
        # Build mask for the decoder input: decoder_mask (1, decoder_input.size(1), decoder_input.size(1))
        decoder_mask = create_decoder_mask(
            decoder_input=decoder_input,
            pad_token_id=pad_token_id,
        )

        # Calculate the output of the decoder: decoder_output (1, _, d_model)
        decoder_output = model.decode(
            tgt=decoder_input,
            encoder_output=encoder_output,
            src_mask=encoder_mask,
            tgt_mask=decoder_mask,
        )

        # Project the decoder output: projection_output (1, 1, tgt_vocab_size)
        projection_output = model.project(x=decoder_output[:, -1, :])

        # Get the token with the highest probability
        next_token = torch.argmax(input=projection_output, dim=-1)

        # Concatenate the next token to the decoder input
        decoder_input = torch.cat(
            tensors=[
                decoder_input,
                torch.empty(1, 1)
                .fill_(value=next_token.item())
                .type_as(source)
                .to(device),
            ],
            dim=1,
        )

        # If the next token is the <EOS> token, stop
        if next_token == eos_token_id:
            break

    result = decoder_input.squeeze(dim=0)
    return result


@torch.no_grad()
def run_validation(
    model: Transformer,
    val_data_loader: DataLoader,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> float:
    model.to(device)
    model.eval()

    sum_loss = 0.0
    count = 0

    for batch in val_data_loader:
        encoder_input = batch["encoder_input"].to(device)  # (batch_size, seq_length)
        decoder_input = batch["decoder_input"].to(device)  # (batch_size, seq_length)
        labels = batch["labels"].to(device)  # (batch_size, seq_length)

        encoder_mask = create_encoder_mask(
            encoder_input=encoder_input,
            pad_token_id=tokenizer_src.token_to_id(SpecialToken.PAD),
        ).to(torch.int64)
        decoder_mask = create_decoder_mask(
            decoder_input=decoder_input,
            pad_token_id=tokenizer_tgt.token_to_id(SpecialToken.PAD),
        ).to(torch.int64)

        decoder_output = model(
            src=encoder_input,
            tgt=decoder_input,
            src_mask=encoder_mask,
            tgt_mask=decoder_mask,
        )
        proj_output = model.project(decoder_output)

        loss = criterion(
            proj_output.view(-1, proj_output.size(-1)),
            labels.view(-1),
        )
        sum_loss += loss.item()
        count += 1

    model.train()

    return sum_loss / count
