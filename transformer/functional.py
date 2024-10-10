import torch

from torch import Tensor


# Create a tensor with zeros above the diagonal, ones below and on the diagonal
def create_causal_mask(size: int) -> Tensor:
    """
    Args
        size: length of the sequence
    Returns
        Tensor with shape `(1, size, size)`
    """
    tril_mask = torch.tril(torch.ones(1, size, size), diagonal=0).int()
    return tril_mask


def create_encoder_mask(encoder_input: Tensor, pad_token_id: int) -> Tensor:
    """
    Args
        encoder_input: input tensor of encoder, shape `(batch_size, seq_length)`
        pad_token_id: token id of padding token
    Returns
        Tensor with shape `(batch_size, 1, 1, seq_length)`
    """
    encoder_mask = encoder_input != pad_token_id
    encoder_mask = encoder_mask.unsqueeze(1).unsqueeze(2).int().to(encoder_input.device)
    return encoder_mask


def create_decoder_mask(decoder_input: Tensor, pad_token_id: int) -> Tensor:
    """
    Args
        decoder_input: input tensor of decoder, shape `(batch_size, seq_length)`
        pad_token_id: token id of padding token
    Returns
        Tensor with shape `(batch_size, 1, seq_length, seq_length)`
    """
    causal_mask = create_causal_mask(
        size=decoder_input.size(-1),
    ).to(decoder_input.device)

    decoder_mask = decoder_input != pad_token_id
    decoder_mask = (decoder_mask.unsqueeze(1).unsqueeze(1).int() & causal_mask).to(
        decoder_input.device
    )

    return decoder_mask
