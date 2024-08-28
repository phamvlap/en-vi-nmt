import torch
import torch.nn as nn

from torch import Tensor
from torch.utils.data import DataLoader

from tokenizers import Tokenizer

from .utils import causal_mask, calc_bleu_score
from .constants import SpecialToken
from transformer.models.transformer import Transformer


"""
Args:
    model: Transformer model
    source: source tensor (batch_size, seq_length)
    source_mask: source mask tensor (batch_size, 1, 1, seq_length)
    tokenizer_tgt: target tokenizer
    seq_length: maximum sequence length
    device: device
"""


def greedy_search_decode(
    model: Transformer,
    source: Tensor,
    source_mask: Tensor,
    tokenizer_tgt: Tensor,
    seq_length: int,
    device: torch.device,
) -> Tensor:
    # Get the <SOS> and <EOS> token ids
    sos_token_id = tokenizer_tgt.token_to_id(SpecialToken.SOS)
    eos_token_id = tokenizer_tgt.token_to_id(SpecialToken.EOS)

    # encoder_output (batch_size, seq_length, d_model)
    encoder_output = model.encode(src=source, src_mask=source_mask)

    # Initialize the decoder input with the <SOS> token: decoder_input (1, 1)
    decoder_input = (
        torch.empty(1, 1).fill_(value=sos_token_id).type_as(source).to(device)
    )
    for _ in range(seq_length):
        # Build mask for the decoder input: decoder_mask (1, decoder_input.size(1), decoder_input.size(1))
        decoder_mask = (
            causal_mask(size=decoder_input.size(1)).type_as(source_mask).to(device)
        )

        # Calculate the output of the decoder: decoder_output (batch_size, _, d_model)
        decoder_output = model.decode(
            encoder_output=encoder_output,
            src_mask=source_mask,
            tgt=decoder_input,
            tgt_mask=decoder_mask,
        )

        # Project the decoder output: projection_output (batch_size, 1, tgt_vocab_size)
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

    return decoder_input.squeeze(dim=0)


def run_validation(
    model: nn.Module,
    val_data_loader: DataLoader,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    seq_length: int,
    device: torch.device,
    num_examples: int = 5,
) -> None:
    logging_interval = len(val_data_loader) // num_examples

    with torch.no_grad():
        count = 0

        source_texts = []
        expected = []
        predicted = []

        for batch in val_data_loader:
            count += 1

            encoder_input = batch["encoder_input"].to(
                device
            )  # (batch_size, seq_length)
            encoder_mask = batch["encoder_mask"].to(
                device
            )  # (batch_size, 1, 1, seq_length)
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]

            # Get the output from the model
            model_output = greedy_search_decode(
                model=model,
                source=encoder_input,
                source_mask=encoder_mask,
                tokenizer_tgt=tokenizer_tgt,
                seq_length=seq_length,
                device=device,
            )
            model_output_text = tokenizer_tgt.decode(
                model_output.detach().cpu().numpy()
            )

            target_tokens = tokenizer_tgt.encode(sequence=target_text).tokens
            predicted_tokens = tokenizer_tgt.encode(sequence=model_output_text).tokens

            # Append the texts
            source_texts.append(source_text)
            expected.append([target_tokens])
            predicted.append(predicted_tokens)

            if count % logging_interval == 0:
                # Print the source, target and predicted text to console
                print("SOURCE: {}".format(source_text))
                print("TARGET: {}".format(target_text))
                print("PREDICTED: {}".format(model_output_text))
                print("TARGET TOKENS: {}".format(target_tokens))
                print("PREDICTED TOKENS: {}".format(predicted_tokens))
                # BLEU Score
                bleu_scores = calc_bleu_score(
                    cans=[predicted_tokens], refs=[[target_tokens]]
                )
                print("BLEU SCORE OF PREDICTION {}TH SENTENCE".format(count))
                for i in range(len(bleu_scores)):
                    print("BLEU-{0}: {1:.4f}".format(i + 1, bleu_scores[i]))

        bleu_score_corpus = calc_bleu_score(cans=predicted, refs=expected)
        return bleu_score_corpus
