import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tokenizers import Tokenizer

from .utils import causal_mask, calc_bleu_score
from .constants import SpecialToken


def greedy_search_decode(
    model: nn.Module,
    source: torch.Tensor,
    source_mask: torch.Tensor | None,
    tokenizer_src: torch.Tensor,
    tokenizer_tgt: torch.Tensor,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    # Get the [SOS] and [EOS] token indexes
    sos_index = tokenizer_tgt.token_to_id(SpecialToken.SOS)
    eos_index = tokenizer_tgt.token_to_id(SpecialToken.EOS)

    # Precompute the encoder output and reuse it for every token we get from the decoder
    # encoder_output (batch_size, seq_length, d_model)
    encoder_output = model.encode(src=source, src_mask=source_mask)
    # Initialize the decoder input with the [SOS] token
    # decoder_input (1, 1)
    decoder_input = torch.empty(1, 1).fill_(value=sos_index).type_as(source).to(device)
    while True:
        if decoder_input.size(1) >= max_length:
            break

        # Build mask for the decoder input (target)
        decoder_mask = (
            causal_mask(size=decoder_input.size(1)).type_as(source_mask).to(device)
        )

        # Calculate the output of the decoder
        decoder_output = model.decode(
            encoder_output=encoder_output,
            src_mask=source_mask,
            tgt=decoder_input,
            tgt_mask=decoder_mask,
        )

        # Project the decoder output
        # decoder_output[:, -1] (1, d_model): get the last token (last column)
        projection_output = model.project(x=decoder_output[:, -1])

        # Get the token with the highest probability
        next_value, next_index = torch.max(input=projection_output, dim=1)
        next_token = (
            torch.empty(1, 1).fill_(value=next_index.item()).type_as(source).to(device)
        )

        # Append the next token to the decoder input
        decoder_input = torch.cat(tensors=[decoder_input, next_token], dim=1)

        # If the next token is the [EOS] token, stop
        if next_index.item() == eos_index:
            break

    return decoder_input.squeeze(dim=0)


def run_validation(
    model: nn.Module,
    val_data_loader: DataLoader,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    max_length: int,
    device: torch.device,
    num_examples: int = 5,
) -> None:
    print_step = len(val_data_loader) // num_examples

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
                tokenizer_src=tokenizer_src,
                tokenizer_tgt=tokenizer_tgt,
                max_length=max_length,
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

            if count % print_step == 0:
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
                print("BLEU SCORE OF PREDICTION: {}".format(count))
                for i in range(len(bleu_scores)):
                    print("BLEU-{0}: {1:.4f}".format(i + 1, bleu_scores[i]))

        bleu_score_corpus = calc_bleu_score(cans=predicted, refs=expected)
        return bleu_score_corpus
