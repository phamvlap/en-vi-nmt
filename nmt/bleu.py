import torch

from tokenizers import Tokenizer

from transformer.models import Transformer
from .bilingual_dataset import BilingualDataset
from .validation import greedy_search_decode
from .utils import calc_bleu_score


@torch.no_grad()
def compute_bleu(
    model: Transformer,
    dataset: BilingualDataset,
    tgt_tokenizer: Tokenizer,
    seq_length: int,
    log_examples: bool = False,
    logging_every_n_steps: int = 1000,
) -> list[float]:
    device = model.device
    model.eval()

    pred_text_list = []
    target_text_list = []

    for idx, data in enumerate(dataset):
        encoder_input = data["encoder_input"]
        labels = data["labels"]

        pred_tokens = greedy_search_decode(
            model=model,
            source=encoder_input,
            tokenizer_tgt=tgt_tokenizer,
            seq_length=seq_length,
            device=device,
        )

        src_tokens = encoder_input.detach().cpu().numpy()
        tgt_tokens = labels.detach().cpu().numpy()
        pred_tokens = pred_tokens.detach().cpu().numpy()

        src_text = tgt_tokenizer.decode(src_tokens, skip_special_tokens=True)
        tgt_text = tgt_tokenizer.decode(tgt_tokens, skip_special_tokens=True)
        pred_text = tgt_tokenizer.decode(pred_tokens, skip_special_tokens=True)

        pred_text_list.append(pred_text)
        target_text_list.append([tgt_text])

        if log_examples and idx % logging_every_n_steps == 0:
            print(f"EXAMPLE {idx}-th")
            print(f"SOURCE TEXT: {src_text}")
            print(f"TARGET TEXT: {tgt_text}")
            print(f"PREDICTED TEXT: {pred_text}\n")

    model.train()

    bleu_score = calc_bleu_score(
        cands=pred_text_list,
        refs=target_text_list,
    )

    return bleu_score
