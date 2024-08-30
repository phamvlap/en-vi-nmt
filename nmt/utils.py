import torch

from torch import Tensor
from torch.utils.data import Dataset

from nltk.translate import bleu_score
from datasets import load_dataset, Dataset as DatasetModel


# Create a tensor with zeros above the diagonal, ones below and on the diagonal
def causal_mask(size: Tensor) -> Tensor:
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).to(dtype=torch.int64)
    return (mask == 0).int()


def create_encoder_mask(
    encoder_input: Tensor,
    pad_token_id: Tensor,
    device: str = "cpu",
) -> Tensor:
    return (
        (encoder_input != pad_token_id)
        .unsqueeze(dim=0)
        .unsqueeze(dim=0)
        .int()
        .to(device=device)
    )


def create_decoder_mask(
    decoder_input: Tensor,
    pad_token_id: Tensor,
    device: str = "cpu",
) -> Tensor:
    return (
        (decoder_input != pad_token_id).unsqueeze(dim=0).unsqueeze(dim=0).int()
        & causal_mask(size=decoder_input.size(0))
    ).to(device=device)


def load_data(config: dict) -> Dataset:
    if config["data_files"] is not None:
        ds = load_dataset(
            path=config["datasource"],
            data_files=config["data_files"],
            split=config["split_mode"],
        )
    else:
        ds = load_dataset(
            path=config["datasource"],
            split=config["split_mode"],
        )
    out = DatasetModel.from_dict(
        {
            config["lang_src"]: ds["English"],
            config["lang_tgt"]: ds["Vietnamese"],
        }
    )
    return out


def calc_bleu_score(cans: list[str], refs: list[list[str]]) -> list[float]:
    scores = []
    for i in range(1, 5):
        weights = tuple([1 / i for _ in range(i)])
        scores.append(
            bleu_score.corpus_bleu(
                list_of_references=refs,
                hypotheses=cans,
                weights=weights,
            )
        )
    return scores
