import torch
from torch.utils.data import Dataset, DataLoader, random_split

from nltk.translate import bleu_score
from datasets import load_dataset
from tokenizers import Tokenizer

from nmt.bilingual_dataset import BilingualDataset


# Create a tensor with zeros above the diagonal, ones below and on the diagonal
def causal_mask(size: torch.Tensor) -> torch.Tensor:
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).to(dtype=torch.int64)
    return (mask == 0).int()


def create_encoder_mask(
    encoder_input: torch.Tensor,
    pad_token_id: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    return (
        (encoder_input != pad_token_id)
        .unsqueeze(dim=0)
        .unsqueeze(dim=0)
        .int()
        .to(device=device)
    )


def create_decoder_mask(
    decoder_input: torch.Tensor,
    pad_token_id: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    return (
        (decoder_input != pad_token_id).unsqueeze(dim=0).unsueeze(dim=0).int()
        & causal_mask(size=decoder_input.size(0))
    ).to(device=device)


def load_data(config: dict, **kwargs) -> Dataset:
    ds = load_dataset(path=config["datasource"], **kwargs)
    out = Dataset.from_dict(
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


def get_bilingual_dataset(
    dataset: Dataset,
    train_size: float,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    lang_src: str,
    lang_tgt: str,
    seq_length: int,
) -> tuple[Dataset, Dataset]:
    # Split the dataset into training (80%) and testing (20%) (default)
    train_dataset_size = int(train_size * len(dataset))
    test_dataset_size = len(dataset) - train_dataset_size

    # Split the dataset
    train_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=[train_dataset_size, test_dataset_size],
    )

    # Create the BilingualDataset
    train_dataset = BilingualDataset(
        dataset=train_dataset,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        lang_src=lang_src,
        lang_tgt=lang_tgt,
        seq_length=seq_length,
    )
    test_dataset = BilingualDataset(
        dataset=test_dataset,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        lang_src=lang_src,
        lang_tgt=lang_tgt,
        seq_length=seq_length,
    )

    return train_dataset, test_dataset


def get_dataloader(
    dataset: Dataset,
    config: dict,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
) -> tuple[DataLoader, DataLoader]:
    train_set, test_set = get_bilingual_dataset(
        dataset=dataset,
        train_size=config["train_size"],
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        lang_src=config["lang_src"],
        lang_tgt=config["lang_tgt"],
        seq_length=-config["seq_length"],
    )

    # Create the data loaders
    train_data_loader = DataLoader(
        dataset=train_set,
        batch_size=config["batch_size"],
        shuffle=True,
    )
    test_data_loader = DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=True,
    )

    return train_data_loader, test_data_loader
