import os
import shutil
import pandas as pd

from pathlib import Path
from typing import Generator
from torch.utils.data import Dataset as DatasetType
from tokenizers import Tokenizer
from tokenizers.models import WordLevel, BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import Trainer, WordLevelTrainer, BpeTrainer
from datasets import Dataset


from .constants import SpecialToken, TokenizerModel


# Return a generator that yields all the sentences in the dataset (iterator)
def get_iterator(dataset: DatasetType, lang: str) -> Generator[str, None, None]:
    for item in dataset:
        yield item[lang]


def create_tokenizer_trainer(
    tokenizer_type: str,
    min_freq: int = 2,
) -> tuple[Tokenizer, Trainer]:
    tokenizer_type = tokenizer_type.strip().lower()

    all_special_tokens = [
        SpecialToken.UNK,
        SpecialToken.PAD,
        SpecialToken.SOS,
        SpecialToken.EOS,
    ]

    if tokenizer_type == TokenizerModel.WORD_LEVEL:
        tokenizer = Tokenizer(model=WordLevel(unk_token=SpecialToken.UNK))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(
            special_tokens=all_special_tokens,
            min_frequency=min_freq,
        )
    elif tokenizer_type == TokenizerModel.BPE:
        tokenizer = Tokenizer(model=BPE(unk_token=SpecialToken.UNK))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            special_tokens=all_special_tokens,
            min_frequency=min_freq,
        )
    else:
        raise ValueError("Unsupported tokenizer model: {}".format(tokenizer_type))

    return tokenizer, trainer


def tokenize(
    dataset: DatasetType,
    config: dict,
    lang: str,
    min_freq: int = 2,
) -> Tokenizer:
    tokenizer_file = config["tokenizer_file"].format(lang)
    tokenizer_path = Path(f"{config['tokenizer_dir']}/{tokenizer_file}")

    tokenizer, trainer = create_tokenizer_trainer(
        tokenizer_type=config["tokenizer_type"],
        min_freq=min_freq,
    )

    tokenizer.train_from_iterator(
        iterator=get_iterator(dataset=dataset, lang=lang),
        trainer=trainer,
    )
    tokenizer.save(path=str(tokenizer_path))

    return tokenizer


def load_tokenizer(config: dict) -> tuple[Tokenizer, Tokenizer]:
    tokenizer_dir = Path(config["tokenizer_dir"])
    src_tokenizer_file = config["tokenizer_file"].format(config["lang_src"])
    tgt_tokenizer_file = config["tokenizer_file"].format(config["lang_tgt"])

    if (
        os.path.exists(tokenizer_dir)
        and os.path.isdir(tokenizer_dir)
        and len(os.listdir(tokenizer_dir)) >= 2
    ):
        tokenizer_src = Tokenizer.from_file(f"{tokenizer_dir}/{src_tokenizer_file}")
        tokenizer_tgt = Tokenizer.from_file(f"{tokenizer_dir}/{tgt_tokenizer_file}")
    else:
        raise ValueError("Tokenizers not found.")

    return tokenizer_src, tokenizer_tgt


def train_tokenizer(config: dict) -> None:
    if config["train_data_file"] is None:
        raise ValueError("Train data file not found.")

    df = pd.read_csv(config["train_data_file"])
    dataset = Dataset.from_pandas(df)

    tokenizer_dir = Path(config["tokenizer_dir"])
    if tokenizer_dir.exists():
        shutil.rmtree(tokenizer_dir)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    src_tokenizer = tokenize(
        dataset=dataset,
        config=config,
        lang=config["lang_src"],
        min_freq=config["min_freq"],
    )
    tgt_tokenizer = tokenize(
        dataset=dataset,
        config=config,
        lang=config["lang_tgt"],
        min_freq=config["min_freq"],
    )

    print(f"Tokenizers saved to {tokenizer_dir}")
    print(f"Tokenizer type: {config['tokenizer_type']}")
    print(f"Source tokenizer vocab size: {src_tokenizer.get_vocab_size()}")
    print(f"Target tokenizer vocab size: {tgt_tokenizer.get_vocab_size()}")
