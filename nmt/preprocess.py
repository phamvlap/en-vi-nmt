import os
from pathlib import Path
from typing import Generator
from torch.utils.data import Dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel, BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer, Trainer

from .constants import SpecialToken, TokenizerModel


# Return a generator that yields all the sentences in the dataset (iterator)
def get_iterator(dataset: Dataset, language: str) -> Generator[str, None, None]:
    for item in dataset:
        yield item[language]


def create_tokenizer_trainer(
    tokenizer_model: str,
    min_freq: int = 2,
) -> tuple[Tokenizer, Trainer]:
    tokenizer_model = tokenizer_model.lower()

    all_special_tokens = [
        SpecialToken.UNK,
        SpecialToken.PAD,
        SpecialToken.SOS,
        SpecialToken.EOS,
    ]

    if tokenizer_model == TokenizerModel.WORD_LEVEL:
        tokenizer = Tokenizer(model=WordLevel(unk_token=SpecialToken.UNK))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(
            special_tokens=all_special_tokens,
            min_frequency=min_freq,
        )
    elif tokenizer_model == TokenizerModel.BPE:
        tokenizer = Tokenizer(model=BPE(unk_token=SpecialToken.UNK))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(
            special_tokens=all_special_tokens,
            min_frequency=min_freq,
        )
    else:
        raise ValueError("Unsupported tokenizer model: {}".format(tokenizer_model))

    return tokenizer, trainer


def tokenize(dataset: Dataset, config: dict, lang: str, min_freq: int = 2) -> Tokenizer:
    tokenizer_path = Path(
        f'{config['tokenizer_dir']}/{config["tokenizer_file"].format(lang)}'
    )

    tokenizer, trainer = create_tokenizer_trainer(
        tokenizer_model=config["tokenizer_model"],
        min_freq=min_freq,
    )

    tokenizer.train_from_iterator(
        iterator=get_iterator(dataset=dataset, language=lang),
        trainer=trainer,
    )
    tokenizer.save(path=str(tokenizer_path))

    return tokenizer


def load_tokenizer(dataset: Dataset, config: dict) -> tuple[Tokenizer, Tokenizer]:
    tokenizer_dir = Path(config["tokenizer_dir"])

    if (
        os.path.exists(tokenizer_dir)
        and os.path.isdir(tokenizer_dir)
        and len(os.listdir(tokenizer_dir)) >= 2
    ):
        tokenizer_src = Tokenizer.from_file(
            Path(
                f'{tokenizer_dir}/{config["tokenizer_file"].format(config["lang_src"])}'
            )
        )
        tokenizer_tgt = Tokenizer.from_file(
            Path(
                f'{tokenizer_dir}/{config["tokenizer_file"].format(config["lang_tgt"])}'
            )
        )
    else:
        Path(tokenizer_dir).mkdir(parents=True, exist_ok=True)
        tokenizer_src = tokenize(
            dataset=dataset,
            config=config,
            lang=config["lang_src"],
        )
        tokenizer_tgt = tokenize(
            dataset=dataset,
            config=config,
            lang=config["lang_tgt"],
        )
    return tokenizer_src, tokenizer_tgt
