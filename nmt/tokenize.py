from pathlib import Path
from torch.utils.data import Dataset

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from nmt.constants import SpecialToken


# Return a generator that yields all the sentences in the dataset (iterator)
def get_all_sentences(dataset: Dataset, language: str):
    for item in dataset:
        yield item[language]


def build_tokenizer(config: dict, dataset: Dataset, language: str) -> Tokenizer:
    tokenizer_path = Path(config["tokenizer_file"].format(language))

    if not Path.exists(tokenizer_path):
        # Apply the WordLevel model to the tokenizer
        tokenizer = Tokenizer(model=WordLevel(unk_token=SpecialToken.UNK))
        # Split the text by whitespace
        tokenizer.pre_tokenizer = Whitespace()
        # Create a trainer
        trainer = WordLevelTrainer(
            special_tokens=[
                SpecialToken.UNK,
                SpecialToken.PAD,
                SpecialToken.SOS,
                SpecialToken.EOS,
            ],
            min_frequency=2,
        )
        # Train the tokenizer
        tokenizer.train_from_iterator(
            iterator=get_all_sentences(dataset=dataset, language=language),
            trainer=trainer,
        )
        # Save the tokenizer
        tokenizer.save(path=str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(path=str(tokenizer_path))

    return tokenizer


def get_tokenizer(dataset: Dataset, config: dict) -> tuple[Tokenizer, Tokenizer]:
    tokenizer_src = build_tokenizer(
        dataset=dataset,
        config=config,
        language=config["lang_src"],
    )
    tokenizer_tgt = build_tokenizer(
        dataset=dataset,
        config=config,
        language=config["lang_tgt"],
    )
    return tokenizer_src, tokenizer_tgt
