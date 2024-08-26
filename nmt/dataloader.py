from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader, random_split

from .bilingual_dataset import BilingualDataset


def get_bilingual_dataset(
    dataset: Dataset,
    train_size: float,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    lang_src: str,
    lang_tgt: str,
    seq_length: int,
) -> tuple[Dataset, Dataset]:
    # Split the dataset into training (90%) and testing (10%) (default)
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
        train_size=config["train_ratio"],
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        lang_src=config["lang_src"],
        lang_tgt=config["lang_tgt"],
        seq_length=-config["seq_length"],
    )

    # Create the data loaders
    train_data_loader = DataLoader(
        dataset=train_set,
        batch_size=config["batch_size_train"],
        shuffle=True,
    )
    test_data_loader = DataLoader(
        dataset=test_set,
        batch_size=config["batch_size_test"],
        shuffle=True,
    )

    return train_data_loader, test_data_loader
