import pandas as pd

from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datasets import load_dataset
from typing import Literal

from .bilingual_dataset import BilingualDataset


def get_dataloader(
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    config: dict,
    split: Literal["train", "val", "test"],
    batch_size: int,
) -> DataLoader:
    if split not in ["train", "val", "test"]:
        raise ValueError(f"split must be one of (train, val, test), got {split}")

    df = pd.reac_csv(config[f"{split}_data_file"])
    dataset = Dataset.from_pandas(df)

    bilingual_dataset = BilingualDataset(
        dataset=dataset,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
        lang_src=config["lang_src"],
        lang_tgt=config["lang_tgt"],
        seq_length=config["seq_length"],
    )

    # Create the data loaders
    dataloader = DataLoader(
        dataset=bilingual_dataset,
        batch_size=batch_size,
        shuffle=config["shuffle_dataloader"],
    )

    return dataloader


def preprocess(config: dict) -> None:
    print("Preprocessing data...")
    if config["datasource"] is None:
        raise ValueError("Data source not provided")
    dataset = load_dataset(config["datasource"])

    df_list: list[pd.DataFrame] = []
    for split in dataset.keys():
        df_list.append(pd.DataFrame(dataset[split]))
    df = pd.concat(df_list)

    if config["shuffle"]:
        df = df.sample(frac=1).reset_index(drop=True)

    if config["is_sampling"]:
        if (
            config["num_samples"] is not None
            and config["num_samples"] > 0
            and config["num_samples"] < len(df)
        ):
            df = df.sample(n=config["num_samples"]).reset_index(drop=True)

    train_size = config["train_size"]
    val_size = config["val_size"]
    test_size = config["test_size"]

    if train_size + val_size + test_size != 1.0:
        raise ValueError("train_size + val_size + test_size != 1")

    length = len(df)
    train_end = int(train_size * length)
    val_end = train_end + int(val_size * length)

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    data_dir = Path(config["train_data_file"]).parent
    data_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(config["train_data_file"], index=False)
    val_df.to_csv(config["val_data_file"], index=False)
    test_df.to_csv(config["test_data_file"], index=False)

    print("Data preprocessed")
    print(f"Data files saved at {data_dir}")
    print(f"Length of dataset: {length}")
    print(f"Length of train dataset: {len(train_df)}")
    print(f"Length of val dataset: {len(val_df)}")
    print(f"Length of test dataset: {len(test_df)}")
