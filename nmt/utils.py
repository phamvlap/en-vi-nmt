import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset
from pathlib import Path

from nltk.translate import bleu_score
from datasets import load_dataset, Dataset as DatasetModel

ADAM = "adam"
ADAMW = "adamw"
NOAM_DECAY = "noam"


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


def calc_bleu_score(cands: list[str], refs: list[list[str]]) -> list[float]:
    scores = []
    for i in range(1, 5):
        weights = tuple([1 / i for _ in range(i)])
        scores.append(
            bleu_score.corpus_bleu(
                list_of_references=refs,
                hypotheses=cands,
                weights=weights,
            )
        )
    return scores


def make_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    optimizer_type = config["optimizer"].strip().lower()

    if optimizer_type == ADAM:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["lr"],
            betas=config["betas"],
            eps=config["eps"],
            weight_decay=config["weight_decay"],
        )
    elif optimizer_type == ADAMW:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            betas=config["betas"],
            eps=config["eps"],
            weight_decay=config["weight_decay"],
        )
    else:
        raise ValueError(
            f"Unknown optimizer type {optimizer_type}. Supported types are: {ADAM}, {ADAMW}"
        )
    return optimizer


def noam_decay(
    model_size: int,
    step: int,
    warmup_steps: int,
    factor: float = 1.0,
) -> float:
    step = max(1, step)
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
    )


def get_lr_scheduler(
    optimizer: optim.Optimizer,
    config: dict,
) -> optim.lr_scheduler.LRScheduler:
    lr_scheduler_type = config["lr_scheduler"].strip().lower()
    lr_scheduler = None

    if lr_scheduler_type == NOAM_DECAY:
        lr_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: noam_decay(
                model_size=config["d_model"],
                step=step,
                warmup_steps=config["warmup_steps"],
            ),
        )

    return lr_scheduler


def get_weights_file_path(model_folder: str, model_basename: str, epoch: str) -> str:
    return f"{model_folder}/{model_basename}{epoch}.pt"


def get_list_weights_file_paths(config: dict) -> list[str] | None:
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    weight_filepaths = list(Path(model_folder).glob(f"{model_basename}*.pt"))
    if len(weight_filepaths) == 0:
        return None
    weight_filepaths = [str(file) for file in sorted(weight_filepaths)]
    return weight_filepaths


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
