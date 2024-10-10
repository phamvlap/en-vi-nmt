import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path

from transformer.models import build_transformer
from .constants import SpecialToken
from .tokenizer import load_tokenizer
from .dataset import get_dataloader
from .validation import run_validation
from .bleu import compute_bleu
from .utils import set_seed, get_list_weights_file_paths


def test_model(config: dict) -> None:
    print("Testing model...")
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading tokenizers...")
    src_tokenizer, tgt_tokenizer = load_tokenizer(config=config)

    print("Making dataloaders...")
    test_dataloader = get_dataloader(
        tokenizer_src=src_tokenizer,
        tokenizer_tgt=tgt_tokenizer,
        config=config,
        split="test",
        batch_size=config["batch_size_test"],
    )

    list_filepaths = get_list_weights_file_paths(config=config)
    if list_filepaths is not None:
        filepath = list_filepaths[-1]
        checkpoint_states = torch.load(filepath, weights_only=True, map_location=device)

        required_keys = [
            "model_state_dict",
            "model_config",
        ]
        for key in required_keys:
            if key not in checkpoint_states:
                raise ValueError(f"Key {key} not found in checkpoint at {filepath}.")
    else:
        raise ValueError("No model weights found.")

    print("Building Transformer model...")
    model_config = checkpoint_states["model_config"]
    model = build_transformer(config=model_config)
    model.load_state_dict(checkpoint_states["model_state_dict"])
    model.to(device)

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=SpecialToken.PAD,
        label_smoothing=config["label_smoothing"],
    )

    print("Evaluating model...")
    test_loss = run_validation(
        model=model,
        val_data_loader=test_dataloader,
        tokenizer_src=src_tokenizer,
        tokenizer_tgt=tgt_tokenizer,
        criterion=loss_fn,
        device=device,
    )

    print("Computing BLEU scores...")
    bleu_scores = compute_bleu(
        model=model,
        dataset=test_dataloader.dataset,
        tgt_tokenizer=tgt_tokenizer,
        seq_length=config["seq_length"],
        log_examples=config["log_examples"],
        logging_every_n_steps=config["logging_every_n_steps"],
    )
    bleu_dict = {f"test/bleu_{i+1}": bleu_scores[i] for i in range(len(bleu_scores))}

    columns = ["test/loss"] + list(bleu_dict.keys())
    data = [test_loss] + list(bleu_dict.values())
    df = pd.DataFrame(data, columns=columns)

    output_filepath = config["output_filepath"]
    dirpath = Path(output_filepath).parent
    dirpath.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_filepath, index=False)
    print(f"Test results saved to {output_filepath}")
    print("TEST RESULTS:")
    print(df)
