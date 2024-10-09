import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from tqdm import tqdm

from .preprocess import load_tokenizer
from .validation import run_validation
from .utils import load_data
from .dataloader import get_dataloader
from .constants import SpecialToken
from transformer.models import Transformer, build_transformer
from config.config import get_weights_file_path


def get_model(
    config: dict,
    vocab_src_length: int,
    vocab_tgt_length: int,
) -> Transformer:
    model = build_transformer(
        src_vocab_size=vocab_src_length,
        tgt_vocab_size=vocab_tgt_length,
        src_seq_length=config["seq_length"],
        tgt_seq_length=config["seq_length"],
        d_model=config["d_model"],
    )
    return model


def save_model(
    model_filename: str,
    epoch: int,
    model: nn.Module,
    optimizer,
    global_step: int,
):
    torch.save(
        obj={
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
        },
        f=model_filename,
    )


def train_model(config: dict) -> None:
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))
    if device.type == "cuda":
        print("Device name: {}".format(torch.cuda.get_device_name(device)))
    else:
        print("Device name: cpu")

    # Create the folder to save model weights
    Path(f"{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    # Load the dataset
    dataset = load_data(config=config)

    # Get the tokenizers
    tokenizer_src, tokenizer_tgt = load_tokenizer(dataset=dataset, config=config)

    # Get the data loaders
    train_data_loader, test_data_loader = get_dataloader(
        dataset=dataset,
        config=config,
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt,
    )

    # Create model
    model = get_model(
        config=config,
        vocab_src_length=tokenizer_src.get_vocab_size(),
        vocab_tgt_length=tokenizer_tgt.get_vocab_size(),
    ).to(device)

    # Optimizer ?? eps
    optimizer = optim.Adam(params=model.parameters(), lr=config["lr"], eps=1e-9)

    # Preload model
    initial_epoch = 0
    global_step = 0
    if config["preload"] is not None:
        model_filename = get_weights_file_path(config=config, epoch=config["preload"])
        print('Preloading model from "{0}"'.format(model_filename))
        state = torch.load(f=model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state_dict=state["optimizer_state_dict"])
        global_step = state["global_step"]
    else:
        print("Training model from scratch ...")

    # Loss function
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id(SpecialToken.PAD),
        label_smoothing=0.1,  # ??
    ).to(device)

    # Training loop
    print("Starting training ...")
    for epoch in range(initial_epoch, config["num_epochs"]):
        # Clear the cache
        torch.cuda.empty_cache()
        # Set the model to training mode
        model.train()
        # Create a progress bar
        batch_iterator = tqdm(
            iterable=train_data_loader,
            desc="Processing epoch {0}".format(epoch),
        )

        train_loss = 0.0

        for batch in batch_iterator:
            # encoder_input(batch_size, seq_length)
            encoder_input = batch["encoder_input"].to(device)
            # decoder_input(batch_size, seq_length)
            decoder_input = batch["decoder_input"].to(device)
            # encoder_mask(batch_size, 1, 1, seq_length)
            encoder_mask = batch["encoder_mask"].to(device)
            # decoder_mask(batch_size, 1, seq_length, seq_length)
            decoder_mask = batch["decoder_mask"].to(device)
            # label(batch_size, seq_length)
            label = batch["label"].to(device)

            # Pass the inputs through the transformer model
            encoder_output = model.encode(
                src=encoder_input,
                src_mask=encoder_mask,
            )
            decoder_output = model.decode(
                encoder_output=encoder_output,
                src_mask=encoder_mask,
                tgt=decoder_input,
                tgt_mask=decoder_mask,
            )

            # Project the decoder output to the vocab size
            # projection_output(batch_size, seq_length, tgt_vocab_size)
            projection_output = model.project(x=decoder_output)

            # Calculate and show the loss
            # (batch_size, seq_length, tgt_vocab_size) -> (batch_size * seq_length, tgt_vocab_size)
            loss = loss_fn(
                projection_output.view(-1, tokenizer_tgt.get_vocab_size()),
                label.view(-1),  # (batch_size * seq_length)
            )
            train_loss += loss.item()
            batch_iterator.set_postfix(
                {
                    "loss": "{:.4f}".format(loss.item()),
                }
            )

            # Backpropagation the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        with torch.no_grad():
            # Run validation
            bleu_scores_corpus = run_validation(
                model=model,
                val_data_loader=test_data_loader,
                tokenizer_src=tokenizer_src,
                tokenizer_tgt=tokenizer_tgt,
                seq_length=config["seq_length"],
                device=device,
            )
            print("BLEU SCORE OF PREDICTION CORPUS")
            for i in range(len(bleu_scores_corpus)):
                print("BLEU-{0}: {1:.4f}".format(i + 1, bleu_scores_corpus[i]))

        print("Mean train loss: {:.4f}".format(train_loss / len(train_data_loader)))

        if epoch == config["num_epochs"] - 1:
            # Save the model weights
            model_filename = get_weights_file_path(
                config=config,
                epoch="{:03d}".format(epoch),
            )
            save_model(
                model_filename=model_filename,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                global_step=global_step,
            )
