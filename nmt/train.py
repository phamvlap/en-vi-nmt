import torch
import torch.nn as nn

from pathlib import Path

from transformer.models import TransformerConfig, build_transformer
from .preprocess import load_tokenizer
from .utils import load_data, get_list_weights_file_paths, get_weights_file_path
from .dataloader import get_dataloader
from .constants import SpecialToken
from .trainer import TrainerArguments, Trainer
from .utils import make_optimizer, get_lr_scheduler, set_seed


def train_model(config: dict) -> None:
    set_seed(config["seed"])

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

    # Preload model
    initial_epoch = 0
    global_step = 0
    scaler_state_dict = None
    filepath = None

    if config["preload"] == "latest":
        list_filepaths = get_list_weights_file_paths(config=config)
        if list_filepaths is not None:
            filepath = list_filepaths[-1]
    elif config["preload"] is not None and isinstance(config["preload"], int):
        filepath = get_weights_file_path(
            model_folder=config["model_folder"],
            model_basename=config["model_basename"],
            epoch=config["preload"],
        )

    if filepath is None:
        print("Training model from scratch...")
        # Build model
        model_config = TransformerConfig(
            src_vocab_size=tokenizer_src.get_vocab_size(),
            tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
            src_seq_length=config["src_seq_length"],
            tgt_seq_length=config["tgt_seq_length"],
            d_model=config["d_model"],
            h=config["num_heads"],
            num_encoder_layers=config["num_encoder_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            dropout=config["dropout"],
            d_ff=config["d_ff"],
            src_pad_token_id=tokenizer_src.token_to_id(SpecialToken.PAD),
            tgt_pad_token_id=tokenizer_tgt.token_to_id(SpecialToken.PAD),
        )
        model = build_transformer(model_config)
        model.to(device)
    else:
        print(f"Loading model from {filepath}...")

        checkpoint = torch.load(filepath, map_location=device)

        required_keys = [
            "model_state_dict",
            "model_config",
            "optimizer_state_dict",
        ]
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(f"Key {key} not found in checkpoint.")

        model_config = checkpoint["model_config"]
        model = build_transformer(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        if "epoch" in checkpoint:
            initial_epoch = checkpoint["epoch"] + 1
        if "global_step" in checkpoint:
            global_step = checkpoint["global_step"]
        if "scaler_state_dict" in checkpoint:
            scaler_state_dict = checkpoint["scaler_state_dict"]

    # Optimizer
    optimizer = make_optimizer(model=model, config=config)

    # Learning rate scheduler
    lr_scheduler = get_lr_scheduler(optimizer=optimizer, config=config)

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    # Loss function
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_tgt.token_to_id(SpecialToken.PAD),
        label_smoothing=0.1,
    ).to(device)

    trainer_args = TrainerArguments(
        seq_length=config["seq_length"],
        initial_epoch=initial_epoch,
        initial_global_step=global_step,
        num_epochs=config["num_epochs"],
        model_folder=config["model_folder"],
        model_basename=config["model_basename"],
        eval_every_n_steps=config["eval_every_n_steps"],
        save_every_n_steps=config["save_every_n_steps"],
        wandb_project=config["wandb_project"],
        wandb_key=config["wandb_key"],
        f16_precision=config["f16_precision"],
        scaler_state_dict=scaler_state_dict,
        max_grad_norm=config["max_grad_norm"],
        log_examples=config["log_examples"],
        logging_every_n_steps=config["logging_every_n_steps"],
    )

    trainer = Trainer(
        model=model,
        model_config=model_config,
        src_tokenizer=tokenizer_src,
        tgt_tokenizer=tokenizer_tgt,
        optimizer=optimizer,
        criterion=loss_fn,
        args=trainer_args,
        lr_scheduler=lr_scheduler,
    )

    trainer.train(
        train_dataloader=train_data_loader,
        val_dataloader=test_data_loader,
    )
