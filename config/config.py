def get_config() -> dict:
    config = {}

    # General
    config["seed"] = 42
    config["model_folder"] = "weights"  # folder to save model weight
    config["model_basename"] = "tmodel_"  # base name for model weight
    config["preload"] = "latest"  # whether to restart training after maybe model crashe
    config["experiment_name"] = "runs/tmodel"  # save the logs while trainin

    # Data
    config["datasource"] = "harouzie/vi_en-translation"
    config["lang_src"] = "en"
    config["lang_tgt"] = "vi"
    config["shuffle"] = True
    config["is_sampling"] = False
    config["num_samples"] = 1000
    config["train_size"] = 0.8
    config["val_size"] = 0.1
    config["test_size"] = 0.1
    config["train_data_file"] = "data/train.csv"
    config["val_data_file"] = "data/val.csv"
    config["test_data_file"] = "data/test.csv"

    # Tokenizers
    config["tokenizer_type"] = "bpe"  # word_level, bpe
    config["tokenizer_dir"] = "tokenizers"
    config["tokenizer_file"] = "tokenizer_{0}.json"
    config["min_freq"] = 2
    config["is_train_tokenizer"] = True

    # Dataloader
    config["batch_size_train"] = 32
    config["batch_size_val"] = 8
    config["batch_size_test"] = 2
    config["shuffle_dataloader"] = True

    # Training
    config["num_epochs"] = 20
    config["eval_every_n_steps"] = 5000
    config["save_every_n_steps"] = 10000
    config["wandb_project"] = "en-vi-nmt"
    config["wandb_key"] = "wandb_key"
    config["f16_precision"] = True
    config["max_grad_norm"] = 1.0
    config["log_examples"] = False
    config["logging_every_n_steps"] = 2000
    config["output_filepath"] = "statictis/testing_results.csv"

    # Model
    config["seq_length"] = 350  # maximum sequence length
    config["src_seq_length"] = 350
    config["tgt_seq_length"] = 350
    config["d_model"] = 512
    config["num_heads"] = 8
    config["num_encoder_layers"] = 6
    config["num_decoder_layers"] = 6
    config["dropout"] = 0.1
    config["d_ff"] = 2048

    # Optimizer and Learning rate scheduler
    config["optimizer"] = "adamw"  # adam, adamw
    config["lr"] = 10**-4  # learning rat
    config["betas"] = (0.9, 0.98)
    config["eps"] = 10**-9
    config["weight_decay"] = 1.0
    config["lr_scheduler"] = "noam"
    config["warmup_steps"] = 4000

    return config
