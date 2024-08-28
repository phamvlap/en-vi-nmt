def get_config() -> dict[str, int | str]:
    config = {
        "datasource": "harouzie/vi_en-translation",
        "batch_size_train": 8,  # number of samples in a batch
        "batch_size_test": 1,
        "num_epochs": 20,  # number of epochs
        "lr": 10**-4,  # learning rate
        "seq_length": 350,  # maximum sequence length
        "d_model": 512,  # model dimension
        "num_encoder": 6,
        "num_decoder": 6,
        "lang_src": "en",
        "lang_tgt": "vi",
        "train_ratio": 0.9,
        "test_ratio": 0.1,
        "dropout": 0.1,
        "model_folder": "weights",  # folder to save model weights
        "model_basename": "tmodel_",  # base name for model weights
        "preload": None,  # whether to restart training after maybe model crashed
        "tokenizer_file": "tokenizer_{0}.json",  # tokenizer file
        "experiment_name": "runs/tmodel",  # save the logs while training
    }

    return config


def get_weights_file_path(config: dict, epoch: str) -> str:
    return "{0}_{1}/{2}{3}.pt".format(
        config["datasource"],
        config["model_folder"],
        config["model_basename"],
        epoch,
    )
