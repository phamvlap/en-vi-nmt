import wandb
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from dataclasses import dataclass
from tqdm import tqdm
from tokenizers import Tokenizer

from transformer.models import Transformer, TransformerConfig
from transformer.functional import create_encoder_mask, create_decoder_mask
from .validation import run_validation
from .bleu import compute_bleu
from .utils import get_weights_file_path


@dataclass
class TrainerArguments:
    seq_length: int
    initial_epoch: int
    initial_global_step: int
    num_epochs: int
    model_folder: str
    model_basename: str
    eval_every_n_steps: int
    save_every_n_steps: int
    wandb_project: str
    wandb_key: str | None = None
    f16_precision: bool = False
    scaler_state_dict: dict | None = None
    max_grad_norm: float | None = None
    log_examples: bool = False
    logging_every_n_steps: int = 1000


class Trainer:
    def __init__(
        self,
        model: Transformer,
        model_config: TransformerConfig,
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        optimizer: optim.Optimizer,
        criterion: nn.CrossEntropyLoss,
        args: TrainerArguments,
        device: torch.device,
        lr_scheduler: optim.lr_scheduler.LRScheduler | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.model_config = model_config
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.optimizer = optimizer
        self.criterion = criterion
        self.args = args
        self.lr_scheduler = lr_scheduler

        # Automatic Mixed Precision
        self.scaler = None
        if self.args.f16_precision and torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler("cuda")
            if self.args.scaler_state_dict is not None:
                self.scaler.load_state_dict(self.args.scaler_state_dict)

        # Wandb
        args_dict = self.args.__dict__
        if self.args.wandb_key is not None:
            wandb.login(key=self.args.wandb_key)
            del args_dict["wandb_key"]
        saved_config = {
            "model_config": self.model_config.__dict__,
            "trainer_args": args_dict,
        }
        self.wb_run = wandb.init(
            project=self.args.wandb_project,
            config=saved_config,
        )

    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
        epoch = self.args.initial_epoch
        global_step = self.args.initial_global_step

        self.model.train()

        for epoch in range(epoch, self.args.num_epochs):
            torch.cuda.empty_cache()

            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Training epoch {epoch}/{self.args.num_epochs}",
            )

            for batch in batch_iterator:
                self.optimizer.zero_grad()

                encoder_input = batch["encoder_input"].to(self.device)
                decoder_input = batch["decoder_input"].to(self.device)
                labels = batch["labels"].to(self.device)

                encoder_mask = create_encoder_mask(
                    encoder_input=encoder_input,
                    pad_token_id=self.model_config.src_pad_token_id,
                )
                decoder_mask = create_decoder_mask(
                    decoder_input=decoder_input,
                    pad_token_id=self.model_config.tgt_pad_token_id,
                )

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.args.f16_precision and torch.cuda.is_available(),
                ):
                    decoder_output = self.model(
                        src=encoder_input,
                        tgt=decoder_input,
                        src_mask=encoder_mask,
                        tgt_mask=decoder_mask,
                    )

                    proj_output = self.model.project(decoder_output)

                    loss = self.criterion(
                        proj_output.view(-1, proj_output.size(-1)),
                        labels.view(-1),
                    )

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()

                    if (
                        self.args.max_grad_norm is not None
                        and self.args.max_grad_norm > 0
                    ):
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.args.max_grad_norm,
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                self.wb_run.log({"train/loss": loss.item()})

                batch_iterator.set_postfix({"loss": loss.item()})
                global_step += 1

                if global_step % self.args.eval_every_n_steps == 0:
                    val_loss = run_validation(
                        model=self.model,
                        val_data_loader=val_dataloader,
                        tokenizer_src=self.src_tokenizer,
                        tokenizer_tgt=self.tgt_tokenizer,
                        criterion=self.criterion,
                        device=self.device,
                    )
                    bleu_scores = compute_bleu(
                        model=self.model,
                        dataset=val_dataloader.dataset,
                        tgt_tokenizer=self.tgt_tokenizer,
                        seq_length=self.args.seq_length,
                        log_examples=self.args.log_examples,
                        logging_every_n_steps=self.args.logging_every_n_steps,
                    )
                    bleu_dict = {
                        f"val/bleu_{i+1}": bleu_scores[i]
                        for i in range(len(bleu_scores))
                    }
                    self.wb_run.log(
                        {
                            "val/loss": val_loss,
                            **bleu_dict,
                        },
                        step=global_step,
                    )

                if global_step % self.args.save_every_n_steps == 0:
                    self._save_checkpoint(global_step, epoch)

        self.wb_run.finish()

    def _save_checkpoint(self, global_step: int, epoch: int) -> None:
        model_filepath = get_weights_file_path(
            model_folder=self.args.model_folder,
            model_basename=self.args.model_basename,
            epoch=epoch,
        )
        checkpoint_states = {
            "epoch": epoch,
            "global_step": global_step,
            "model_config": self.model_config,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        if self.lr_scheduler is not None:
            checkpoint_states["lr_scheduler_state_dict"] = (
                self.lr_scheduler.state_dict()
            )
        if self.scaler is not None:
            checkpoint_states["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint_states, model_filepath)
