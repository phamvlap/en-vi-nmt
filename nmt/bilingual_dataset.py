import torch

from torch import Tensor
from torch.utils.data import Dataset
from tokenizers import Tokenizer

from .constants import SpecialToken


class BilingualDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        tokenizer_src: Tokenizer,
        tokenizer_tgt: Tokenizer,
        lang_src: str,
        lang_tgt: str,
        seq_length: int,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        self.seq_length = seq_length

        self.src_pad_token_id = tokenizer_src.token_to_id(token=SpecialToken.PAD)
        self.tgt_pad_token_id = tokenizer_tgt.token_to_id(token=SpecialToken.PAD)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: str) -> dict:
        # Locate the source and target text
        source_target_pair = self.dataset[index]
        src_text = source_target_pair[self.lang_src]
        tgt_text = source_target_pair[self.lang_tgt]

        # Tokenize the source and target text
        encoder_input_tokens = self.tokenizer_src.encode(sequence=src_text).ids
        decoder_input_tokens = self.tokenizer_tgt.encode(sequence=tgt_text).ids

        encoder_input_tokens = Tensor(encoder_input_tokens).to(torch.int64)
        decoder_input_tokens = Tensor(decoder_input_tokens).to(torch.int64)

        # Calculate the number of padding tokens
        # SOS + EOS = 2
        src_num_pad_tokens = self.seq_length - len(encoder_input_tokens) - 2
        # SOS = 1
        tgt_num_pad_tokens = self.seq_length - len(decoder_input_tokens) - 1

        if src_num_pad_tokens >= 0:
            # Add SOS, EOS and PAD tokens to the encoder input
            encoder_input = torch.cat(
                [
                    self.sos_token,
                    encoder_input_tokens,
                    self.eos_token,
                    Tensor([self.src_pad_token_id] * src_num_pad_tokens).to(
                        torch.int64
                    ),
                ]
            )
        else:
            # Truncate the encoder input
            encoder_input = torch.cat(
                [
                    self.sos_token,
                    encoder_input_tokens[: self.seq_length - 1],
                    self.eos_token,
                ]
            )

        if tgt_num_pad_tokens >= 0:
            # Add SOS and PAD tokens to the decoder input
            decoder_input = torch.cat(
                [
                    self.sos_token,
                    torch.tensor(decoder_input_tokens, dtype=torch.int64),
                    Tensor([self.tgt_pad_token_id] * tgt_num_pad_tokens).to(
                        torch.int64
                    ),
                ]
            )
            # Add EOS token to the label (target - what the decoder should output)
            labels = torch.cat(
                [
                    torch.tensor(decoder_input_tokens, dtype=torch.int64),
                    self.eos_token,
                    Tensor([self.tgt_pad_token_id] * tgt_num_pad_tokens).to(
                        torch.int64
                    ),
                ]
            )
        else:
            # Truncate the decoder input and label
            decoder_input = torch.cat(
                [
                    self.sos_token,
                    decoder_input_tokens[: self.seq_length - 1],
                ]
            )
            labels = torch.cat(
                [
                    decoder_input_tokens[: self.seq_length - 1],
                    self.eos_token,
                ]
            )

        return {
            "encoder_input": encoder_input,  # (seq_length, )
            "decoder_input": decoder_input,  # (seq_length, )
            "labels": labels,  # (seq_length, )
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
