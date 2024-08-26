import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

from nmt.utils import create_encoder_mask, create_decoder_mask
from nmt.constants import SpecialToken


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

        self.sos_token = torch.tensor(
            [tokenizer_src.token_to_id(token=SpecialToken.SOS)], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [tokenizer_src.token_to_id(token=SpecialToken.EOS)], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [tokenizer_src.token_to_id(token=SpecialToken.PAD)], dtype=torch.int64
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: str) -> dict:
        # Locate the source and target text
        src_target_pair = self.dataset[index]
        src_text = src_target_pair["translation"][self.lang_src]
        tgt_text = src_target_pair["translation"][self.lang_tgt]

        # Tokenize the source and target text
        encoder_input_tokens = self.tokenizer_src.encode(sequence=src_text).ids
        decoder_input_tokens = self.tokenizer_tgt.encode(sequence=tgt_text).ids

        # Calculate the number of padding tokens
        # SOS + EOS = 2
        encoder_num_pad_tokens = self.seq_length - len(encoder_input_tokens) - 2
        # SOS = 1
        decoder_num_pad_tokens = self.seq_length - len(decoder_input_tokens) - 1

        if encoder_num_pad_tokens < 0 or decoder_num_pad_tokens < 0:
            return ValueError("Sequence length is too short")

        # Add SOS, EOS and PAD tokens to the encoder input
        encoder_input = torch.cat(
            tensors=[
                self.sos_token,
                torch.tensor(encoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * encoder_num_pad_tokens, dtype=torch.int64
                ),
            ]
        )
        # Add SOS and PAD tokens to the decoder input
        decoder_input = torch.cat(
            tensors=[
                self.sos_token,
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * decoder_num_pad_tokens, dtype=torch.int64
                ),
            ]
        )
        # Add EOS token to the label (target - what the decoder should output)
        label = torch.cat(
            tensors=[
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * decoder_num_pad_tokens, dtype=torch.int64
                ),
            ]
        )

        return {
            "encoder_input": encoder_input,  # (seq_length)
            "decoder_input": decoder_input,  # (seq_length)
            "encoder_mask": create_encoder_mask(
                encoder_input=encoder_input,
                pad_token_id=self.pad_token,
            ),  # (1, 1, seq_length)
            "decoder_mask": create_decoder_mask(
                decoder_input=decoder_input,
                pad_token_id=self.pad_token,
            ),  # (1, seq_length, seq_length)
            "label": label,  # (seq_length)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }
