import os
from functools import partial
from typing import List

import numpy as np
import torch
from megatron.core import mpu, tensor_parallel
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.t5_dataset import T5MaskedWordPieceDataset, T5MaskedWordPieceDatasetConfig
from megatron.training import get_args, get_tokenizer, print_rank_0
from megatron.training.training import build_train_valid_test_data_iterators
from megatron.training.utils import (
    average_losses_across_data_parallel_group,
    get_batch_on_this_tp_rank,
    get_ltor_masks_and_position_ids,
)
from torch import Tensor
from torch.utils.data import Dataset

from galvatron.core.runtime.hybrid_parallel_config import get_chunks
from galvatron.core.runtime.pipeline.utils import chunk_batch


def random_collate_fn(batch):
    enc_input_ids, dec_input_ids, dec_output_ids, enc_mask, dec_mask, enc_dec_mask = zip(*batch)
    enc_tokens = torch.stack(enc_input_ids, dim=0)
    dec_tokens = torch.stack(dec_input_ids, dim=0)
    dec_labels = torch.stack(dec_output_ids, dim=0)
    enc_mask = torch.stack(enc_mask, dim=0) < 0.5
    dec_mask = torch.stack(dec_mask, dim=0)
    enc_dec_mask = torch.stack(enc_dec_mask, dim=0)

    return (
        enc_tokens,
        {
            "dec_tokens": dec_tokens,
            "enc_attn_mask": enc_mask,
            "dec_attn_mask": dec_mask,
            "enc_dec_attn_mask": enc_dec_mask,
            "dec_labels": dec_labels,
        },
        None,
    )


def _make_attention_mask(source_block: np.ndarray, target_block: np.ndarray) -> np.ndarray:
    mask = (target_block[None, :] >= 1) * (source_block[:, None] >= 1)
    return mask.astype(np.int64)


def _make_history_mask(block: np.ndarray) -> np.ndarray:
    arange = np.arange(block.shape[0])
    mask = arange[None,] <= arange[:, None]
    return mask.astype(np.int64)


class DataLoaderForT5(Dataset):
    def __init__(self, args, device, dataset_size=20 * 16):
        self.vocab_size = args.vocab_size
        self.encoder_seq_length = args.encoder_seq_length
        self.decoder_seq_length = args.decoder_seq_length
        self.dataset_size = dataset_size
        self.device = device

        cache_dir = os.path.join(os.path.dirname(__file__), "data_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(
            cache_dir,
            f"t5_data_v{self.vocab_size}_{self.encoder_seq_length}_{self.decoder_seq_length}_{dataset_size}.npz",
        )

        if os.path.exists(cache_file):
            print(f"Loading cached dataset from {cache_file}")
            cached_data = np.load(cache_file)
            self.enc_input_ids = cached_data["enc_input_ids"]
            self.dec_input_ids = cached_data["dec_input_ids"]
            self.dec_output_ids = cached_data["dec_output_ids"]
            self.enc_mask = cached_data["enc_mask"]
            self.dec_mask = cached_data["dec_mask"]
            self.enc_dec_mask = cached_data["enc_dec_mask"]
        else:
            print(f"Generating new dataset and caching to {cache_file}")

            self.enc_data_length = np.random.randint(1, self.encoder_seq_length + 1, (self.dataset_size,))
            self.dec_data_length = np.random.randint(1, self.decoder_seq_length + 1, (self.dataset_size,))
            self.enc_input_ids = []
            self.dec_input_ids = []
            self.dec_output_ids = []
            self.enc_mask = []
            self.dec_mask = []
            self.enc_dec_mask = []
            for i in range(self.dataset_size):
                enc_sentence = np.random.randint(1, self.vocab_size, (self.encoder_seq_length,))
                enc_sentence[self.enc_data_length[i] :] = 0
                padding_enc_sentence = np.zeros(self.encoder_seq_length, dtype=enc_sentence.dtype)
                padding_enc_sentence[: self.encoder_seq_length] = enc_sentence
                self.enc_input_ids.append(padding_enc_sentence)

                dec_sentence = np.random.randint(1, self.vocab_size, (self.decoder_seq_length,))
                dec_sentence[self.dec_data_length[i] :] = 0
                padding_dec_sentence = np.zeros(self.decoder_seq_length, dtype=dec_sentence.dtype)
                padding_dec_sentence[: self.decoder_seq_length] = dec_sentence
                self.dec_input_ids.append(padding_dec_sentence)

                label = dec_sentence[1:]
                label = np.append(label, -1)
                label[self.dec_data_length[i] - 1 :] = -1
                self.dec_output_ids.append(label)

                self.enc_mask.append(_make_attention_mask(enc_sentence, enc_sentence))
                self.dec_mask.append(
                    _make_attention_mask(dec_sentence, dec_sentence) * _make_history_mask(dec_sentence)
                )
                self.enc_dec_mask.append(_make_attention_mask(dec_sentence, enc_sentence))

            self.enc_input_ids = np.array(self.enc_input_ids)
            self.dec_input_ids = np.array(self.dec_input_ids)
            self.dec_output_ids = np.array(self.dec_output_ids)
            self.enc_mask = np.array(self.enc_mask)
            self.dec_mask = np.array(self.dec_mask)
            self.enc_dec_mask = np.array(self.enc_dec_mask)

            np.savez(
                cache_file,
                enc_input_ids=self.enc_input_ids,
                dec_input_ids=self.dec_input_ids,
                dec_output_ids=self.dec_output_ids,
                enc_mask=self.enc_mask,
                dec_mask=self.dec_mask,
                enc_dec_mask=self.enc_dec_mask,
            )

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if idx >= self.dataset_size:
            raise IndexError
        enc_input_ids = torch.LongTensor(self.enc_input_ids[idx]).to(self.device)
        dec_input_ids = torch.LongTensor(self.dec_input_ids[idx]).to(self.device)
        dec_output_ids = torch.LongTensor(self.dec_output_ids[idx]).to(self.device)
        enc_mask = torch.LongTensor(self.enc_mask[idx]).to(self.device)
        dec_mask = torch.LongTensor(self.dec_mask[idx]).to(self.device)
        enc_dec_mask = torch.LongTensor(self.enc_dec_mask[idx]).to(self.device)
        return enc_input_ids, dec_input_ids, dec_output_ids, enc_mask, dec_mask, enc_dec_mask


def is_dataset_built_on_rank():
    return mpu.get_tensor_model_parallel_rank() == 0


def core_t5_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return T5MaskedWordPieceDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.encoder_seq_length,
        sequence_length_decoder=args.decoder_seq_length,
        blend=args.data_path,
        blend_per_split=[args.train_data_path, args.valid_data_path, args.test_data_path],
        split=args.split,
        path_to_cache=args.data_cache_path,
        mock=False,
        tokenizer=tokenizer,
        masking_probability=args.mask_prob,
        short_sequence_probability=args.short_seq_prob,
        masking_max_ngram=10,
        masking_do_full_word=True,
        masking_do_permutation=False,
        masking_use_longer_ngrams=False,
        masking_use_geometric_distribution=True,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        T5MaskedWordPieceDataset,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        core_t5_dataset_config_from_args(args),
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def get_train_valid_test_data_iterators():
    train_valid_test_datasets_provider.is_distributed = True
    train_data_iterator, valid_data_iterator, test_data_iterator = build_train_valid_test_data_iterators(
        train_valid_test_datasets_provider
    )
    return train_data_iterator, valid_data_iterator, test_data_iterator


def fake_tensor(bsz):
    return torch.zeros([bsz, 1], device="cuda")


def get_batch(data_iterator):
    """Generate a batch."""

    keys = ["text_enc", "text_dec", "labels", "loss_mask", "enc_mask", "dec_mask", "enc_dec_mask"]
    datatype = torch.int64
    args = get_args()

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_enc = data_b["text_enc"].long()
    tokens_dec = data_b["text_dec"].long()
    labels = data_b["labels"].long()
    loss_mask = data_b["loss_mask"].float()
    micro_lossmask = chunk_batch([loss_mask], get_chunks(args))

    enc_mask = data_b["enc_mask"] < 0.5
    dec_mask = data_b["dec_mask"] < 0.5
    enc_dec_mask = data_b["enc_dec_mask"] < 0.5

    return (
        tokens_enc,
        {
            "dec_tokens": tokens_dec,
            "enc_attn_mask": enc_mask,
            "dec_attn_mask": dec_mask,
            "enc_dec_attn_mask": enc_dec_mask,
            "dec_labels": labels,
        },
        partial(loss_func, micro_lossmask),
    )


def loss_func(micro_lossmask: List, label: List, output_tensor: List):
    """Loss function.

    Args:
        loss_mask (Tensor): Used to mask out some portions of the loss
        output_tensor (Tensor): The tensor with the losses
    """
    loss_mask = micro_lossmask[0][0]
    args = get_args()
    output_tensor = output_tensor[0]
    losses = output_tensor.float()
    # if torch.cuda.current_device()==0:
    #     print(f"loss {losses}")
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    averaged_loss = average_losses_across_data_parallel_group([loss])

    micro_lossmask.pop(0)
    return loss, averaged_loss[0]
