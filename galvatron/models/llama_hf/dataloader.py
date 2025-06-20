from functools import partial
from typing import List

import numpy as np
import torch
from megatron.core import mpu, tensor_parallel
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.training import get_args, get_tokenizer, print_rank_0
from megatron.training.training import build_train_valid_test_data_iterators
from megatron.training.utils import (
    average_losses_across_data_parallel_group,
    get_batch_on_this_tp_rank,
    get_batch_on_this_cp_rank,
    get_ltor_masks_and_position_ids,
)
from torch import Tensor
from torch.utils.data import Dataset

from galvatron.core.runtime.hybrid_parallel_config import get_chunks
from galvatron.core.runtime.pipeline.utils import chunk_batch


def random_get_ltor_masks_and_position_ids(data):
    """Build masks and position id for left to right model."""
    micro_batch_size, seq_length = data.size()
    att_mask_batch = 1
    attention_mask = torch.tril(torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)).view(
        att_mask_batch, 1, seq_length, seq_length
    )

    # position_ids = torch.arange(seq_length, dtype=torch.long,
    #                             device=data.device)
    # position_ids = position_ids.unsqueeze(0).expand_as(data)
    attention_mask = attention_mask < 0.5

    return attention_mask  # , position_ids


def random_collate_fn(batch):
    tokens_ = torch.stack(batch, dim=0)
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    args = get_args()
    if not args.use_flash_attn:
        attention_mask = random_get_ltor_masks_and_position_ids(tokens)
    else:
        attention_mask = None
    return tokens, {"attention_mask": attention_mask, "labels": labels}, None


class DataLoaderForLlama(Dataset):
    def __init__(self, args, device, dataset_size=2560 * 16):
        self.vocab_size = args.vocab_size
        self.sentence_length = args.seq_length
        self.dataset_size = dataset_size
        self.data_length = np.random.randint(1, self.sentence_length + 1, (self.dataset_size,))
        self.device = device

        self.input_ids = []
        for i in range(self.dataset_size):
            sentence = np.random.randint(0, self.vocab_size, (self.sentence_length,))
            sentence[self.data_length[i] :] = 0
            mask = np.ones((self.sentence_length,))
            mask[self.data_length[i] :] = 0

            padding_sentence = np.zeros(self.sentence_length + 1, dtype=sentence.dtype)
            padding_sentence[: self.sentence_length] = sentence
            self.input_ids.append(padding_sentence)

        self.input_ids = np.array(self.input_ids)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if idx >= self.dataset_size:
            raise IndexError
        input_ids = torch.LongTensor(self.input_ids[idx]).to(self.device)
        return input_ids


def is_dataset_built_on_rank():
    return (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()) and mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=args.data_path,
        blend_per_split=[args.train_data_path, args.valid_data_path, args.test_data_path],
        split=args.split,
        path_to_cache=args.data_cache_path,
        mock=args.mock_data,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        GPTDataset, train_val_test_num_samples, is_dataset_built_on_rank, core_gpt_dataset_config_from_args(args)
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

    args = get_args()
    # TODO: this is pretty hacky, find a better way
    batch_size = args.global_train_batch_size // mpu.get_data_parallel_world_size()
    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return fake_tensor(batch_size), {}, None
        # return torch.empty(args.micro_batch_size,args.seq_length+1).cuda().long()

    args = get_args()
    batch = get_batch_on_this_tp_rank(data_iterator)
    if args.cp_mode == "zigzag":
        batch = get_batch_on_this_cp_rank(batch)#TODO
    micro_lossmask = chunk_batch([batch["loss_mask"]], get_chunks(args))
    # print(f"Rank {torch.cuda.current_device()} with input {tokens}")
    if batch["tokens"] == None:
        batch["tokens"] = fake_tensor(batch_size)
    return (
        batch["tokens"],
        {
            "position_ids": batch["position_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        },
        partial(loss_func, micro_lossmask),
    )

#TODO：摆脱对mpu的依赖，怎么靠galvatron实现
def loss_func(micro_lossmask: Tensor, label: List, output_tensor: List):
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
#average_losses_across_seq_data_parallel_group
    averaged_loss = average_losses_across_data_parallel_group([loss])
    #averaged_loss = average_losses_across_context_parallel_group([averaged_loss])

    micro_lossmask.pop(0)
    return loss, averaged_loss[0]


def average_losses_across_context_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses,
                                 group=mpu.get_context_parallel_group())
    averaged_losses = averaged_losses / \
        torch.distributed.get_world_size(group=mpu.get_context_parallel_group())

    return averaged_losses

#TODO:似乎没有用了
def get_zigzag_tokens_on_this_cp_rank(tokens: torch.Tensor, cp_rank: int, cp_size: int) -> torch.Tensor:
    if cp_size == 1: 
        return tokens

    batch_size, original_seq_len = tokens.shape
    num_total_chunks = 2 * cp_size

    chunk_len = original_seq_len // num_total_chunks
    processed_seq_len = chunk_len * num_total_chunks
    tokens_to_process = tokens[:, :processed_seq_len].contiguous()

   
    reshaped_tokens = tokens_to_process.view(batch_size, num_total_chunks, chunk_len)
    
    idx1 = cp_rank
    idx2 = num_total_chunks - 1 - cp_rank
    index_tensor = torch.tensor([idx1, idx2], device=tokens.device, dtype=torch.long)

    selected_chunks = torch.index_select(reshaped_tokens, 1, index_tensor)

    output_tokens = selected_chunks.contiguous().view(batch_size, -1)

    return output_tokens
