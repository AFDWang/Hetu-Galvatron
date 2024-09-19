import torch
from torch.utils.data import Dataset
import numpy as np

from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.blended_megatron_dataset_config import GPTDatasetConfig
from megatron.core import mpu, tensor_parallel
from megatron import print_rank_0, get_args
from megatron.training import build_train_valid_test_data_iterators
from megatron.core.datasets.gpt_dataset import GPTDataset

class DataLoaderForLlama(Dataset):
    def __init__(self, args, device):
        self.vocab_size = args.vocab_size
        self.sentence_length = args.seq_length
        self.dataset_size = 2560 * 16
        self.data_length = np.random.randint(1,self.sentence_length+1,(self.dataset_size,))
        self.device = device

        self.input_ids = []
        for i in range(self.dataset_size):
            sentence = np.random.randint(0,self.vocab_size,(self.sentence_length,))
            sentence[self.data_length[i]:] = 0
            mask = np.ones((self.sentence_length,))
            mask[self.data_length[i]:] = 0
            
            padding_sentence = np.zeros(self.sentence_length + 1, dtype=sentence.dtype)
            padding_sentence[:self.sentence_length] = sentence
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
    return GPTDatasetConfig(
        is_built_on_rank=is_dataset_built_on_rank,
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=args.data_path,
        blend_per_split=[args.train_data_path, args.valid_data_path, args.test_data_path],
        split=args.split,
        path_to_cache=args.data_cache_path,
        return_document_ids=args.retro_return_doc_ids
    )

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        GPTDataset,
        train_val_test_num_samples,
        core_gpt_dataset_config_from_args(args)
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds

def get_train_valid_test_data_iterators():
    train_valid_test_datasets_provider.is_distributed = True
    train_data_iterator, valid_data_iterator, test_data_iterator \
            = build_train_valid_test_data_iterators(
                train_valid_test_datasets_provider)
    return train_data_iterator, valid_data_iterator, test_data_iterator


# TODO: support tokenizer, reset_position_ids, reset_attention_mask, eod_mask_loss
def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        args = get_args()
        return torch.empty(args.micro_batch_size,args.seq_length+1).cuda().long()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()

    return tokens_
