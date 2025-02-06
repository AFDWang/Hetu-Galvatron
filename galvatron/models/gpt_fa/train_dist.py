import os

import torch
from flash_attn.models.gpt import GPTLMHeadModel
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from galvatron.core import initialize_galvatron
from galvatron.core.runtime.utils import set_megatron_args_for_dataset
from galvatron.models.gpt_fa.arguments import model_args
from galvatron.models.gpt_fa.dataloader import DataLoaderForGPT, get_batch, get_train_valid_test_data_iterators
from galvatron.models.gpt_fa.GPTModel_hybrid_parallel import get_gpt_config, get_runtime_profiler, gpt_model_hp
from galvatron.models.gpt_fa.meta_configs import model_layer_configs, model_name
from galvatron.utils import distributed_dataloader, print_loss, set_seed


def train(args):
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()

    config = get_gpt_config(args)
    model = gpt_model_hp(config, args)

    if local_rank == 0:
        print("Creating Dataset...")

    set_megatron_args_for_dataset(
        args, model, model.sp_groups_whole[0] if args.vocab_sp else model.tp_groups_whole[0], model.dp_groups_whole[0]
    )
    # if local_rank == 0:
    #     _print_args("arguments", args)

    train_data_iterator, valid_data_iterator, test_data_iterator = get_train_valid_test_data_iterators()

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    path = os.path.dirname(os.path.abspath(__file__))
    profiler = get_runtime_profiler(args, path, config)

    profiler.profile_memory(0, "After creating model")
    if local_rank == 0:
        print("Start training...")
    for iter in range(args.train_iters):
        # if not args.check_loss and not args.profile:
        #     trainloader = tqdm(trainloader)
        tokens, kwargs, loss_func = get_batch(train_data_iterator)

        # print(batch.shape)
        # print(batch)
        profiler.profile_time_start(iter)
        profiler.profile_memory(iter, "Before Forward")

        input_ids = tokens
        batch = [input_ids]

        loss = model.forward_backward(batch, iter, profiler, loss_func=loss_func, **kwargs)

        profiler.profile_memory(iter, "After Backward")

        optimizer.step()

        profiler.profile_memory(iter, "After optimizer_step")

        optimizer.zero_grad()

        # print_loss(args, loss, -1, iter)

        profiler.post_profile_memory(iter)
        profiler.profile_time_end(iter, loss)

        torch.distributed.barrier()


if __name__ == "__main__":
    args = initialize_galvatron(model_args, mode="train_dist")
    set_seed()
    train(args)
