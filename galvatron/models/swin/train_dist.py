import os

import torch
from torch import nn
from tqdm import tqdm

from galvatron.core import initialize_galvatron
from galvatron.core.runtime.initialize import init_empty_weights
from galvatron.core.runtime.utils import (
    clip_grad_norm,
    get_optimizer_and_param_scheduler,
    set_megatron_args_for_dataset,
)
from galvatron.models.swin.arguments import model_args
from galvatron.models.swin.dataloader import DataLoaderForSwin, get_batch, get_train_valid_test_data_iterators, loss_func
from galvatron.models.swin.SwinModel_checkpoint import save_swin_module
from galvatron.models.swin.SwinModel_hybrid_parallel import get_runtime_profiler, get_swin_config, swin_model_hp
from galvatron.utils import distributed_dataloader, print_loss, set_seed
from megatron.training.arguments import _print_args


def train(args):
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()

    config = get_swin_config(args)
    model = swin_model_hp(config, args)

    if local_rank == 0:
        print("Creating Dataset...")

    set_megatron_args_for_dataset(
        args, model, model.sp_groups_whole[0] if args.vocab_sp else model.tp_groups_whole[0], model.dp_groups_whole[0]
    )
    if local_rank == 0:
        _print_args("arguments", args)

    train_data_iterator, valid_data_iterator, test_data_iterator = get_train_valid_test_data_iterators()

    optimizer, opt_param_scheduler = get_optimizer_and_param_scheduler(model, args)

    path = os.path.dirname(os.path.abspath(__file__))
    profiler = get_runtime_profiler(args, path, config)

    profiler.profile_memory(0, "After creating model")
    if local_rank == 0:
        print("Start training...")

    for iter in range(args.iteration, args.train_iters):
        tokens, kwargs, loss_func = get_batch(train_data_iterator)
        profiler.profile_time_start(iter)
        profiler.profile_memory(iter, "Before Forward")

        input_ids = tokens
        batch = [input_ids]

        loss = model.forward_backward(batch, iter, profiler, loss_func=loss_func, **kwargs)

        profiler.profile_memory(iter, "After Backward")

        total_norm = clip_grad_norm(model, args.clip_grad)
        optimizer.step()
        opt_param_scheduler.step(increment=args.global_batch_size)

        profiler.profile_memory(iter, "After optimizer_step")

        optimizer.zero_grad()

        profiler.post_profile_memory(iter)
        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]
        profiler.profile_time_end(iter, loss, learning_rate, total_norm)

        torch.distributed.barrier()

        if args.save != None and (iter + 1) % args.save_interval == 0:
            save_swin_module(args.save, model, optimizer, opt_param_scheduler, iter + 1, args)


if __name__ == "__main__":
    args = initialize_galvatron(model_args, mode="train_dist")
    set_seed()
    train(args)
