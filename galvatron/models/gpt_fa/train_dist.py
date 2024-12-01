import torch
from torch import nn
from torch.optim import Adam
from flash_attn.models.gpt import GPTLMHeadModel
from tqdm import tqdm
import os
from galvatron.utils import set_seed, distributed_dataloader, print_loss
from galvatron.core import initialize_galvatron, GalvatronProfiler
from galvatron.models.gpt_fa.GPTModel_hybrid_parallel import get_hybrid_parallel_configs, construct_hybrid_parallel_model
from galvatron.models.gpt_fa.dataloader import DataLoaderForGPT, get_batch, get_train_valid_test_data_iterators
from galvatron.models.gpt_fa.meta_configs import config_from_meta, set_model_config, model_name, model_layer_configs
from galvatron.models.gpt_fa.arguments import model_args
from galvatron.core.utils import set_megatron_args_for_dataset

def train(args):
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()

    config = config_from_meta(args.model_size)
    config = set_model_config(config, args)
    if local_rank == 0:
        print(config)
    
    hybrid_parallel_configs = get_hybrid_parallel_configs(model_config=config, training_args=args)
    if local_rank == 0:
        print("Creating Model...")

    gpt_model = GPTLMHeadModel(config, device='meta' if args.initialize_on_meta else 'cpu')
    model = construct_hybrid_parallel_model(
        model=gpt_model, 
        model_config=config, 
        training_args=args, 
        hybrid_parallel_configs=hybrid_parallel_configs
    )
    
    # from galvatron.models.gpt_fa import gpt_model_hp
    # model = gpt_model_hp(config, args)
    
    if local_rank == 0:
        print("Creating Dataset...")
    
    set_megatron_args_for_dataset(args, model, model.sp_groups_whole[0] if args.vocab_sp else model.tp_groups_whole[0], model.dp_groups_whole[0])
    # if local_rank == 0:
    #     _print_args("arguments", args)

    train_data_iterator, valid_data_iterator, test_data_iterator = get_train_valid_test_data_iterators()
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    path = os.path.dirname(os.path.abspath(__file__))
    profiler = GalvatronProfiler(args)
    profiler.set_profiler_dist(path, model_layer_configs(config), model_name(config))
    
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
        
        loss = model.forward_backward(batch, iter, profiler, 
                                      loss_func=loss_func,
                                      **kwargs)
        
        profiler.profile_memory(iter, "After Backward")
        
        optimizer.step()
        
        profiler.profile_memory(iter, "After optimizer_step")
        
        optimizer.zero_grad()
        
        # print_loss(args, loss, -1, iter)

        profiler.post_profile_memory(iter)
        profiler.profile_time_end(iter, loss)

        torch.distributed.barrier()

if __name__ == '__main__':
    args = initialize_galvatron(model_args, mode='train_dist')
    set_seed()
    train(args)