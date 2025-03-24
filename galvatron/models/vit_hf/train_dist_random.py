import torch
from torch import nn
from torch.optim import Adam
from transformers import ViTConfig, ViTForImageClassification
from tqdm import tqdm
import os
from galvatron.utils import set_seed, distributed_dataloader, print_loss
from galvatron.core import initialize_galvatron
from galvatron.models.vit_hf.ViTModel_hybrid_parallel import vit_model_hp, get_vit_config, get_runtime_profiler
from galvatron.models.vit_hf.dataloader import DataLoaderForViT, vit_collate_fn
from galvatron.models.vit_hf.meta_configs import model_name, model_layer_configs
from galvatron.models.vit_hf.arguments import model_args
from galvatron.core.runtime.initialize import init_empty_weights
from megatron.training.arguments import _print_args

def train(args):
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()

    config = get_vit_config(args)
    model = vit_model_hp(config, args)
    
    if local_rank == 0:
        print("Creating Dataset...")
   
    trainloader = distributed_dataloader(
        dataset=DataLoaderForViT(config, device),  
        global_bsz=args.global_train_batch_size,
        shuffle=True,
        args=args,
        group=model.dp_groups_whole[0].group,
        collate_fn=vit_collate_fn  
    )
    
    if local_rank == 0:
        _print_args("arguments", args)
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    path = os.path.dirname(os.path.abspath(__file__))
    profiler = get_runtime_profiler(args, path, config)
    
    profiler.profile_memory(0, "After creating model")
    if local_rank == 0:
        print("Start training...")
    
    if args.profile_forward:
        torch.set_grad_enabled(False)

    for ep in range(args.epochs):
        if not args.check_loss and not args.profile:
            trainloader = tqdm(trainloader)
        for iter, batch in enumerate(trainloader):
            model_inputs, kwargs, loss_func = batch

            profiler.profile_time_start(iter)
            profiler.profile_memory(iter, "Before Forward")

            input_ids = model_inputs
            batch = [input_ids]

            loss = model.forward_backward(
                batch=batch,
                iter=iter,
                profiler=profiler,
                loss_func=loss_func,
                **kwargs
            )

            profiler.profile_memory(iter, "After Backward")
            
            optimizer.step()
            
            profiler.profile_memory(iter, "After optimizer_step")
            
            optimizer.zero_grad()
            
            print_loss(args, loss, ep, iter)

            profiler.post_profile_memory(iter)
            profiler.profile_time_end(iter)

            torch.distributed.barrier()
            
if __name__ == '__main__':
    args = initialize_galvatron(model_args, mode='train_dist')
    set_seed()
    train(args)