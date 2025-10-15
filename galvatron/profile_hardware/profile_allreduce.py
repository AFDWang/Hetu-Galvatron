import torch
from torch import nn
from torch.optim import Adam
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import random
from tqdm import tqdm
import time
import os
import sys
from typing import Tuple, List
import argparse

import galvatron
from galvatron.core.runtime.pipeline import PipelineParallel, PipeSequential
from galvatron.core.runtime.comm_groups import gen_comm_groups
from galvatron.utils import read_json_config, write_json_config


def init_method_constant(val):
    def init_(tensor):
        return torch.nn.init.constant_(tensor, val)
    return init_

class pre_sync_module(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, hidden_states):
        return hidden_states

class pre_mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1024, 1024)

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        return hidden_states

def _reduce(input_, group):
    """All-reduce the the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size(group=group)==1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_.contiguous(), group=group)

    return input_

class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""
    
    @staticmethod
    def forward(ctx, input_, group):
        return _reduce(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""
    
    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output, ctx.group), None

def reduce_from_tensor_model_parallel_region_group(input_, group):
    return _ReduceFromModelParallelRegion.apply(input_, group)

def copy_to_tensor_model_parallel_region_group(input_, group):
    return _CopyToModelParallelRegion.apply(input_, group)

class allreduce_block(nn.Module):
    def __init__(self, tp_group):
        super().__init__()
        self.tp_group = tp_group
        self.linear = nn.Linear(1024, 1024)

    def forward(self, hidden_states):
        hidden_states = copy_to_tensor_model_parallel_region_group(hidden_states, self.tp_group.group)
        hidden_states = reduce_from_tensor_model_parallel_region_group(hidden_states, self.tp_group.group)
        hidden_states = hidden_states.requires_grad_(True)
        return hidden_states

class DataLoaderRandom(Dataset):
    def __init__(self, local_bsz, profile_time):
        # world_size = torch.distributed.get_world_size()
        self.dataset_size = local_bsz*11
        self.input = np.random.rand(*(self.dataset_size, 512, 1024))
        self.profile_time = profile_time

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if idx >= self.dataset_size:
            raise IndexError
        if self.profile_time == 1:
            input = torch.tensor(self.input[idx],dtype=torch.bfloat16)
        else:
            input = torch.FloatTensor(self.input[idx])
        return input

def fake_loss_func(labels, outputs):
    output = outputs[0]
    loss = output.sum()
    loss = loss.requires_grad_(True)
    return loss, loss

def set_seed(rank):
    seed = 123 + rank
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train(args):
    torch.distributed.init_process_group(backend="nccl")
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    set_seed(rank)
    world_size = torch.distributed.get_world_size()
    node_num = world_size / args.nproc_per_node
    # assert(args.nproc_per_node == 8)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Initialize torch.profiler before initializing CUDA context to avoid the bug
    # Refer to https://github.com/pytorch/pytorch/issues/60158
    with torch.profiler.profile() as p:
        pass

    pp_deg = args.pp_deg
    args.num_layers = 24
    # args.local_batch_size = 32
    train_batch_size_input = args.local_batch_size
    if rank == 0:
        print('local_bsz = %d'%train_batch_size_input)

    dataset = DataLoaderRandom(train_batch_size_input, args.profile_time)
    trainloader = DataLoader(dataset=dataset,
                            batch_size=train_batch_size_input)

    all_tp_sizes = [args.global_tp_deg] * 24
    all_sp_sizes = [1] * 24
    tp_consecutive_flags = [args.global_tp_consec] * 24
    pp_group, tp_groups, _, _, _, _, _, _, _, _ = gen_comm_groups(all_tp_sizes, all_sp_sizes, pp_deg, tp_consecutive_flags)


    model = PipeSequential()
    model.add_module('pre_sync_module', pre_sync_module())
    model.add_module('pre_mlp', pre_mlp())
    for i in range(len(all_tp_sizes)):
        module = allreduce_block(tp_group=tp_groups[i])
        model.add_module('mlp_%d'%i, module)

    if args.profile_time == 1:
        model = model.bfloat16()

    avg_num_layers = args.num_layers // args.pp_deg
    pp_ranks = [0, 0]
    for i in range(args.pp_deg):
        pp_ranks += [i] * avg_num_layers
    
    layer_output_tensor_shapes = [[[-1, 512, 1024]]] * len(pp_ranks)
    model = model.to(device)
    model = PipelineParallel(
                model = model, 
                model_ranks = pp_ranks, 
                layer_output_tensor_shapes = layer_output_tensor_shapes, 
                chunks=1, 
                process_group = pp_group.ranks, 
                nproc_per_node=8,
                info=False)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=0.01)

    # Calculate theoretical communication message size
    tp_size = args.global_tp_deg
    pp_size = args.pp_deg
    dp_size = world_size // pp_deg // tp_size
    bs = args.local_batch_size
    # per size: 1MB * bs when profile time == 1
    allreduce_message_size_per_layer = 2*(tp_size-1)/tp_size*(bs*512*1024*2*4/1024/1024)
    allreduce_message_size_total = allreduce_message_size_per_layer * 24 // pp_deg
    if rank == 0:
        print('Strategy: %d_%d_%d'%(pp_size,tp_size,args.global_tp_consec))
        print('[allreduce_message_size]: per_layer %d MB, total %d MB'%(allreduce_message_size_per_layer,allreduce_message_size_total))

    def trace_handler(prof):
        # if rank == 0:
        try:
            table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=5)
            if rank == 0:
                print(table)
            table = table.split('\n')
            def split_line(line):
                line = line.split('  ')
                ls = []
                for s in line:
                    if len(s):
                        ls.append(s.strip())
                return ls
            def str2time(s):
                if 'ms' in s:
                    return float(s[:-2])
                elif 'us' in s:
                    return float(s[:-2])*1e-3
                else:
                    return float(s[:-1])*1e3
            for line in table:
                if 'Name' in line:
                    title = split_line(line)
                if 'ncclKernel_AllReduce' in line:
                    result = split_line(line)
            for i in range(len(title)):
                # print('%s: %s'%(title[i],result[i]))
                if 'CUDA total' in title[i]:
                    cuda_total_idx = i
                if "Calls" in title[i]:
                    comm_num = int(result[i])
            comm_time = str2time(result[cuda_total_idx])
            
            if args.profile_time == 0:
                allreduce_time_24_layer = comm_time / 10
                comm_coe = allreduce_message_size_total / allreduce_time_24_layer
                comm_coe = torch.tensor([comm_coe]).to(device)
                torch.distributed.all_reduce(comm_coe, group=tp_groups[0].group, op=torch.distributed.ReduceOp.SUM)
                comm_coe = comm_coe.cpu().numpy()[0] / tp_groups[0].size
                if rank == 0:
                    print('**********')
                    print('comm_coe_%d_%d_%d:'%(pp_size,tp_size,args.global_tp_consec), comm_coe)
                    print('**********')
                    path = os.path.dirname(os.path.abspath(__file__))
                    env_config_path = os.path.join(path, './hardware_configs/allreduce_bandwidth_%dnodes_%dgpus_per_node.json'%(node_num,args.nproc_per_node))
                    config = read_json_config(env_config_path) if os.path.exists(env_config_path) else dict()
                    key = 'allreduce_size_%d_consec_%d'%(tp_size,args.global_tp_consec)
                    config[key] = comm_coe # * 2 * (args.global_tp_deg - 1) / args.global_tp_deg
                    write_json_config(config, env_config_path)
                    print('Already written allreduce bandwidth into env config file %s!'%(env_config_path))
            else:
                per_comm_time = comm_time / comm_num
                per_comm_time = torch.tensor([per_comm_time]).to(device)
                torch.distributed.all_reduce(per_comm_time, group=tp_groups[0].group, op=torch.distributed.ReduceOp.SUM)
                per_comm_time = per_comm_time.cpu().numpy()[0] / tp_groups[0].size
                if rank == 0:
                    print('**********')
                    print('comm_time_%dMB_%d_%d:'%(args.local_batch_size, pp_size, tp_size), per_comm_time)
                    print('**********')
                    path = os.path.dirname(os.path.abspath(__file__))
                    env_config_path = os.path.join(path, './hardware_configs/sp_time_%dnodes_%dgpus_per_node.json'%(node_num,args.nproc_per_node))
                    config = read_json_config(env_config_path) if os.path.exists(env_config_path) else dict()
                    key = 'allreduce_size_%d_%dMB_time'%(tp_size,args.local_batch_size)
                    config[key] = per_comm_time
                    write_json_config(config, env_config_path)
                    print('Already written allreduce bandwidth into env config file %s!'%(env_config_path))
        except Exception as e:
            print(f"Profiler error: {e}")
            return
    
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],
                                schedule=torch.profiler.schedule(wait=0,warmup=1,active=10),
                                on_trace_ready=trace_handler) as p:
        for i, input in enumerate(tqdm(trainloader)):
            input = input.to(device)
            batch = [[input], [input]]
            loss = model.no_pipeline_forward_backward(batch, fake_loss_func)
            optimizer.step()
            optimizer.zero_grad()
            p.step()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--global_tp_deg", type=int, default=-1, help="Global tensor parallel degree.", choices=[-1,1,2,4,8,16,32,64,128,256],
    )
    parser.add_argument(
        "--global_tp_consec", type=int, default=-1, help="Global tensor parallel group consecutive flag."
    )
    parser.add_argument(
        "--pp_deg", type=int, default=2, help="Pipeline parallel degree.", choices=[1,2,4,8,16,32,64,128,256],
    )
    parser.add_argument(
        "--local_batch_size", type=int, default=32, help="local training batch size"
    )
    parser.add_argument(
        "--num_layers", type=int, default=24, help="Number of layers"
    )
    parser.add_argument("--local-rank" ,type=int,default=-1)
    parser.add_argument(
        "--nproc_per_node", type=int, default=-1, help="Nproc per node",
    )
    parser.add_argument(
        "--profile_time", type=int, default=0, help="Profile time",
    )
    args = parser.parse_args()
    from megatron.training.global_vars import set_args
    args.sequence_parallel = False
    args.shape_order = "SBH"
    args.use_ulysses = False
    args.async_grad_reduce = True
    set_args(args)
    train(args)