import torch
import numpy as np
import argparse
import torch.distributed as dist
from typing import Any, Tuple
from torch import Tensor
from galvatron.core.comm_groups import gen_tp_group_dist

## scatter=2, gather=1 [b,s,h] scatter h gather s 
def single_all_to_all(input, scatter_idx, gather_idx, group):
    seq_world_size = dist.get_world_size(group)
    inp_shape = list(input.shape)
    inp_shape[scatter_idx] = inp_shape[scatter_idx] // seq_world_size
    if scatter_idx < 2:
        input_t = input.reshape(
            [seq_world_size, inp_shape[scatter_idx]] + \
            inp_shape[scatter_idx + 1:]
        ).contiguous()
    else:
        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        input_t = input.reshape(
            [-1, seq_world_size, inp_shape[scatter_idx]] + \
            inp_shape[scatter_idx + 1:]
        ).transpose(0, 1).contiguous()

    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=group)

    # if scattering the seq-dim, transpose the heads back to the original dimension
    if scatter_idx < 2:
        output = output.transpose(0, 1).contiguous()

    return output.reshape(
        inp_shape[: gather_idx] + \
        [inp_shape[gather_idx] * seq_world_size,] + \
        inp_shape[gather_idx + 1:]).contiguous()


class _SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor, scatter_idx: int, gather_idx: int) -> Tensor:

        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        return single_all_to_all(input, scatter_idx, gather_idx, group)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        return (None, _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx), None, None)

# def gen_sp_group(sp_size, to_print = True, consecutive = True):
#     rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
#     all_sp_groups, sp_group = [], None
#     dp_size = world_size // sp_size
#     num_sp_groups = world_size // sp_size

#     for i in range(num_sp_groups):
#         ranks = range(i * sp_size, (i+1) * sp_size)
#         group = CommGroup(ranks)
#         all_sp_groups.append(group)
#         if group.has_rank(rank):
#             tp_group = group
    
#     if rank == 0 and to_print:
#         print("SP groups:", end = ' ')
#         show_groups(all_sp_groups)
#     return tp_group

class AlltoAllTest():
    def __init__(self, args):
        self.b, self.s, self.h = args.bsz, args.seqlen, args.hidden_size
        self.local_rank = args.local_rank
        self.rank = torch.distributed.get_rank()
        self.device = torch.device("cuda", self.local_rank)
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.avg_iter = 20
        self.time_list = []
        self.mb_size = self.b * self.s * self.h * 2 / 1024 / 1024
        
    def test_all2all_perf(self, group_size):
        local_rank, rank, device = self.local_rank, self.rank, self.device
        if rank == 0:
            print('='*20, 'SP = %d'%group_size, '='*20)
            print('Total Tensor Size: %d MB, Device Tensor Size: %d MB'%(self.mb_size, self.mb_size//group_size))
            
        group = gen_tp_group_dist(group_size, 1)
        inputs = torch.tensor(np.ones(shape=(self.b, self.s//group.size, self.h)) * rank, dtype=torch.bfloat16).to(device)
        
        # if rank == 0:
        #     print('[Rank %d]'%rank, group.ranks)
        #     print('[Rank %d]'%rank, inputs.shape, inputs)
        
        self.time_list = []
        for i in range(self.avg_iter):
            torch.cuda.synchronize()
            self.start.record()

            outputs = single_all_to_all(inputs, scatter_idx=2, gather_idx=1, group=group.group)
            
            self.end.record()
            torch.cuda.synchronize()
            iter_time = self.start.elapsed_time(self.end)
            self.time_list.append(iter_time)
        
        if rank == 0:
            # print('[Rank %d]'%rank, outputs.shape, outputs)
            print('[Rank %d]'%rank, outputs.shape)
            avg_time = np.mean(self.time_list[3:])
            print('Avg Time: %.4f ms'%avg_time, 'Bandwidth Total: %.2f GB/s'%(self.mb_size/avg_time), 'Bandwidth Device: %.2f GB/s'%(self.mb_size//group_size/avg_time))
            print('='*50)

def test(args):
    torch.distributed.init_process_group(backend="nccl")
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    tests = AlltoAllTest(args)

    group_size = 2
    while group_size <= world_size:
        tests.test_all2all_perf(group_size)
        group_size *= 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bsz", type=int, default=4, 
    )
    parser.add_argument(
        "--seqlen", type=int, default=4096, 
    )
    parser.add_argument(
        "--hidden_size", type=int, default=4096, 
    )
    parser.add_argument("--local-rank" ,type=int,default=-1)
    args = parser.parse_args()
    test(args)
