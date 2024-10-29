import torch
import numpy as np
import argparse
import torch.distributed as dist
from typing import Any, Tuple
from torch import Tensor
from galvatron.core.comm_groups import gen_tp_group_dist

def _reduce(input_, group):
    
    if torch.distributed.get_world_size(group=group) == 1:
        return input_

    torch.distributed.all_reduce(input_.contiguous(), group = group)
    
    return input_

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
        inputs = torch.tensor(np.ones(shape=(self.b, self.s, self.h)) * rank, dtype=torch.bfloat16).to(device)
        
        # if rank == 0:
        #     print('[Rank %d]'%rank, group.ranks)
        #     print('[Rank %d]'%rank, inputs.shape, inputs)
        
        self.time_list = []
        for i in range(self.avg_iter):
            torch.cuda.synchronize()
            self.start.record()

            outputs = _reduce(inputs, group=group.group)
            
            self.end.record()
            torch.cuda.synchronize()
            iter_time = self.start.elapsed_time(self.end)
            self.time_list.append(iter_time)
        
        if rank == 0:
            # print('[Rank %d]'%rank, outputs.shape, outputs)
            print('[Rank %d]'%rank, outputs.shape)
            avg_time = np.mean(self.time_list[3:])
            print('Avg Time: %.4f ms'%avg_time, 'Bandwidth Device: %.2f GB/s'%(self.mb_size/avg_time))
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