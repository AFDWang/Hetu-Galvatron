import torch
import numpy as np
import random 
import torch.distributed as dist
import torch.distributed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from galvatron.core import get_args
from galvatron.flexsp_solver import Sequence, SeqBucket
import multiprocessing as mp

global_group_set = [] #indicate wheter a group is created
group_pool = {}
is_first_iter = True
solve_process = None
mp_manager = mp.Manager()
solved_globalbatch_gps =mp_manager.list()
flexSP_optimizer = None
prev_batch = None

def solve_target(seqs, shared_globalbatch_gps):
    args = get_args()
    flexSP_optimizer.token_info(seqs)
    import time
    ignore_strategies = [32] if 'wikipedia' in args.dataset and args.seq_length <= 192000 else []
    if flexSP_optimizer.strategy == 'flexSP':
        start = time.time()
        print('Running Solver...')
        chunk_alg = 'sort_consec'
        globalbatch_groups, globalbatch_results = flexSP_optimizer.solve_flexSP_globalbatch_mp_gbmb(seqs, chunk_alg = chunk_alg, mb_option_num = 4)
        end = time.time()
        print('Solver Time Cost: %.4f'%(end-start))
    elif flexSP_optimizer.strategy == 'adaptive_bfd':
        print('Running Baseline BFD (adaptive sp size)...')
        globalbatch_groups, globalbatch_results = \
            flexSP_optimizer.homo_sp_baseline_ffd_bfd_globalbatch(seqs, 'bfd', sp_select_rule='adaptive', ignore_strategies=ignore_strategies)
    elif flexSP_optimizer.strategy == 'fix_sp_bfd':
        print(f"Running Baseline BFD (fix sp size={flexSP_optimizer.fix_sp_size})...")
        globalbatch_groups, globalbatch_results = \
            flexSP_optimizer.homo_sp_baseline_ffd_bfd_globalbatch(seqs, 'bfd', sp_select_rule='fix_sp', ignore_strategies=ignore_strategies)
    globalbatch_time = sum([results['M'] for results in globalbatch_results])
    mb_num = len(globalbatch_groups)
    print(f'Globalbatch Final Results: microbatch size = {mb_num}, Time = {globalbatch_time:.2f}')
    for idx, (groups, results) in enumerate(zip(globalbatch_groups, globalbatch_results)):
        print(f"============= Microbatch {idx}, Time: {results['M']:.2f} =============")
        for sp_size, group in groups:
            flexSP_optimizer.print_group_seqs_info(group, sp_size)
    for mbsz in globalbatch_groups:
        for _ in range(len(mbsz)):
            mbsz[_] = (mbsz[_][0], [seq.id for seq in mbsz[_][1]])
    shared_globalbatch_gps.extend(globalbatch_groups)

def convert_microbatch_res(micro_res):
    global global_group_set, group_pool
    cum_cnt = 0
    sp_group = None
    batch_indices = []
    for res_tuple in micro_res:
        sp_size, seq_id_list  = res_tuple
        rank_start = cum_cnt
        rank_end = cum_cnt + sp_size
        ranks = list(range(rank_start, rank_end))
        if tuple(ranks) not in global_group_set:
            sp_group_ = torch.distributed.new_group(ranks)
            global_group_set.append(tuple(ranks))
            if torch.distributed.get_rank() in ranks:
                group_pool[tuple(ranks)] = sp_group_
        cum_cnt += sp_size
        if torch.distributed.get_rank() in ranks:
            sp_group = group_pool[tuple(ranks)]
            batch_indices = seq_id_list
    return batch_indices, sp_group

def set_seed():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    

def collate_fn(batch):
    global is_first_iter, solve_process, solved_globalbatch_gps, prev_batch
    max_len = max([len(seq) for seq in batch])
    world_size = torch.distributed.get_world_size()
    max_len = ((max_len - 1) // world_size + 1) * world_size
    args = get_args()
    max_len = min(max_len, get_args().seq_length)
    if not get_args().use_packing:
        padded_batch = torch.zeros((len(batch), max_len), dtype=torch.long,device=batch[0].device)
        for i, seq in enumerate(batch):
            padded_batch[i, :len(seq)] = seq
        return padded_batch
    else:
        if flexSP_optimizer:
            seqs = [Sequence(sentence.shape[0], id = i) for i, sentence in enumerate(batch)]
            rank = dist.get_rank()
            # flexSP_optimizer.seqs = seqs
            if rank == 0:
                if not is_first_iter:
                    solve_process.join()
                    globalbatch_groups = list(solved_globalbatch_gps)
                    solved_globalbatch_gps = mp_manager.list()
                solve_process = mp.Process(target=solve_target, args = (seqs, solved_globalbatch_gps))
                solve_process.start()
            if is_first_iter:
                is_first_iter = False
                prev_batch = batch
                return None
            
            # flexSP_results = flexSP_optimizer.solve_flexSP_bucket_seqs(seqs, bucket_num = 10)
            # flexSP_results = flexSP_optimizer.solve_flexSP()
            torch.distributed.barrier()
            args = get_args()
            args.sp_groups = []
            microbatches = []
            n_mbatch = 0
            if rank == 0:
                micro_bsz = torch.LongTensor([len(globalbatch_groups)]).cuda()
                dist.broadcast(micro_bsz, 0)
            else:
                micro_bsz = torch.LongTensor([0]).cuda()
                dist.broadcast(micro_bsz, 0)
            # batch_indices, sp_group = flexSP_optimizer.convert_solve_res(flexSP_result)
            # note: with time limit ,solver may get different results, use broadcast to unify results
            for _  in range(micro_bsz.item()):
                n_mbatch += 1
                if rank == 0:
                    microbatch_group = globalbatch_groups[_]
                    ele_num = torch.LongTensor([len(microbatch_group)]).cuda()
                    dist.broadcast(ele_num, 0)
                    for i in range(ele_num):
                        sp_size_, seq_ids_ =  microbatch_group[i]
                        sp_size_, seq_ids_ = torch.LongTensor([sp_size_]).cuda(), torch.LongTensor(seq_ids_).cuda()
                        num_seqs = torch.LongTensor([len(seq_ids_)]).cuda()
                        dist.broadcast(sp_size_, 0)
                        dist.broadcast(num_seqs, 0)
                        dist.broadcast(seq_ids_, 0)
                        
                else:
                    microbatch_group = []
                    ele_num = torch.LongTensor([0]).cuda()
                    dist.broadcast(ele_num, 0)
                    for i in range(ele_num):
                        sp_size_, num_seqs = torch.LongTensor([0]).cuda(), torch.LongTensor([0]).cuda()
                        dist.broadcast(sp_size_, 0)
                        dist.broadcast(num_seqs, 0)
                        seq_ids_ = torch.LongTensor(num_seqs.item()).cuda()
                        dist.broadcast(seq_ids_, 0)
                        seqs_ = [j.item() for j in seq_ids_]
                        microbatch_group.append((sp_size_.item(), seqs_))
                batch_indices, sp_group = convert_microbatch_res(microbatch_group)
                m_batch = [prev_batch[idx] for idx in batch_indices]
                args.sp_groups.append(sp_group)
                # if torch.distributed.get_rank(sp_group) == 0:
                #     print(f"sp_group ranks:{torch.distributed.get_process_group_ranks(sp_group)} micro_batch_rank: {n_mbatch}, \tallocated_seqs: {seqlens}, \ttot_seqlens:{sum(seqlens)}, \tsp_size: {torch.distributed.get_world_size(sp_group)}")
                torch.distributed.barrier()
                cu_seqlens = torch.empty(len(m_batch)+1, dtype=torch.int64)
                cu_seqlens[0] = 0
                for _ in range(1, len(cu_seqlens)):
                    cu_seqlens[_] = cu_seqlens[_-1] + len(m_batch[_ - 1])
                m_batch = torch.concat(m_batch)
                microbatches.append([[m_batch, cu_seqlens]])
            prev_batch = batch
            return microbatches
        else:
            cu_seqlens = torch.empty(len(batch)+1, dtype=torch.int64)
            cu_seqlens[0] = 0
            for _ in range(1, len(cu_seqlens)):
                cu_seqlens[_] = cu_seqlens[_-1] + len(batch[_ - 1])
            batch = torch.concat(batch)
            return [batch, cu_seqlens]

def distributed_dataloader(dataset, global_bsz, shuffle = True, args = None, group = None, flexSP_optimizer_ = None):
    rank = torch.distributed.get_rank(group)
    world_size = torch.distributed.get_world_size(group)
    global flexSP_optimizer
    flexSP_optimizer = flexSP_optimizer_
    # pp_deg = args.pp_deg if args is not None and 'pp_deg' in args else 1
    # data_num_replicas = world_size // pp_deg
    train_batch_size_input = global_bsz // world_size
    trainloader = DataLoader(dataset=dataset,
                            batch_size=train_batch_size_input,
                            sampler=DistributedSampler(dataset,shuffle=shuffle,num_replicas=world_size,rank=rank), 
                            collate_fn=collate_fn)
    return trainloader

def print_loss(args, loss, ep, iter):
    if args.check_loss or args.profile:
        if loss is None:
            return
        if isinstance(loss, (list, tuple)): # Average loss of each microbatch
            if len(loss) == 0:
                return
            if isinstance(loss[0], torch.Tensor):
                loss = np.mean([l.item() for l in loss])
            else:
                loss = np.mean(loss)
        else:
            loss = loss.item() if isinstance(loss, torch.Tensor) else loss
        print('[Epoch %d] (Iteration %d): Loss = %.3f'% (ep,iter,loss))
