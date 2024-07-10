import torch
from .redistribute import gather_from_group, split_to_group

class CommGroup(object):
    def __init__(self, ranks):
        assert isinstance(ranks, list) or isinstance(ranks, range), 'Rank list or range should be provided to create a CommGroup!'
        self.ranks = sorted(list(set(list(ranks))))
        self.size = len(self.ranks)
        self.group = torch.distributed.new_group(self.ranks)
    def has_rank(self, rank):
        if rank in self.ranks:
            self.intra_group_id = self.ranks.index(rank)
            return True
        return False
    def allgather(self, input, is_input):
        return gather_from_group(input, self.group, is_input)
    def split(self, input, is_input):
        return split_to_group(input, self.group, is_input)
    
    def print(self):
        print(self.ranks, end = ' ')

def show_groups(groups):
    for group in groups:
        if group is None:
            print('None', end = ' ')
        else:
            group.print()
    print()

def sort_ranks(ranks = None):
    if ranks is None:
        return None
    else:
        return sorted(list(set(list(ranks))))

def get_world_size(world_ranks = None):
    if world_ranks is None:
        return torch.distributed.get_world_size()
    else:
        assert isinstance(world_ranks, list)
        return len(world_ranks)

def get_group_rank(world_ranks = None):
    if world_ranks is None:
        return torch.distributed.get_rank()
    else:
        assert isinstance(world_ranks, list)
        return world_ranks.index(torch.distributed.get_rank())

def index_ranks(ranks, world_ranks = None):
    if world_ranks is None:
        return ranks
    else:
        assert isinstance(world_ranks, list)
        return [world_ranks[i] for i in ranks]

def gen_tp_group_dist(tp_size, pp_size, to_print = True, consecutive = True, world_ranks = None):
    world_ranks = sort_ranks(world_ranks)
    rank, world_size = torch.distributed.get_rank(), get_world_size(world_ranks)
    all_tp_groups, tp_group = [], None
    dp_size = world_size // tp_size // pp_size
    num_pp_groups = world_size // pp_size
    num_tp_groups = world_size // tp_size

    if consecutive:
        for i in range(num_tp_groups):
            ranks = range(i * tp_size, (i+1) * tp_size)
            ranks = index_ranks(ranks, world_ranks)
            group = CommGroup(ranks)
            all_tp_groups.append(group)
            if group.has_rank(rank):
                tp_group = group
    else:
        for i in range(pp_size):
            start_rank = i * num_pp_groups
            end_rank = (i + 1) * num_pp_groups
            for j in range(dp_size):
                ranks = range(start_rank + j, end_rank, dp_size)
                ranks = index_ranks(ranks, world_ranks)
                group = CommGroup(ranks)
                all_tp_groups.append(group)
                if group.has_rank(rank):
                    tp_group = group
    
    if rank == 0 and to_print:
        print("TP groups:", end = ' ')
        show_groups(all_tp_groups)
    return tp_group

def gen_dp_group_dist(tp_size, pp_size, to_print = True, consecutive = False, world_ranks = None):
    world_ranks = sort_ranks(world_ranks)
    rank, world_size = torch.distributed.get_rank(), get_world_size(world_ranks)
    all_dp_groups, dp_group = [], None
    dp_size = world_size // tp_size // pp_size
    num_pp_groups = world_size // pp_size
    num_dp_groups = world_size // dp_size

    if not consecutive:
        for i in range(pp_size):
            start_rank = i * num_pp_groups
            end_rank = (i + 1) * num_pp_groups
            for j in range(tp_size):
                ranks = range(start_rank + j, end_rank, tp_size)
                ranks = index_ranks(ranks, world_ranks)
                group = CommGroup(ranks)
                all_dp_groups.append(group)
                if group.has_rank(rank):
                    dp_group = group
    else:
        for i in range(num_dp_groups):
            ranks = range(i * dp_size, (i+1) * dp_size)
            ranks = index_ranks(ranks, world_ranks)
            group = CommGroup(ranks)
            all_dp_groups.append(group)
            if group.has_rank(rank):
                dp_group = group
    
    if rank == 0 and to_print:
        print("DP groups:", end = ' ')
        show_groups(all_dp_groups)
    return dp_group

def gen_pp_group_dist(pp_size, to_print = True, world_ranks = None):
    world_ranks = sort_ranks(world_ranks)
    rank, world_size = torch.distributed.get_rank(), get_world_size(world_ranks)
    all_pp_groups, pp_group = [], None
    num_pp_groups = world_size // pp_size
    for i in range(num_pp_groups):
        ranks = range(i, world_size, num_pp_groups)
        ranks = index_ranks(ranks, world_ranks)
        group = CommGroup(ranks)
        all_pp_groups.append(group)
        if group.has_rank(rank):
            pp_group = group

    if rank == 0 and to_print:
        print("PP groups:", end = ' ')
        show_groups(all_pp_groups)
    return pp_group

def get_tp_group_dict_dist(all_tp_sizes, pp_size, consecutive = True, world_ranks =  None):
    tp_sizes_set = list(set(all_tp_sizes))
    tp_group_dict={}
    for tp_size in tp_sizes_set:
        tp_group_dict[tp_size] = gen_tp_group_dist(tp_size, pp_size, to_print=False, consecutive=consecutive, world_ranks=world_ranks)
    return tp_group_dict

def get_dp_group_dict_dist(all_tp_sizes, pp_size, consecutive = False, world_ranks = None):
    tp_sizes_set = list(set(all_tp_sizes))
    dp_group_dict={}
    for tp_size in tp_sizes_set:
        dp_group_dict[tp_size] = gen_dp_group_dist(tp_size, pp_size, to_print=False, consecutive=consecutive, world_ranks=world_ranks)
    return dp_group_dict

def gen_redistributed_group(tp_size_old, tp_size_new, tp_consec_old, tp_consec_new, tp_group_old, tp_group_new):
    if tp_size_old == tp_size_new and tp_consec_old == tp_consec_new:
        return (None, None)
    tp_group_old = None if tp_size_old == 1 else tp_group_old
    tp_group_new = None if tp_size_new == 1 else tp_group_new
    return (tp_group_old, tp_group_new)

def merge_redistributed_group(split_group, allgather_group, world_ranks = None):
    if split_group is None or allgather_group is None:
        return None, None

    split_tp_size = split_group.size
    allgather_tp_size = allgather_group.size
    if split_group.ranks[1] - split_group.ranks[0] == 1:
        split_consecutive = 1
    else:
        split_consecutive = 0
    if allgather_group.ranks[1] - allgather_group.ranks[0] == 1:
        allgather_consecutive = 1
    else:
        allgather_consecutive = 0
    
    if split_consecutive == 0 or allgather_consecutive == 0:
        return None, None
    
    world_ranks = sort_ranks(world_ranks)
    rank, world_size = torch.distributed.get_rank(), get_world_size(world_ranks)
    

    if split_tp_size > allgather_tp_size:
        num_tp_groups = world_size // split_tp_size
        # mul = split_tp_size // allgather_tp_size
        for i in range(num_tp_groups):
            for j in range(allgather_tp_size):
                ranks = range(i * split_tp_size + j, (i+1) * split_tp_size + j, allgather_tp_size)
                group = CommGroup(ranks)
                if group.has_rank(rank):
                    fused_group = group
        return fused_group, None
    
    if split_tp_size < allgather_tp_size:
        num_tp_groups = world_size // allgather_tp_size
        # mul = allgather_tp_size // split_tp_size
        for i in range(num_tp_groups):
            for j in range(split_tp_size):
                ranks = range(i * allgather_tp_size + j, (i+1) * allgather_tp_size + j, split_tp_size)
                group = CommGroup(ranks)
                if group.has_rank(rank):
                    fused_group = group
        return None, fused_group
    
    assert False,"merge_redistributed_group error!"

def gen_comm_groups(all_tp_sizes, pp_size, tp_consecutive_flags, show_rank = -1, world_ranks = None):
    world_ranks = sort_ranks(world_ranks)
    world_size = get_world_size(world_ranks)
    world_size_per_stage = world_size // pp_size
    for i in range(len(all_tp_sizes)):
        tp_consec = tp_consecutive_flags[i]
        assert tp_consec == 0 or tp_consec == 1
        if all_tp_sizes[i] in [1, world_size_per_stage]:
            tp_consecutive_flags[i] = 1
    tp_groups, dp_groups = [], []
    dp_groups = []
    allgather_groups, split_groups = [None], [None]
    fused_split_groups, fused_allgather_groups = [None], [None]
    pp_group = gen_pp_group_dist(pp_size, to_print=False, world_ranks=world_ranks)
    tp_group_dict, dp_group_dict = {}, {}
    for consec in [0, 1]:
        tp_group_dict[consec] = get_tp_group_dict_dist(all_tp_sizes, pp_size, consec, world_ranks=world_ranks)
        dp_group_dict[consec] = get_dp_group_dict_dist(all_tp_sizes, pp_size, consec, world_ranks=world_ranks)
    for i in range(len(all_tp_sizes)):
        tp_groups.append(tp_group_dict[tp_consecutive_flags[i]][all_tp_sizes[i]])
        dp_groups.append(dp_group_dict[1-tp_consecutive_flags[i]][all_tp_sizes[i]])
    for i in range(1, len(all_tp_sizes)):
        split_group, allgather_group = gen_redistributed_group(all_tp_sizes[i-1], all_tp_sizes[i], tp_consecutive_flags[i-1], tp_consecutive_flags[i], tp_groups[i-1], tp_groups[i])
        fused_split_group, fused_allgather_group = merge_redistributed_group(split_group, allgather_group)
        allgather_groups.append(allgather_group)
        split_groups.append(split_group)
        fused_split_groups.append(fused_split_group)
        fused_allgather_groups.append(fused_allgather_group)
    
    show_rank = 0
    if show_rank >= 0 and torch.distributed.get_rank() == show_rank:
        print('====================== Galvatron Communication Group ===========================')
        print("TP groups for rank %d (all layers):"%show_rank)
        show_groups(tp_groups)
        print("DP groups for rank %d (all layers):"%show_rank)
        show_groups(dp_groups)
        print("Split groups for rank %d:"%show_rank)
        show_groups(split_groups)
        print("AllGather groups for rank %d:"%show_rank)
        show_groups(allgather_groups)
        print("Fused split groups for rank %d:"%show_rank)
        show_groups(fused_split_groups)
        print("Fused allgather groups for rank %d:"%show_rank)
        show_groups(fused_allgather_groups)
        print('================================================================================')
    return pp_group, tp_groups, dp_groups, allgather_groups, split_groups, fused_allgather_groups, fused_split_groups