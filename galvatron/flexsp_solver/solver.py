from typing import List, Dict, Union, Literal
import numpy as np
from collections import Counter
from pyscipopt import Model, quicksum, multidict
import random
import multiprocessing as mp
import galvatron.flexsp_solver
# from sequence_module_py import Sequence, SeqBucket, print_seqs, get_lens, bucketing_seqs, chunk_globalbatch
from sequence_module import Sequence, SeqBucket, print_seqs, get_lens, bucketing_seqs, chunk_globalbatch
import argparse
import time

class flexSPCostModel():
    def __init__(self, 
                 cluster_size: int = 16,
                 hidden_size: int = 4096,
                 layer_num: int = 32,
                 param_size_B: float = 7,
                 zero_stage: int = 3,
                 mixed_precision: bool = True,
                 act_per_token: float = 4.71,
                 cpt_alpha1: float = 5.128 * 1e-6,
                 cpt_alpha2: float = 183.9576 * 1e-3,
                 cpt_beta1: float = 629.3563,
                 alltoall_bandwidth_dict_gbs: Dict = {1: 1e10, 2: 154, 4: 137, 8: 121, 16: 8.7},
                 ):
        self.N = cluster_size
        self.h = hidden_size
        self.l = layer_num
        self.p = param_size_B
        self.zero_stage = zero_stage
        self.act_per_token = act_per_token
        self.cpt_alpha1 = cpt_alpha1 # * 1e-6
        self.cpt_alpha2 = cpt_alpha2 # * 1e-3
        self.cpt_beta1 = cpt_beta1
        self.zero_ratio = {
            0: 1,
            1: (6/8 * (1/self.N) + 2/8) if mixed_precision else (2/4 * (1/self.N) + 2/4),
            2: (7/8 * (1/self.N) + 1/8) if mixed_precision else (3/4 * (1/self.N) + 1/4),
            3: 1/self.N
        }[self.zero_stage]
        self.model_states_mb = param_size_B * 16 * self.zero_ratio * 1024
        # print(self.model_states_mb)
        self.alltoall_bandwidth_dict_gbs = alltoall_bandwidth_dict_gbs
        
    def check_costmodel(self, seqlen: Union[int,List[int]], sp_size: int = 1):
        if not isinstance(seqlen, List):
            seqlen = [seqlen]
        print(f'\n[Seqlen = {seqlen}, SP = {sp_size}]')
        print(f'Activation Size: {self.activation_size(seqlen, sp_size)}')
        print(f'Total Memory: {self.total_memory(seqlen, sp_size)}')
        print(f'Computation Time: {self.compute_time(seqlen, sp_size)}')
        print(f'AlltoAll Time: {self.alltoall_time(seqlen, sp_size)}')
        print(f'Total Time: {self.total_time(seqlen, sp_size)}')
        print(f'AlltoAll Ratio: {self.alltoall_time(seqlen, sp_size)/self.total_time(seqlen, sp_size)*100: .2f}')
        
    ### ======== Memory Cost Related ========
    def activation_size(self, seqlen: Union[int,List[int]], sp_size: int = 1):
        if isinstance(seqlen, List):
            seqlen = sum(seqlen)
        return self.act_per_token * seqlen / sp_size
    
    def total_memory(self, seqlen: Union[int,List[int]] = 0, sp_size: int = 1):
        return self.model_states_mb + self.activation_size(seqlen, sp_size)
    
    def memory_capacity(self, memory_limit_gb: int, seqlen: Union[int,List[int]] = 0, sp_size: int = 1):
        return memory_limit_gb * 1024 - self.total_memory(seqlen, sp_size)
    
    def token_capacity(self, memory_limit_gb: int, seqlen: Union[int,List[int]] = 0, sp_size: int = 1):
        return int((memory_limit_gb * 1024 - self.total_memory(seqlen, sp_size)) / self.act_per_token)
    
    ### ======== Time Cost Related ========
    def compute_time_single(self, seqlen: int, sp_size: int = 1): # ms
        return (self.cpt_alpha1 * (seqlen ** 2) + self.cpt_alpha2 * seqlen) / sp_size
    
    def compute_time(self, seqlen: Union[int,List[int]], sp_size: int = 1): # ms
        if not isinstance(seqlen, List):
            seqlen = [seqlen]
        cpt_times = [self.compute_time_single(seq, sp_size) for seq in seqlen]
        return sum(cpt_times) + self.cpt_beta1
    
    def compute_bias(self):
        return self.cpt_beta1
    
    def alltoall_time(self, seqlen: Union[int,List[int]] = 0, sp_size: int = 1): # ms
        if sp_size == 1:
            return 0
        if isinstance(seqlen, List):
            seqlen = sum(seqlen)
        v = self.alltoall_bandwidth_dict_gbs[sp_size]
        all2all_tensor_size = 4 * 2 * self.l * self.h * seqlen * 2 / 1024 / 1024 / sp_size
        # print(all2all_tensor_size)
        return all2all_tensor_size / v # * (sp_size - 1) / sp_size
    
    def total_time_single(self, seqlen: int, sp_size: int = 1):
        return self.compute_time_single(seqlen, sp_size) + self.alltoall_time(seqlen, sp_size)
    
    def total_time(self, seqlen: Union[int,List[int]] = 0, sp_size: int = 1):
        return self.compute_time(seqlen, sp_size) + self.alltoall_time(seqlen, sp_size)

class flexSPOptimizer():
    def __init__(self, 
                 cluster_size: int,
                 memory_limit_gb: int, 
                 costmodel: flexSPCostModel,
                 hide_scipoutput: bool = False,
                 hide_alloutput: bool = False,
                 concurrent: bool = False,
                 scip_param_dict: dict = {},
                 strategy: Literal['flexSP', 'adaptive_bfd', 'fix_sp_bfd'] = 'flexSP',
                 redist_without_empty_group: bool = False,
                #  seq_bucket_size: int = 1024,
                 ):
        self.N = cluster_size
        self.mem_limit_gb = memory_limit_gb
        self.mem_limit_mb = memory_limit_gb * 1024
        self.costmodel = costmodel
        self.device_token_capacity = self.costmodel.token_capacity(self.mem_limit_gb)
        self.cluster_token_capacity = self.device_token_capacity*self.N
        # self.group_pool = {}
        # self.global_group_set = [] #indicate wheter a group is created
        self.hide_scipoutput = hide_scipoutput or hide_alloutput
        self.hide_alloutput = hide_alloutput
        self.concurrent = concurrent
        self.scip_param_dict = scip_param_dict
        self.strategy = strategy
        self.redist_without_empty_group = redist_without_empty_group
        
    def token_info(self, seqs: List[Sequence]):
        print('\n============= Token Info =============')
        # print_seqs(seqs)
        print('Device Token Capacity: %d'%self.device_token_capacity)
        print('Cluster Token Capacity: %d'%(self.device_token_capacity*self.N))
        print('Total Token Number: %d'%sum(get_lens(seqs)))
        print('Total Sequence Number: %d'%len(seqs))

    def total_tokens(self, seqs: List[Sequence]):
        return sum(get_lens(seqs))
        
    def judge_feasibility(self, seqs, A, K, P, sp):
        if A is None:
            return -1
        M = -1
        for p in range(P):
            group_token= sum([seqs[k].seq * A[k, p] for k in range(K)]) / sp
            if group_token > self.device_token_capacity:
                return -1
            group_time = sum([self.costmodel.total_time_single(seqs[k].seq, sp) * A[k, p] for k in range(K)]) + self.costmodel.compute_bias()
            M = max(group_time, M)
        return M
        
    def solve_homo_sp_even(self, seqs: List[Sequence], sp_size: int, group_num: int):
        sp_options = [sp_size] * group_num
        K, P = len(seqs), len(sp_options)
        A = np.zeros(shape=(K, P), dtype=np.int32)
        num = int(np.ceil(K / group_num))
        for group in range(group_num):
            start = num*group
            end = num*(group+1) if group < group_num - 1 else K
            A[start:end, group] = 1
        results = None
        M = self.judge_feasibility(seqs, A, K, P, sp_size)
        if M > 0:
            results = {
                'seqs': seqs,
                'sp_options': sp_options,
                'A': A,
                'M': M,
            }
        return results
    
    def solve_homo_sp_ffd_bfd(self, seqs: List[Sequence], sp_size: int, group_num: int, type: str = 'bfd'):
        sp_options = [sp_size] * group_num
        K, P = len(seqs), len(sp_options)
        if type == 'bfd':
            from galvatron.flexsp_solver.utils import BestFitDecreasing
            A = BestFitDecreasing(seqs, self.device_token_capacity * sp_size, group_num)
        elif type == 'ffd':
            from galvatron.flexsp_solver.utils import FirstFitDecreasing
            A = FirstFitDecreasing(seqs, self.device_token_capacity * sp_size, group_num)
        results = None
        M = self.judge_feasibility(seqs, A, K, P, sp_size)
        if M > 0:
            results = {
                'seqs': seqs,
                'sp_options': sp_options,
                'A': A,
                'M': M,
            }
        return results
    
    def solve_homo_sp_lp(self, seqs: List[Sequence], sp_size: int, group_num: int, return_groups: bool = False):
        sp_options = [sp_size] * group_num
        K, P = len(seqs), len(sp_options)
    
        ### ======== Define Optimization Problem ========
        model = Model("Homo-SP LP Optimization Problem")
        
        if self.hide_scipoutput:
            model.hideOutput()
        model.setParams(self.scip_param_dict)

        # Optimization Target
        M = model.addVar(vtype="C", name="M", lb=0)  # Maximum of the execution time of SP groups
        model.setObjective(M, "minimize")
        
        # A[k,p] denotes whether the k^th sequence is put in the p^th SP group
        A = {(k, p): model.addVar(vtype="B", name=f"A_{k}_{p}") for k in range(K) for p in range(P)}

        ### ======== Constraints of LP Problem ========
        # Each sequence can only be put into one SP group
        for k in range(K):
            model.addCons(quicksum(A[k, p] for p in range(P)) == 1)
        
        for p in range(P):
            sp_size = sp_options[p]
            # SP group memory capacity limitation
            model.addCons(quicksum(seqs[k].seq * A[k, p] for k in range(K)) / sp_size <= self.device_token_capacity)

            # SP group execution time limitation
            model.addCons(quicksum(self.costmodel.total_time_single(seqs[k].seq, sp_size) * A[k, p] for k in range(K)) + self.costmodel.compute_bias() <= M)
            
            # Each SP group is occupied
            model.addCons(quicksum(A[k, p] for k in range(K)) >= 1)

        ### ======== Solve Optimization Problem ========
        if self.concurrent:
            model.solveConcurrent()
        else:
            model.optimize()
        status = model.getStatus()
        if not self.hide_alloutput:
            print("Model status:", status)
        if status not in ["optimal", "timelimit"]:
            results = None
        else:
            # Optimization Results
            results = {
                'seqs': seqs,
                'sp_options': sp_options,
                'A': np.zeros(shape=(K, P), dtype=np.int32),
                'M':model.getVal(M),
            }
            for p in range(P):
                for k in range(K):
                    results['A'][k, p] = round(model.getVal(A[k, p]))
            if return_groups:
                groups = []
                for p in range(P):
                    sp_size, group_seqs = sp_options[p], []
                    for k in range(K):
                        if results['A'][k, p] > 0:
                            group_seqs.append(seqs[k])
                    groups.append((sp_size, group_seqs, 1))
                results = groups
        return results
        
    def homo_sp_baseline_random(self, seqs: List[Sequence]):
        seqs = random.sample(seqs, len(seqs))
        baseline_results = {}
        sp = 1
        while sp <= self.N:
            baseline_results[sp] = self.solve_homo_sp_even(seqs, sp, self.N // sp)
            sp *= 2
        return baseline_results
        
    def homo_sp_baseline_ffd_bfd(self, seqs: List[Sequence], type: str = 'bfd'):
        baseline_results = {}
        sp = 1
        while sp <= self.N:
            baseline_results[sp] = self.solve_homo_sp_ffd_bfd(seqs, sp, self.N // sp, type=type)
            sp *= 2
        return baseline_results
    
    def homo_sp_baseline_lp(self, seqs: List[Sequence]):
        baseline_results = {}
        sp = 1
        while sp <= self.N:
            baseline_results[sp] = self.solve_homo_sp_lp(seqs, sp, self.N // sp)
            sp *= 2
        return baseline_results
    
    def get_sp_options(self, device_num, sp_max=0, cnt_max=0):
        sp_options = []
        sp = 1
        while sp <= device_num:
            # ignore sp=1/2/4. only use sp = 8
            if sp in [1]:
            # if sp in [1,2]:
            # if sp in [1,2,4]:
            # if False:
                sp *= 2
                continue
            if sp_max and cnt_max and sp < sp_max:
                max_num = int((device_num-(sp_max * cnt_max))//sp)
            else:
                max_num = int(device_num//sp)
            sp_options.extend([sp]*max_num)
            sp *= 2
        return sp_options
        
    def calculate_min_sp(self, seq, capacity):
        min_sp = int(np.ceil(seq / self.device_token_capacity))
        log_2 = np.log(min_sp)/np.log(2)
        # print(log_2, int(np.ceil(log_2)))
        min_sp = 2 ** int(np.ceil(log_2))
        return min_sp
        
    def get_seq_min_sp_size(self, seqs):
        return [self.calculate_min_sp(sq.seq, self.device_token_capacity) for sq in seqs]
    
    def get_bucket_min_sp_size(self, buckets):
        return [self.calculate_min_sp(bkt.boundary, self.device_token_capacity) for bkt in buckets]
        
    def get_sp_options_prune(self, seqs):
        seq_min_sp = self.get_seq_min_sp_size(seqs)
        sp_max = max(seq_min_sp)
        cnt_max = Counter(seq_min_sp)[sp_max]
        sp_options = self.get_sp_options(self.N, sp_max, cnt_max) 
        # print(f"sp_max: {sp_max}, cnt_max: {cnt_max}")
        if not self.hide_alloutput:
            print(f"seq_min_sp: {seq_min_sp}")
            print(f"sp_options: {sp_options}")
        return sp_options
        
    # No bucket
    def solve_flexSP(self, seqs: List[Sequence]):
        sp_options = self.get_sp_options_prune(seqs)
        seq_min_sp_size = self.get_seq_min_sp_size(seqs)
        
        K, P = len(seqs), len(sp_options)
        total_seq_num = K

        ### ======== Define Optimization Problem ========
        model = Model("FlexSP Optimization Problem")
        
        if self.hide_scipoutput:
            model.hideOutput()
        model.setParams(self.scip_param_dict)
        
        # Optimization Target
        M = model.addVar(vtype="C", name="M", lb=0)  # Maximum of the execution time of SP groups
        model.setObjective(M, "minimize")
        
        # A[k,p] denotes whether the k^th sequence is put in the p^th SP group
        A = {(k, p): model.addVar(vtype="B", name=f"A_{k}_{p}") for k in range(K) for p in range(P)}
        
        # m[p] denotes whether the p^th SP group is occupied
        m = {p: model.addVar(vtype="B", name=f"m_{p}") for p in range(P)}

        ### ======== Constraints of LP Problem ========
        # Each sequence can only be put into one SP group
        for k in range(K):
            model.addCons(quicksum(A[k, p] for p in range(P)) == 1)
            
            # Sequences cannot be put into group with sp size less than seq_min_sp_size
            for p in range(P):
                if sp_options[p] < seq_min_sp_size[k]:
                    model.addCons(A[k, p] == 0)
        
        for p in range(P):
            sp_size = sp_options[p]
            # SP group memory capacity limitation
            model.addCons(quicksum(seqs[k].seq * A[k, p] for k in range(K)) / sp_size <= self.device_token_capacity)

            # SP group execution time limitation
            model.addCons(quicksum(self.costmodel.total_time_single(seqs[k].seq, sp_size) * A[k, p] for k in range(K)) + self.costmodel.compute_bias() <= M)
            
            # Ensure no sequence is assigned to group p if m[p] == 0
            model.addCons(quicksum(A[k, p] for k in range(K)) <= total_seq_num * m[p])
            
            # Ensure there are sequences assigned to group p if m[p] == 1
            model.addCons(quicksum(A[k, p] for k in range(K)) >= m[p])
        
        # Device number constraint
        model.addCons(quicksum(m[p] * sp_options[p] for p in range(P)) == self.N)

        ### ======== Solve Optimization Problem ========
        if self.concurrent:
            model.solveConcurrent()
        else:
            model.optimize()
        status = model.getStatus()
        if not self.hide_alloutput:
            print("Model status:", status)
        if status not in ["optimal", "timelimit"]:
            return None
        if not self.hide_alloutput:
            print("Optimal value (minimum maximum group execution time):", model.getVal(M))
        # Optimization Results
        results = {
            'seqs': seqs,
            'sp_options': sp_options,
            'A': np.zeros(shape=(K, P), dtype=np.int32),
            'M': model.getVal(M),
            'm': np.zeros(shape=(P), dtype=np.int32),
        }
        
        for p in range(P):
            results['m'][p] = round(model.getVal(m[p]))
            for k in range(K):
                results['A'][k, p] = round(model.getVal(A[k, p]))
        return results
    
    def bucket_seqs(self, seqs: List[Sequence], bucket_num: int):
        if bucket_num > 0: # for bucket_alg=dp
            bucket_error_ths = (sum(get_lens(seqs)) + self.cluster_token_capacity)/2
            bucket_num = min(len(seqs), bucket_num)
            while True:
                buckets, avg_error = bucketing_seqs(seqs, bucket_num)
                bucket_total_token = sum([bucket.boundary * bucket.size for bucket in buckets])
                if bucket_total_token <= bucket_error_ths:
                    return buckets, avg_error, bucket_num
                bucket_num += 1
        elif bucket_num < 0: # for bucket_alg=even_dist
            bucket_num  = -bucket_num
            max_seq = max(seq.seq for seq in seqs)
            seq_chunk = max_seq // bucket_num
            buckets = []
            avg_error = 0.
            seqs.sort()
            idx = 0
            real_bucket_num = 0 
            for i in range(bucket_num):
                bound = (i + 1) * seq_chunk
                bucket = SeqBucket(bound)
                sel_seqs = []
                while idx < len(seqs) and seqs[idx].seq < bound:
                    sel_seqs.append(seqs[idx])
                    idx += 1
                if len(sel_seqs) > 0:
                    bucket.add_seqs(sel_seqs)
                    for seq in sel_seqs:
                        avg_error += (bound - seq.seq)
                    buckets.append(bucket)
                    real_bucket_num += 1
            avg_error /= len(seqs)
            return buckets, avg_error, real_bucket_num
        else:
            raise NotImplementedError()
    
    def solve_flexSP_bucket_seqs(self, seqs: List[Sequence], bucket_num: int = 10):
        sp_options = self.get_sp_options_prune(seqs)

        # bucket_num = min(len(seqs), bucket_num)
        # buckets, avg_error = bucketing_seqs(seqs, bucket_num)
        buckets, avg_error, bucket_num = self.bucket_seqs(seqs, bucket_num)
        if len(buckets) >= len(seqs):
            # print('[Warning] Fallen back into solver w.o. sequence bucketing!')
            return self.solve_flexSP(seqs)
        if not self.hide_alloutput:
            print( "Bucket Num:", bucket_num, "Average total error:", avg_error)
            for bucket in buckets:
                bucket.print()
        bucket_min_sp_size = self.get_bucket_min_sp_size(buckets)
        
        K, P = len(buckets), len(sp_options)
        total_seq_num = sum([bucket.size for bucket in buckets])

        ### ======== Define Optimization Problem ========
        model = Model("FlexSP Optimization Problem")
        
        if self.hide_scipoutput:
            model.hideOutput()
        model.setParams(self.scip_param_dict)
        
        # Optimization Target
        M = model.addVar(vtype="C", name="M", lb=0)  # Maximum of the execution time of SP groups
        model.setObjective(M, "minimize")
        
        # A[k,p] denotes the number of the sequences in k^th bucket that are put in the p^th SP group
        A = {(k, p): model.addVar(vtype="I", lb=0, name=f"A_{k}_{p}") for k in range(K) for p in range(P)}
        
        # m[p] denotes whether the p^th SP group is occupied
        m = {p: model.addVar(vtype="B", name=f"m_{p}") for p in range(P)}

        ### ======== Constraints of LP Problem ========
        # All sequences in each bucket are put into groups
        for k in range(K):
            model.addCons(quicksum(A[k, p] for p in range(P)) == buckets[k].size)
            
            # Sequences cannot be put into group with sp size less than seq_min_sp_size
            for p in range(P):
                if sp_options[p] < bucket_min_sp_size[k]:
                    model.addCons(A[k, p] == 0)
        
        for p in range(P):
            sp_size = sp_options[p]
            # SP group memory capacity limitation
            model.addCons(quicksum(buckets[k].boundary * A[k, p] for k in range(K)) / sp_size <= self.device_token_capacity)

            # SP group execution time limitation
            model.addCons(quicksum(self.costmodel.total_time_single(buckets[k].boundary, sp_size) * A[k, p] for k in range(K)) + self.costmodel.compute_bias() <= M)
            
            # Ensure no sequence is assigned to group p if m[p] == 0
            model.addCons(quicksum(A[k, p] for k in range(K)) <= total_seq_num * m[p])
            
            # Ensure there are sequences assigned to group p if m[p] == 1
            model.addCons(quicksum(A[k, p] for k in range(K)) >= m[p])
        
        # Device number constraint
        model.addCons(quicksum(m[p] * sp_options[p] for p in range(P)) == self.N)

        # Add initial feasible solution
        from galvatron.flexsp_solver.utils import generate_balanced_initial_solution
        for sp_size in list(set(sp_options)):
            generate_balanced_initial_solution(model, sp_options, M, m, A, buckets, K, P, sp_size, self.hide_alloutput, self.device_token_capacity, self.costmodel)

        ### ======== Solve Optimization Problem ========
        if self.concurrent:
            model.solveConcurrent()
        else:
            model.optimize()
        status = model.getStatus()
        if not self.hide_alloutput:
            print("Model status:", status)
        if status not in ["optimal", "timelimit"]:
            return None
        if not self.hide_alloutput:
            print("Optimal value (minimum maximum group execution time):", model.getVal(M))
        
        # Optimization Results
        results = {
            'K': K,
            'P': P,
            'sp_options': sp_options,
            'A': np.zeros(shape=(K, P), dtype=np.int32),
            'm': np.zeros(shape=(P), dtype=np.int32),
            'M': model.getVal(M),
            'seqs': seqs,
            'buckets': buckets,
        }
        
        # for k in range(K):
        #     for p in range(P):
        #         print(model.getVal(A[k, p]), end=' ')
        #     print()
        
        for p in range(P):
            results['m'][p] = round(model.getVal(m[p]))
            for k in range(K):
                results['A'][k, p] = round(model.getVal(A[k, p]))
        return results

    def get_total_time(self, seqs, sp_size, m=None):
        seqlens = get_lens(seqs)
        if m is not None:
            return sum([self.costmodel.total_time_single(seq, sp_size) for seq in seqlens]) / m + self.costmodel.compute_bias()
        else:
            return self.costmodel.total_time(seqlens, sp_size)

    def get_groups_total_token(self, groups):
        total_token = 0
        for sp_size, seqs in groups:
            total_token += sum(get_lens(seqs))
        return total_token

    def print_group_seqs_info(self, seqs, sp_size, m=None):
        seqlens = get_lens(seqs)
        if m is not None and m > 1:
            memory = self.costmodel.total_memory([sum(seqlens)/m], sp_size)
            total_time = sum([self.costmodel.total_time_single(seq, sp_size) for seq in seqlens]) / m + self.costmodel.compute_bias()
            total_token = sum(seqlens) / sp_size / m
        else:
            memory = self.costmodel.total_memory(seqlens, sp_size)
            total_time = self.costmodel.total_time(seqlens, sp_size)
            total_token = sum(seqlens) / sp_size
        if m is not None:
            print(f"SP group: SP = {sp_size}, Num = {m}, Time = {total_time:.2f}, Token Num = {total_token}, Memory = {memory:.1f}, Sequences = {seqlens}")
        else:
            print(f"SP group: SP = {sp_size}, Time = {total_time:.2f}, Token Num = {total_token}, Memory = {memory:.1f}, Sequences = {seqlens}")

    def get_min_valid_microbatch_num(self, seqs_gb: List[Sequence], chunk_alg: str = 'sort_consec'):
        total_tokens = sum(get_lens(seqs_gb))
        mb_num = int(np.ceil(total_tokens / self.cluster_token_capacity))
        while True:
            seqs_mb_all = chunk_globalbatch(seqs_gb, mb_num, chunk_alg)
            valid = True
            for seqs_mb in seqs_mb_all:
                if self.total_tokens(seqs_mb) >= self.cluster_token_capacity:
                    valid = False
                    break
            if valid:
                break
            mb_num += 1
        return mb_num

    def solve_flexSP_globalbatch(self, seqs_gb: List[Sequence], chunk_alg: str = 'sort_consec', bucket_num: int = 16):
        mb_num = self.get_min_valid_microbatch_num(seqs_gb, chunk_alg)

        globalbatch_groups, globalbatch_results = [], []
        while True:
            if not self.hide_alloutput:
                print(f'\n=========== Trying microbatch size = {mb_num} ===========')
            seqs_mb_all = chunk_globalbatch(seqs_gb, mb_num, chunk_alg)
            feasible = True
            for seqs_mb in seqs_mb_all:
                if not self.hide_alloutput:
                    self.token_info(seqs_mb)
                results = self.solve_flexSP_bucket_seqs(seqs_mb, bucket_num = bucket_num)
                # results = self.solve_flexSP_bucket_seqs_groups(seqs_mb, bucket_num = 16)
                if results is None:
                    feasible = False
                    globalbatch_groups, globalbatch_results = [], []
                    break
                groups = self.show_results(results)
                globalbatch_groups.append(groups)
                globalbatch_results.append(results)
            if feasible:
                if not self.hide_alloutput:
                    print(f'\n=========== Success with microbatch size = {mb_num} ! ===========')
                break
            if not self.hide_alloutput:
                print(f'\n=========== Failed microbatch size = {mb_num} ! ===========')
            mb_num += 1
        return globalbatch_groups, globalbatch_results

    def solve_flexSP_globalbatch_mp(self, seqs_gb: List['Sequence'], chunk_alg: str = 'sort_consec', bucket_num: int = 16):
        from galvatron.flexsp_solver.multiprocess_utils import serialize_seqs, deserialize_seq_groups, mp_worker
        mb_num = self.get_min_valid_microbatch_num(seqs_gb, chunk_alg)
        
        while True:
            globalbatch_groups, globalbatch_results = [], []
            if not self.hide_alloutput:
                print(f'=========== Trying microbatch size = {mb_num} ===========')
            
            seqs_mb_all = chunk_globalbatch(seqs_gb, mb_num, chunk_alg)
            seqs_mb_all = [serialize_seqs(seqs_mb) for seqs_mb in seqs_mb_all]

            manager = mp.Manager()
            stop_flag = manager.Value('i', 0)
            results_list = manager.list()
            
            pool = mp.Pool(processes=mb_num)

            async_results = [
                pool.apply_async(mp_worker, args=(
                    seqs_mb, stop_flag, 
                    self.hide_alloutput, 
                    self.solve_flexSP_bucket_seqs, 
                    self.show_results,
                    self.token_info,
                    bucket_num)) 
                for seqs_mb in seqs_mb_all
            ]
            
            processed_results = [False] * len(async_results)

            feasible = True
            try:
                while not all(processed_results):
                    for i, ar in enumerate(async_results):
                        if not processed_results[i] and ar.ready():
                            result = ar.get()
                            if result is None:
                                feasible = False
                                stop_flag.value = 1
                                pool.terminate()
                                pool.join()
                                globalbatch_groups, globalbatch_results = [], []
                                break
                            else:
                                groups, results = result
                                groups = deserialize_seq_groups(groups)
                                globalbatch_groups.append(groups)
                                globalbatch_results.append(results)
                                processed_results[i] = True
                    if stop_flag.value == 1:
                        feasible = False
                        break
            except Exception as e:
                pool.terminate()
                pool.join()
                raise e

            if feasible:
                if not self.hide_alloutput:
                    print(f'=========== Success with microbatch size = {mb_num} ! ===========')
                pool.close()
                pool.join()
                break

            if not self.hide_alloutput:
                print(f'\n=========== Failed microbatch size = {mb_num} ! ===========')
            
            pool.close()
            pool.join()
            mb_num += 1

        return globalbatch_groups, globalbatch_results
    
    def get_globalbatch_total_time(globalbatch_results: List[Dict]):
        total_time = 0.
        for result in globalbatch_results:
            total_time += result['M']
        return total_time
    
    def homo_sp_baseline_ffd_bfd_globalbatch_fix_sp(self, seqs_all_iter: List[List[Sequence]], type: str = 'bfd'):
        sp = 1
        for seqs in seqs_all_iter:
            sp_min, (groups, results) = self.homo_sp_baseline_ffd_bfd_globalbatch(seqs, type, sp_select_rule='min_sp')
            sp = max(sp, sp_min)
        return sp
        
    def homo_sp_baseline_ffd_bfd_globalbatch(self, seqs: List[Sequence], type: str = 'bfd', sp_select_rule: str = 'adaptive', sp_size: int = 0, fill_empty : bool = True, ignore_strategies : List = []):
        baseline_results = {}
        
        if sp_select_rule == 'fix_sp':
            if sp_size == 0:
                sp_size = self.fix_sp_size
            globalbatch_groups, globalbatch_results = self.solve_homo_sp_ffd_bfd_globalbatch(seqs, sp_size, self.N // sp_size, type=type, fill_empty=fill_empty, even_distribute=sp_size>=16)
            assert globalbatch_results is not None
            baseline_results[sp_size] = (globalbatch_groups, globalbatch_results)
        else:
            sp = 1
            sp_min = self.N + 1
            while sp <= self.N:
                if sp in ignore_strategies:
                    sp *= 2
                    continue
                globalbatch_groups, globalbatch_results = self.solve_homo_sp_ffd_bfd_globalbatch(seqs, sp, self.N // sp, type=type, fill_empty=fill_empty, even_distribute=sp>=16)
                if globalbatch_results is not None:
                    baseline_results[sp] = (globalbatch_groups, globalbatch_results)
                    sp_min = min(sp_min, sp)
                sp *= 2
        
        assert len(baseline_results) != 0
        if sp_select_rule == 'adaptive':
            sp_best, time_best = 0, 1e20
            for sp, (globalbatch_groups, globalbatch_results) in baseline_results.items():
                time = sum([results['M'] for results in globalbatch_results])
                if sp_best == 0 or time < time_best:
                    sp_best, time_best = sp, time
            return baseline_results[sp_best]
        elif sp_select_rule == 'min_sp':
            return baseline_results[sp_min]
        elif sp_select_rule == 'fix_sp':
            if sp_size == 0:
                sp_size = self.fix_sp_size
            assert sp_size in baseline_results
            return baseline_results[sp_size]
        elif sp_select_rule == 'all_sp':
            return baseline_results
    
    def solve_homo_sp_ffd_bfd_globalbatch(self, seqs_gb: List[Sequence], sp_size: int, group_num_per_mb: int, type: str = 'bfd', fill_empty: bool = True, even_distribute: bool = True):
        group_capacity = max(self.device_token_capacity * sp_size * 0.9, max(get_lens(seqs_gb)))
        if type == 'bfd':
            from galvatron.flexsp_solver.utils import BestFitDecreasing
            A = BestFitDecreasing(seqs_gb, group_capacity)
        elif type == 'ffd':
            from galvatron.flexsp_solver.utils import FirstFitDecreasing
            A = FirstFitDecreasing(seqs_gb, group_capacity)
        
        if A is None:
            return None, None
        
        group_num = A.shape[-1]
        mb_num = (group_num + group_num_per_mb - 1) // group_num_per_mb
        empty_group_num = group_num_per_mb * mb_num - group_num
        sp_options = [sp_size] * group_num_per_mb
        K, P = len(seqs_gb), len(sp_options)
        A = np.append(A, np.zeros(shape=(K, empty_group_num), dtype=np.int32), axis=1)
        
        def _token_lensum(groupA: np.ndarray) -> int:
            return sum([groupA[k] * seqs_gb[k].seq for k in range(groupA.shape[0])])

        if fill_empty:
            group_id_0, group_id_1, group_id_2 = (mb_num-1) * group_num_per_mb, mb_num * group_num_per_mb - empty_group_num, mb_num * group_num_per_mb
            if empty_group_num > 0:
                for i in range(group_id_1, group_id_2):
                    j = group_id_1 - 1
                    while j >= 0:
                        if np.sum(A[:,j]) > 1:
                            k = 0
                            while A[k,j] == 0:
                                k += 1
                            break
                        j -= 1
                    if j < 0:
                        assert False
                    A[k,j], A[k,i] = 0, 1
            
            
            even_distribute = even_distribute and (empty_group_num > 0 or self.redist_without_empty_group)
            # To evenly distribute all tokens in the last microbatch
            if even_distribute:
                mb_token_lensum = sum([_token_lensum(A[:,i]) for i in range(group_id_0, group_id_2)])
                group_token_lensum_avg = mb_token_lensum // group_num_per_mb
                for i in range(group_id_0 + 1, group_id_2):
                    if _token_lensum(A[:,i]) > group_token_lensum_avg:
                        continue
                    for j in range(group_id_0, i):
                        k = 0
                        while _token_lensum(A[:,i]) < group_token_lensum_avg and _token_lensum(A[:,j]) > group_token_lensum_avg and np.sum(A[:,j]) > 1:
                            if A[k,j] == 1 and _token_lensum(A[:,i])+seqs_gb[k].seq <= group_capacity:
                                A[k,j], A[k,i] = 0, 1
                            k += 1
        
        
        globalbatch_groups, globalbatch_results = [], []
        result = None
        feasible = True
        for i in range(mb_num):
            A_mb = A[:, i*group_num_per_mb:(i+1)*group_num_per_mb]
            M = self.judge_feasibility(seqs_gb, A_mb, K, P, sp_size)
            if M > 0:
                result = {
                    'seqs': seqs_gb,
                    'sp_options': sp_options,
                    'A': A_mb,
                    'M': M,
                }
            else:
                return None, None
            groups = []
            for j in range(group_num_per_mb):
                group_seqs = []
                for k in range(K):
                    if A_mb[seqs_gb[k].id, j] == 1:
                        group_seqs.append(seqs_gb[k])
                groups.append((sp_size, group_seqs))
            globalbatch_groups.append(groups)
            globalbatch_results.append(result)
        return globalbatch_groups, globalbatch_results

    def solve_flexSP_globalbatch_mp_gbmb(self, seqs_gb: List['Sequence'], chunk_alg: str = 'sort_consec', bucket_alg: Literal['no_bucket', 'even_dist', 'dp'] = 'dp', mb_option_num: int = 5, bucket_num: int = 16):
        from galvatron.flexsp_solver.multiprocess_utils import serialize_seqs, deserialize_seq_groups, mp_microbatch_worker

        mb_num = self.get_min_valid_microbatch_num(seqs_gb, chunk_alg)
        mb_num_options = [mb_num + i for i in range(mb_option_num)]

        seqs_gb_serialized = serialize_seqs(seqs_gb)

        manager = mp.Manager()
        result_dict = manager.dict()
        processes = []
        for mb_num in mb_num_options:
            if bucket_alg == 'dp':
                bucket_num = bucket_num
            elif bucket_alg == 'no_bucket':
                bucket_num = 1e10
            elif bucket_alg == 'even_dist':
                bucket_num = -bucket_num
            else:
                raise NotImplementedError(f"Bucketing algorithm {bucket_alg} is not implemented.")
            
            p = mp.Process(target=mp_microbatch_worker, args=(
                seqs_gb_serialized, mb_num, 
                self.hide_alloutput, 
                self.solve_flexSP_bucket_seqs, 
                self.show_results,
                self.token_info,
                result_dict,
                chunk_alg,
                bucket_num,
            ))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()

        feasible_results = []
        for mb_num, (globalbatch_time, globalbatch_groups, globalbatch_results) in result_dict.items():
            if globalbatch_groups is not None and globalbatch_results is not None:
                globalbatch_groups = [deserialize_seq_groups(groups) for groups in globalbatch_groups]
                feasible_results.append((mb_num, globalbatch_time, globalbatch_groups, globalbatch_results))
        
        if feasible_results:
            best_result = min(feasible_results, key=lambda x: x[1])
            best_mb_num, best_time, best_groups, best_results = best_result
            if not self.hide_alloutput:
                print(f'=========== Best microbatch size = {best_mb_num} with time = {best_time} ===========')
            return best_groups, best_results
        else:
            if not self.hide_alloutput:
                print('=========== All mb_num attempts failed! ===========')
            return [], []


    def show_results(self, results):
        if results is None:
            print('Infeasible!')
            return
        seqs, sp_options, A, M = results['seqs'], results['sp_options'], results['A'], results['M']
        P = len(sp_options)
        m = results['m'] if 'm' in results.keys() else np.ones(shape=(P), dtype=np.int32)
        if 'buckets' in results.keys():
            buckets = results['buckets']
            K = len(buckets)
        else:
            buckets = None
            K = len(seqs)
        
        for k in range(K):
            if buckets is None:
                assert(np.sum(A[k]) == 1)
            else:
                assert(np.sum(A[k]) == buckets[k].size)
        assert(sum([m[p]*sp_options[p] for p in range(P)])==self.N)
        
        minimized_time, groups = -1, []
        if buckets is not None:
            bucket_minimized_time, bucketing_groups = -1, []
        group_replica = False
        for p in range(P):
            if m[p] == 0:
                continue
            elif m[p] > 1:
                group_replica = True
            sp_size, group_seqs = sp_options[p], []
            group_bucket_seqs = [] if buckets is not None else None
            for k in range(K):
                if A[k,p] > 0:
                    if buckets is None:
                        group_seqs.append(seqs[k])
                    else:
                        group_seqs.extend(buckets[k].random_pop_seqs(A[k,p]))
                        for i in range(A[k, p]):
                            group_bucket_seqs.append(Sequence(seq=buckets[k].boundary, id=-1))
            groups.append((sp_size, group_seqs, m[p]))
            minimized_time = max(minimized_time, self.get_total_time(group_seqs, sp_size, m[p]))
            if buckets is not None:
                bucketing_groups.append((sp_size, group_bucket_seqs, m[p]))
                bucket_minimized_time = max(bucket_minimized_time, self.get_total_time(group_bucket_seqs, sp_size, m[p]))

        if not self.hide_alloutput:
            print(f'============= Minimized Time: {minimized_time:.2f} =============')
            for sp_size, group_seqs, m in groups:
                self.print_group_seqs_info(group_seqs, sp_size, m)

        if group_replica:
            groups_final = []
            for sp_size, group_seqs, m in groups:
                if m == 1:
                    groups_final.append((sp_size, group_seqs, m))
                else:
                    group = self.solve_homo_sp_lp(group_seqs, sp_size, m, return_groups = True)
                    if group is None:
                        assert False
                    groups_final.extend(group)
            groups = groups_final
            minimized_time = -1
            for sp_size, group_seqs, m in groups:
                minimized_time = max(minimized_time, self.get_total_time(group_seqs, sp_size, m))
            
            if not self.hide_alloutput:
                print(f'============= Minimized Time: {minimized_time:.2f} =============')
                for sp_size, group_seqs, m in groups:
                    self.print_group_seqs_info(group_seqs, sp_size, m)
        
        results['M'] = minimized_time
            
        groups = [(sp_size, group_seqs) for sp_size, group_seqs, m in groups]
        return groups

def check_costmodel():
    cluster_size = 16
    sp_size = 16
    
    costmodel = flexSPCostModel(
                 cluster_size = cluster_size,
                 hidden_size = 4096,
                 layer_num = 32,
                 param_size_B = 7, 
                 zero_stage = 3,
                 mixed_precision = True,
                 act_per_token = 4.71, 
                 cpt_alpha1 = 5.128 * 1e-6, 
                 cpt_alpha2  = 183.9576 * 1e-3, 
                 cpt_beta1 = 629.3563,
                 alltoall_bandwidth_dict_gbs = {1: 1e10, 2: 154, 4: 137, 8: 121, 16: 8, 32:6, 64:5},
                )
    
    print('\n--------------- [Single GPU Computation Time Modeling] ---------------')
    for seq in [256, 512, 1024, 2048, 4096]:
        for num in [1, 2, 4, 8, 16, 32]:
            time = costmodel.total_time([seq] * num, 1) / 1e3
            print(f'[Seq: {seq}, Num: {num}] Time: {time:.4f}')
    
    print('\n--------------- [16 GPUs EndtoEnd Time Modeling] ---------------')
    for seq in [256, 512, 1024, 2048, 4096, 8192]:
        for num in [1, 2, 4, 8, 16, 32]:
            time = costmodel.total_time([seq] * num, 16) / 1e3
            print(f'[Seq: {seq}, Num: {num}] Time: {time:.4f}')
            
    for seq in [256, 512, 1024, 2048, 4096, 8192]:
        for num in [1, 2, 4, 8, 16, 32]:
            costmodel.check_costmodel([seq] * num, 16)

def read_dataset(name = 'github', seq_limit=65536, world_size=64):
    with open('../datasets/'+name+'.txt', 'r') as file:
        lines = file.readlines()
        data = [int(l.strip('\n')) for l in lines]
    new_data = []
    idx = 0
    for d in data:
        sentence_len = d
        pad_len = ((sentence_len - 1) // (2 * world_size) + 1) * (2 * world_size) #force the length to be multiple of 2 * world_size
        if pad_len <= seq_limit - 2 * world_size:
            new_data.append(pad_len)
            idx += 1
    return new_data

def get_global_batch(data: List[int], iter: int = 0, global_batch_size: int = 64):
    return data[iter * global_batch_size: (iter + 1) * global_batch_size]


def solver_args(parser):
    group = parser.add_argument_group(title="Solver Arguments")

    group.add_argument(
        "--cluster_size", type=int, default=64,
    )
    group.add_argument(
        "--memory_limit_gb", type=int, default=28,
    )
    group.add_argument(
        "--model_size", type=str, default='gpt-7b', choices=['gpt-7b', 'gpt-13b', 'gpt-30b'],
    )
    group.add_argument(
        "--time_limit", type=int, default=5,
    )
    group.add_argument(
        "--dataset", type=str, default='github',# choices=['github', 'common_crawl', 'wikipedia'],
    )
    group.add_argument(
        "--seq_limit_k", type=int, default=128,# choices=[32, 64, 128, 256],
    )
    group.add_argument(
        "--global_batch_size", type=int, default=1024,
    )
    group.add_argument(
        "--method_type", type=str, default='adaptive', choices=['flexSP', 'adaptive', 'static']
    )
    group.add_argument(
        "--chunk_alg", type=str, default='sort_consec', choices=['sort_consec', 'distribution'],
    )
    group.add_argument(
        "--bucket_alg", type=str, default='dp', choices=['no_bucket', 'even_dist', 'dp'], help="Algorithm for FlexSP sequence bucketing"
    )
    group.add_argument(
        "--iter_num", type=int, default=30,
    )
    group.add_argument(
        "--show_strategy", type=int, default=0,
    )
    group.add_argument(
        "--start_iter", type=int, default=3,
    )
    group.add_argument(
        "--no-seq-bucket", action="store_true",
    )
    group.add_argument(
        "--redist-without-empty-group", action="store_true",
    )

    return parser

optimizer_param_dict = {
    'gpt-7b' :{
        'act_per_token' : 4.480441895,
         'cpt_alpha1' : 5.128 * 1e-6,
         'cpt_alpha2' : 183.9576 * 1e-3,
         'cpt_beta1' : 629.3563,
         'hidden_size': 4096,
         'layer_num': 32,
         'param_size_B': {4: 6.51, 192: 7.1174468994140625, 384: 7.8498687744140625},
    },
    'gpt-13b' : {
        'act_per_token': 4.220439453,
         'cpt_alpha1' : 9.2852 * 1e-6,
         'cpt_alpha2' : 306.0189 * 1e-3,
         'cpt_beta1' : 1132.5632,
         'hidden_size': 5120,
         'layer_num': 40,
         'param_size_B': {4: 12.3568, 192: 13.11605453491211, 384: 14.03158187866211},
    },
    'gpt-30b' : {
        'act_per_token' : 3.417381836,
         'cpt_alpha1' : 15.4262 * 1e-6,
         'cpt_alpha2' : 803.5742 * 1e-3,
         'cpt_beta1' : 2789.3644,
         'hidden_size': 6656,
         'layer_num': 60,
         'param_size_B': {4: 30.538, 192: 31.52513885498047, 384: 32.71532440185547},
    }
}


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser = solver_args(parser)
    args = parser.parse_args()
    model_config = optimizer_param_dict[args.model_size]
    
    random.seed(0)
    
    cluster_size = args.cluster_size
    memory_limit_gb = args.memory_limit_gb

    # alltoall_bandwidth_dict_gbs = {1: 1e10, 2: 154, 4: 137, 8: 121, 16: 8, 32:6, 64:5}
    # alltoall_bandwidth_dict_gbs = {1: 1e10, 2: 105.24, 4: 94.05, 8: 86.34, 16: 19.12, 32:11.01, 64:75} # 2k seq
    # alltoall_bandwidth_dict_gbs = {1: 1e10, 2: 119.54, 4: 104.07, 8: 96.5, 16: 20.01, 32:10.53, 64:7.77} # 4k seq
    alltoall_bandwidth_dict_gbs = {1: 1e10, 2: 119.54, 4: 104.07, 8: 96.5, 16: 10.33, 32:5.94, 64:4.87} # 4k seq, 2 IB

    param_sizes = model_config['param_size_B']
    seq_limit_k = args.seq_limit_k
    if seq_limit_k in param_sizes.keys():
        param_size_B = param_sizes[seq_limit_k]
    else:
        param_size_B = (param_sizes[384]-param_sizes[192])/192*(seq_limit_k-192)+param_sizes[192]
    print('Model Size: ', param_size_B)
    
    act_per_token = model_config['act_per_token']
    if args.model_size == 'gpt-7b' and cluster_size in [16, 32]:
        act_per_token = {
            16: 2.668764648,
            # 16: 4.480441895,
            32: 4.480441895,
        }[cluster_size]

    costmodel = flexSPCostModel(
                 cluster_size = cluster_size,
                 hidden_size = model_config['hidden_size'],
                 layer_num = model_config['layer_num'],
                 param_size_B = param_size_B, 
                 zero_stage = 3,
                 mixed_precision = True,
                 act_per_token = act_per_token, 
                 cpt_alpha1 = model_config['cpt_alpha1'], 
                 cpt_alpha2  = model_config['cpt_alpha2'], 
                 cpt_beta1 = model_config['cpt_beta1'],
                 alltoall_bandwidth_dict_gbs = alltoall_bandwidth_dict_gbs,
                )

    param_dict = {
                "limits/time": args.time_limit,
                }
    flexSP_optimizer = flexSPOptimizer(
                 cluster_size = cluster_size,
                 memory_limit_gb = memory_limit_gb, 
                 costmodel = costmodel,
                 hide_scipoutput = True,
                 hide_alloutput = True,
                 concurrent = False,
                 scip_param_dict = param_dict,
                 strategy = 'adaptive_bfd',
                 redist_without_empty_group = args.redist_without_empty_group,
    )

    print(f'Cluster Size: {cluster_size}, Model Size: {args.model_size}, Memory: {memory_limit_gb}')
    flexSP_optimizer.token_info([])
    print('='*20)
    print()
    # exit(0)

    dataset_name = args.dataset
    global_batch_size = args.global_batch_size
    seq_limit = args.seq_limit_k * 1000
    data = read_dataset(dataset_name, seq_limit = seq_limit)
    
    ignore_strategies = [32] if 'wikipedia' in args.dataset and args.seq_limit_k <= 192 else []
    
    # dataset analysis
    total_token = 0
    max_seq, max_seq_iter = -1, -1
    for iter in range(args.start_iter, args.start_iter+args.iter_num):
        sequences = get_global_batch(data, iter, global_batch_size)
        batch_token = sum(sequences)
        total_token += batch_token
        if max(sequences) > max_seq:
            max_seq = max(sequences)
            max_seq_iter = iter
    sequences = [Sequence(seq=seq, id=id) for id, seq in enumerate([max_seq] * (flexSP_optimizer.cluster_token_capacity//max_seq))]
    globalbatch_groups, globalbatch_results = flexSP_optimizer.homo_sp_baseline_ffd_bfd_globalbatch(sequences, 'bfd', sp_select_rule='min_sp', fill_empty=False, ignore_strategies=ignore_strategies)
    min_sp_size, _ = globalbatch_groups[0][0] if len(globalbatch_groups) and len(globalbatch_groups[0]) else (-1,-1)
    print('Dataset %s: Avg Tokens per Batch = %.3f Million, Max Seq = %d (%d), Min-SP = %d'%(args.dataset, total_token/args.iter_num/1024/1024, max_seq, max_seq_iter, min_sp_size))
    print('Maxseq: %d'%args.seq_limit_k, 'Total tokens: %d'%total_token)
    print()
    # exit(0)
    
    if args.method_type == 'static':
        print(f"--------------- [Baseline BFD Homo-SP] ---------------")
        total_time = 0.
        fix_sp_size = 64 # min_sp_size
        for iter in range(args.start_iter, args.start_iter+args.iter_num):
            sequences = get_global_batch(data, iter, global_batch_size)
            sequences = [Sequence(seq=seq, id=id) for id, seq in enumerate(sequences)]
            globalbatch_groups, globalbatch_results = flexSP_optimizer.homo_sp_baseline_ffd_bfd_globalbatch(sequences, 'bfd', sp_select_rule='fix_sp', sp_size=fix_sp_size, ignore_strategies=ignore_strategies)
            globalbatch_time = sum([results['M'] for results in globalbatch_results])
            total_time += globalbatch_time
            print(f'Iteration[{iter}] Final Results: sp size = {fix_sp_size}, # microbatch = {len(globalbatch_groups)}, Time = {globalbatch_time:.2f}', flush=True)
            if args.show_strategy:
                for idx, (groups, results) in enumerate(zip(globalbatch_groups, globalbatch_results)):
                    microbatch_token = flexSP_optimizer.get_groups_total_token(groups)
                    total_token = flexSP_optimizer.cluster_token_capacity
                    print(f"============= Microbatch {idx}, Time: {results['M']:.2f}, Total Token: {microbatch_token} / {total_token} =============")
                    for sp_size, group in groups:
                        flexSP_optimizer.print_group_seqs_info(group, sp_size)
                print()
        print(f"--------------- [Baseline BFD Homo-SP] ---------------")
        print(f"Total Time for {args.iter_num} iterations: {total_time / 1000.:.2f} s\n")
    
    if args.method_type == 'adaptive':
        print(f"--------------- [Baseline BFD Homo-SP (adaptive across global batches)] ---------------")
        total_time = 0.
        for iter in range(args.start_iter, args.start_iter+args.iter_num):
            sequences = get_global_batch(data, iter, global_batch_size)
            sequences = [Sequence(seq=seq, id=id) for id, seq in enumerate(sequences)]
            globalbatch_groups, globalbatch_results = flexSP_optimizer.homo_sp_baseline_ffd_bfd_globalbatch(sequences, 'bfd', sp_select_rule='adaptive', ignore_strategies=ignore_strategies)
            sp_size, _ = globalbatch_groups[0][0]
            globalbatch_time = sum([results['M'] for results in globalbatch_results])
            total_time += globalbatch_time
            print(f'Iteration[{iter}] Final Results: sp size = {sp_size}, # microbatch = {len(globalbatch_groups)}, Time = {globalbatch_time:.2f}', flush=True)
            if args.show_strategy:
                for idx, (groups, results) in enumerate(zip(globalbatch_groups, globalbatch_results)):
                    microbatch_token = flexSP_optimizer.get_groups_total_token(groups)
                    total_token = flexSP_optimizer.cluster_token_capacity
                    print(f"============= Microbatch {idx}, Time: {results['M']:.2f}, Total Token: {microbatch_token} / {total_token} =============")
                    for sp_size, group in groups:
                        flexSP_optimizer.print_group_seqs_info(group, sp_size)
                print()
        print(f"--------------- [Baseline BFD Homo-SP (adaptive across global batches)] ---------------")
        print(f"Total Time for {args.iter_num} iterations: {total_time / 1000.:.2f} s\n")

    if args.method_type == 'flexSP':
        print(f"--------------- [FlexSP] ---------------")
        total_time = 0.
        for iter in range(args.start_iter, args.start_iter+args.iter_num):
            sequences = get_global_batch(data, iter, global_batch_size)
            sequences = [Sequence(seq=seq, id=id) for id, seq in enumerate(sequences)]
            chunk_alg = args.chunk_alg
            bucket_alg = args.bucket_alg
            bucket_num = 16
            if args.no_seq_bucket:
                bucket_num = 1e10 # bucket num set as inf, fallen back into solver without sequence bucketing
            start = time.time()
            globalbatch_groups, globalbatch_results = flexSP_optimizer.solve_flexSP_globalbatch_mp_gbmb(sequences, bucket_alg=bucket_alg, chunk_alg = chunk_alg, mb_option_num = 5, bucket_num = bucket_num)
            end = time.time()

            globalbatch_time = sum([results['M'] for results in globalbatch_results])
            total_time += globalbatch_time
            print(f'Iteration[{iter}] Final Results: Search Time = {end-start:.4f}, # microbatch = {len(globalbatch_groups)}, Time = {globalbatch_time:.2f}', flush=True)
            if args.show_strategy:
                for idx, (groups, results) in enumerate(zip(globalbatch_groups, globalbatch_results)):
                    microbatch_token = flexSP_optimizer.get_groups_total_token(groups)
                    total_token = flexSP_optimizer.cluster_token_capacity
                    print(f"============= Microbatch {idx}, Time: {results['M']:.2f}, Total Token: {microbatch_token} / {total_token} =============")
                    for sp_size, group in groups:
                        flexSP_optimizer.print_group_seqs_info(group, sp_size)
                print()
        print(f"--------------- [FlexSP] ---------------")
        print(f"Total Time for {args.iter_num} iterations: {total_time / 1000.:.2f} s\n")
        
    exit(0)

    # test
    sequences = get_global_batch(data, 4, global_batch_size)
    sequences = [Sequence(seq=seq, id=id) for id, seq in enumerate(sequences)]   
    start = time.time()
    chunk_alg = args.chunk_alg
    bucket_num = 16
    if args.no_seq_bucket:
        bucket_num = 1e10 # bucket num set as inf, fallen back into solver without sequence bucketing
    # globalbatch_groups, globalbatch_results = flexSP_optimizer.solve_flexSP_globalbatch(sequences, chunk_alg = chunk_alg, bucket_num = bucket_num)
    # globalbatch_groups, globalbatch_results = flexSP_optimizer.solve_flexSP_globalbatch_mp(sequences, chunk_alg = chunk_alg, bucket_num = bucket_num)
    globalbatch_groups, globalbatch_results = flexSP_optimizer.solve_flexSP_globalbatch_mp_gbmb(sequences, chunk_alg = chunk_alg, mb_option_num = 5, bucket_num = bucket_num)
    end = time.time()
    print('Time cost: %.4f'%(end-start))

    globalbatch_time = sum([results['M'] for results in globalbatch_results])
    print(f'\n\nGlobalbatch Final Results: # microbatch = {len(globalbatch_groups)}, Time = {globalbatch_time:.2f}')
    for idx, (groups, results) in enumerate(zip(globalbatch_groups, globalbatch_results)):
        microbatch_token = flexSP_optimizer.get_groups_total_token(groups)
        total_token = flexSP_optimizer.cluster_token_capacity
        print(f"============= Microbatch {idx}, Time: {results['M']:.2f}, Total Token: {microbatch_token} / {total_token} =============")
        for sp_size, group in groups:
            flexSP_optimizer.print_group_seqs_info(group, sp_size)

    exit(0)

    # sequences = sorted(sequences, reverse=True)
    # sequences = sequences[-68:]


    # baseline_results_random = flexSP_optimizer.homo_sp_baseline_random(sequences)
    # baseline_results_lp = flexSP_optimizer.homo_sp_baseline_lp(sequences)
    # baseline_results_bfd = flexSP_optimizer.homo_sp_baseline_ffd_bfd(sequences, type='bfd')
    # flexSP_results = flexSP_optimizer.solve_flexSP(sequences)
    flexSP_results = flexSP_optimizer.solve_flexSP_bucket_seqs(sequences, bucket_num=10)
    # flexSP_results = flexSP_optimizer.solve_flexSP_bucket_seqs_groups(sequences, bucket_num=12)
    
    # for sp, results in baseline_results_random.items():
    #     print('\n--------------- [Baseline Random Homo-SP = %d] ---------------'%sp)
    #     flexSP_optimizer.show_results(results)
    # for sp, results in baseline_results_lp.items():
    #     print('\n--------------- [Baseline LP Homo-SP = %d] ---------------'%sp)
    #     flexSP_optimizer.show_results(results)
    # for sp, results in baseline_results_bfd.items():
    #     print('\n--------------- [Baseline BFD Homo-SP = %d] ---------------'%sp)
    #     flexSP_optimizer.show_results(results)
    print('\n--------------- [flexSP Results] ---------------')
    groups = flexSP_optimizer.show_results(flexSP_results)
    
    print('\n============= Final Results =============')
    # for sp, results in baseline_results_random.items():
    #     if results is not None:
    #         print('Baseline Random Homo-SP = %d: Minimized Time = %.2f'%(sp, results['M']))
    # for sp, results in baseline_results_lp.items():
    #     if results is not None:
    #         print('Baseline LP Homo-SP = %d: Minimized Time = %.2f'%(sp, results['M']))
    # for sp, results in baseline_results_bfd.items():
    #     if results is not None:
    #         print('Baseline BFD Homo-SP = %d: Minimized Time = %.2f'%(sp, results['M']))
    if flexSP_results is not None:
        print('flexSP: Minimized Time = %.2f'%(flexSP_results['M']))
        
    flexSP_optimizer.token_info(sequences)
    
    # for sp_size, seqs in groups:
    #     print(sp_size)
    #     print_seqs(seqs)
