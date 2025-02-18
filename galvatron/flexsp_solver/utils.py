import numpy as np
from pyscipopt import Model, quicksum
import heapq

def FirstFitDecreasing(seqs, bin_capacity, bin_num=-1):
    K, P = len(seqs), bin_num
    if bin_num == -1:
        P = bin_num = K
    A = np.zeros(shape=(K, P), dtype=np.int32)
    seqs = sorted(seqs, reverse=True)
    bins = [bin_capacity] * P
    max_bin_id = -1
    for seq in seqs:
        placed = False
        # Try to place the sequence in the first bin that can accommodate it
        for p in range(P):
            if bins[p] >= seq.seq:
                bins[p] -= seq.seq  # Update the remaining capacity
                A[seq.id, p] = 1
                placed = True
                max_bin_id = max(max_bin_id, p)
                break
        if not placed:
            # print(f"Infeasible!")
            return None
    
    if max_bin_id+1 < bin_num:
        A = A[:, :max_bin_id+1]
    return A

def BestFitDecreasing(seqs, bin_capacity, bin_num=-1):
    K, P = len(seqs), bin_num
    if bin_num == -1:
        P = bin_num = K
    A = np.zeros(shape=(K, P), dtype=np.int32)
    seqs = sorted(seqs, reverse=True)
    bins = [bin_capacity] * P
    max_bin_id = -1
    for seq in seqs:
        # Find the best bin (i.e., the one that will leave the least remaining capacity)
        best_bin_idx = -1
        min_remain_capacity = bin_capacity + 1
        for p in range(P):
            if bins[p] >= seq.seq and (bins[p] - seq.seq) < min_remain_capacity:
                best_bin_idx = p
                min_remain_capacity = bins[p] - seq.seq
        
        # If a suitable bin is found, place the sequence in that bin
        if best_bin_idx != -1:
            bins[best_bin_idx] -= seq.seq  # Update the remaining capacity
            A[seq.id, best_bin_idx] = 1
            max_bin_id = max(max_bin_id, best_bin_idx)
        else:
            # print(f"Infeasible!")
            return None
    
    if max_bin_id+1 < bin_num:
        A = A[:, :max_bin_id+1]
    return A

def generate_balanced_initial_solution(model, sp_options, M, m, A, buckets, K, P, sp_size, hide_output, device_token_capacity, costmodel):
    if sp_size not in sp_options:
        return

    solution = model.createSol()

    active_groups = []
    for p in range(P):
        if sp_options[p] == sp_size:
            model.setSolVal(solution, m[p], 1)
            active_groups.append(p)
        else:
            model.setSolVal(solution, m[p], 0)

    num_active_groups = len(active_groups)
    
    if num_active_groups == 1:
        target_group = active_groups[0]
        total_execution_time = 0
        for bucket_idx, bucket in enumerate(buckets):
            total_sequences = bucket.size
            boundary = bucket.boundary

            model.setSolVal(solution, A[bucket_idx, target_group], total_sequences)
            
            total_execution_time += costmodel.total_time_single(boundary, sp_size) * total_sequences

        model.setSolVal(solution, M, total_execution_time + costmodel.compute_bias())

    else:
        group_total_lengths = {p: 0 for p in active_groups}
        group_execution_times = {p: 0 for p in active_groups}
        heap = [(0, p) for p in active_groups]
        heapq.heapify(heap)
        buckets_sorted = sorted(buckets, key=lambda b: b.boundary, reverse=True)
        for bucket_idx, bucket in enumerate(buckets_sorted):
            total_sequences = bucket.size
            boundary = bucket.boundary
            
            for _ in range(total_sequences):
                assigned = False
                tried_groups = set()
                while heap:
                    min_length, target_group = heapq.heappop(heap)
                    tried_groups.add(target_group)
                    
                    new_length = group_total_lengths[target_group] + boundary
                    new_execution_time = group_execution_times[target_group] + costmodel.total_time_single(boundary, sp_size)
                    
                    if new_length / sp_size > device_token_capacity:
                        continue
                    
                    k = buckets.index(bucket)
                    current_val = model.getSolVal(solution, A[k, target_group]) or 0

                    model.setSolVal(solution, A[k, target_group], current_val + 1)
                    
                    group_total_lengths[target_group] = new_length
                    group_execution_times[target_group] = new_execution_time

                    heapq.heappush(heap, (group_total_lengths[target_group], target_group))
                    
                    assigned = True
                    break

                if not assigned:
                    return

                heap = [(group_total_lengths[p], p) for p in active_groups if p not in tried_groups]
                heapq.heapify(heap)

        max_execution_time = max(group_execution_times.values())
        model.setSolVal(solution, M, max_execution_time + costmodel.compute_bias())

    if model.addSol(solution):
        if not hide_output:
            print("Added initialized solution with Homo-SP=%d into model!"%sp_size)

