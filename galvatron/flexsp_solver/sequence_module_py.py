from typing import List
import numpy as np
import random

class Sequence():
    def __init__(self, seq: int, id: int = 0):
        self.seq = seq
        self.id = id
        
    def __lt__(self, other):
        return self.seq < other.seq

    def __str__(self):
        return "{}-{}".format(self.id, self.seq)

def print_seqs(seqs: List[Sequence]):
    if len(seqs) == 0:
        print('[]')
        return
    print('[', end='')
    for seq in seqs[:-1]:
        print(seq, end = ', ')
    print(seqs[-1], end=']\n')
    
def get_lens(seqs: List[Sequence]):
    return [seq.seq for seq in seqs]

class SeqBucket():
    def __init__(self, boundary: int):
        self.boundary = boundary
        self.seqs = []
        self.size = 0
        
    def add_seqs(self, seqs: List[Sequence]):
        self.size += len(seqs)
        self.seqs.extend(seqs)
    
    def random_pop_seqs(self, num: int = 1):
        if self.size == 0:
            return None
        if num > self.size:
            raise ValueError("Num cannot be greater than the bucket size.")
        indices = random.sample(range(self.size), num)
        indices.sort(reverse=True)
        poped_seqs = []
        for index in indices:
            poped_seqs.append(self.seqs.pop(index))
        self.size -= num
        return poped_seqs
    
    def print(self):
        print(f'[Bucket] Boundary: {self.boundary}, Size: {self.size}, Seqs:', end=' ')
        print_seqs(self.seqs)

def bucketing_seqs(sequences: List[Sequence], B: int):
    n = len(sequences)
    sequences_sorted = sorted(sequences)
    
    inf = float('inf')
    dp = [[inf] * (B + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    
    prefix_sum = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix_sum[i] = prefix_sum[i - 1] + sequences_sorted[i - 1].seq
    
    for b in range(1, B + 1):
        for i in range(1, n + 1):
            for j in range(i):
                cost = (sequences_sorted[i - 1].seq * (i - j) - (prefix_sum[i] - prefix_sum[j]))
                if dp[j][b - 1] + cost < dp[i][b]:
                    dp[i][b] = dp[j][b - 1] + cost
    
    avg_error = dp[n][B] / n
    
    buckets = []
    current = n
    last_boundary = float('inf')
    for b in range(B, 0, -1):
        for i in range(current):
            cost = (sequences_sorted[current - 1].seq * (current - i) - (prefix_sum[current] - prefix_sum[i]))
            if dp[i][b - 1] + cost == dp[current][b]:
                boundary = sequences_sorted[current - 1].seq
                bucket = SeqBucket(boundary)
                bucket.add_seqs(sequences_sorted[i:current])
                buckets.append(bucket)
                current = i
                break

    if current > 0:
        bucket = SeqBucket(last_boundary)
        bucket.add_seqs(sequences_sorted[:current])
        buckets.append(bucket)
    
    buckets.reverse()
    return buckets, avg_error

def chunk_globalbatch(seqs_gb: List[Sequence], mb_num: int, chunk_alg: str = 'sort_consec') -> List[List[Sequence]]:
    if chunk_alg == 'sort_consec':
        seqs_gb.sort(key=lambda x: x.seq, reverse=True)
        n = len(seqs_gb)
        prefix_sum = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix_sum[i] = prefix_sum[i - 1] + seqs_gb[i - 1].seq

        dp = [[float('inf')] * (mb_num + 1) for _ in range(n + 1)]
        partition = [[0] * (mb_num + 1) for _ in range(n + 1)]
        
        dp[0][0] = 0

        for k in range(1, mb_num + 1):
            for i in range(1, n + 1):
                for j in range(k - 1, i):
                    cost = prefix_sum[i] - prefix_sum[j]
                    if max(dp[j][k - 1], cost) < dp[i][k]:
                        dp[i][k] = max(dp[j][k - 1], cost)
                        partition[i][k] = j

        micro_batches = []
        k = mb_num
        index = n
        while k > 0:
            start_index = partition[index][k]
            micro_batches.append(seqs_gb[start_index:index])
            index = start_index
            k -= 1

        micro_batches.reverse()
    elif chunk_alg == 'distribution':
        n = len(seqs_gb)
        prefix_sum = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix_sum[i] = prefix_sum[i - 1] + seqs_gb[i - 1].seq
        dp = [[float('inf')] * (mb_num + 1) for _ in range(n + 1)]
        partition = [[0] * (mb_num + 1) for _ in range(n + 1)]
        
        dp[0][0] = 0

        for k in range(1, mb_num + 1):
            for i in range(1, n + 1):
                for j in range(k - 1, i):
                    cost = prefix_sum[i] - prefix_sum[j]
                    if max(dp[j][k - 1], cost) < dp[i][k]:
                        dp[i][k] = max(dp[j][k - 1], cost)
                        partition[i][k] = j
        micro_batches = []
        k = mb_num
        index = n
        while k > 0:
            start_index = partition[index][k]
            micro_batches.append(seqs_gb[start_index:index])
            index = start_index
            k -= 1

        micro_batches.reverse()

    return micro_batches