import galvatron.flexsp_solver
import sequence_module as seq_cpp
import sequence_module_py as seq_py

if __name__ == '__main__':
    seqs = [1024, 1536, 1762, 1428, 2479, 5728, 3739, 8729, 15273, 6482, 1726, 837, 8627, 9273, 17263, 18729, 6273, 8172, 8274, 37264]

    seqs_cpp = [seq_cpp.Sequence(seq=seq, id=id) for id, seq in enumerate(seqs)]
    seqs_py = [seq_py.Sequence(seq=seq, id=id) for id, seq in enumerate(seqs)]

    seq = seqs_cpp[0]
    print(seq.seq)
    print(seq.id)

    seq_cpp.print_seqs(seqs_cpp)
    seq_py.print_seqs(seqs_py)

    print(seq_cpp.get_lens(seqs_cpp))
    print(seq_py.get_lens(seqs_py))

    bucket_cpp = seq_cpp.SeqBucket(max(seqs))
    bucket_py = seq_py.SeqBucket(max(seqs))

    bucket_cpp.add_seqs(seqs_cpp)
    bucket_cpp.print()
    bucket_cpp.random_pop_seqs(3)
    bucket_cpp.print()
    bucket_cpp.random_pop_seqs(14)
    bucket_cpp.print()
    bucket_cpp.random_pop_seqs(3)
    bucket_cpp.print()

    bucket_py.add_seqs(seqs_py)
    bucket_py.print()
    bucket_py.random_pop_seqs(3)
    bucket_py.print()
    bucket_py.random_pop_seqs(14)
    bucket_py.print()
    bucket_py.random_pop_seqs(3)
    bucket_py.print()

    print()

    # bucketing test
    B = 10

    buckets, avg_error = seq_cpp.bucketing_seqs(seqs_cpp, B)
    for bucket in buckets:
        bucket.print()
    print("Average total error:", avg_error)
    print()
    buckets, avg_error = seq_py.bucketing_seqs(seqs_py, B)
    for bucket in buckets:
        bucket.print()
    print("Average total error:", avg_error)

    # chunk globalbatch test
    mb_num = 3
    seqs_mb_all = seq_cpp.chunk_globalbatch(seqs_cpp, mb_num, 'sort_consec')
    for seqs_mb in seqs_mb_all:
        print(sum(seq_cpp.get_lens(seqs_mb)))
        seq_cpp.print_seqs(seqs_mb)
    print()
    seqs_mb_all = seq_py.chunk_globalbatch(seqs_py, mb_num, 'sort_consec')
    for seqs_mb in seqs_mb_all:
        print(sum(seq_py.get_lens(seqs_mb)))
        seq_py.print_seqs(seqs_mb)
    print()

    seqs_mb_all = seq_cpp.chunk_globalbatch(seqs_cpp, mb_num, 'distribution')
    for seqs_mb in seqs_mb_all:
        print(sum(seq_cpp.get_lens(seqs_mb)))
        seq_cpp.print_seqs(seqs_mb)
    print()
    seqs_mb_all = seq_py.chunk_globalbatch(seqs_py, mb_num, 'distribution')
    for seqs_mb in seqs_mb_all:
        print(sum(seq_py.get_lens(seqs_mb)))
        seq_py.print_seqs(seqs_mb)
