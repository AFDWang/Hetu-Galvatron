from sequence_module import Sequence, chunk_globalbatch
# from sequence_module_py import Sequence, chunk_globalbatch
import multiprocessing as mp

def serialize_seqs(seqs):
    return [(seq.seq, seq.id) for seq in seqs]

def deserialize_seqs(seqs):
    return [Sequence(seq, id) for seq, id in seqs]

def serialize_seq_groups(groups):
    return [(sp_size, serialize_seqs(seqs)) for sp_size, seqs in groups]

def deserialize_seq_groups(groups):
    return [(sp_size, deserialize_seqs(seqs)) for sp_size, seqs in groups]

def mp_worker(seqs_mb, stop_flag, hide_alloutput, solve_flexSP_bucket_seqs, show_results, token_info, bucket_num):
    seqs_mb = deserialize_seqs(seqs_mb)
    if stop_flag.value == 1:  # 如果已经检测到None结果，就跳过
        return None
    if not hide_alloutput:
        token_info(seqs_mb)
    results = solve_flexSP_bucket_seqs(seqs_mb, bucket_num=bucket_num)
    if results is None:
        stop_flag.value = 1  # 设置标志表示有None结果
        return None
    groups = show_results(results)
    groups = serialize_seq_groups(groups)
    results = {'M': results['M']}
    return (groups, results)


def mp_microbatch_worker(seqs_gb, mb_num,
                hide_alloutput,
                solve_flexSP_bucket_seqs,
                show_results,
                token_info,
                result_dict,
                chunk_alg,
                bucket_num):
    seqs_gb = deserialize_seqs(seqs_gb)
    
    globalbatch_groups, globalbatch_results = [], []
    if not hide_alloutput:
        print(f'=========== Trying microbatch size = {mb_num} ===========')
    
    seqs_mb_all = chunk_globalbatch(seqs_gb, mb_num, chunk_alg)
    
    seqs_mb_all_serialized = [serialize_seqs(seqs_mb) for seqs_mb in seqs_mb_all]
    
    manager = mp.Manager()
    stop_flag = manager.Value('i', 0)
    
    pool = mp.Pool(processes=mb_num)
    
    async_results = [
        pool.apply_async(mp_worker, args=(
            seqs_mb_serialized, stop_flag, 
            hide_alloutput, 
            solve_flexSP_bucket_seqs, 
            show_results,
            token_info,
            bucket_num)) 
        for seqs_mb_serialized in seqs_mb_all_serialized
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
                        globalbatch_groups, globalbatch_results = None, None
                        break
                    else:
                        groups, results = result
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
    
    pool.close()
    pool.join()
    
    if feasible:
        globalbatch_time = sum([result['M'] for result in globalbatch_results])
        result_dict[mb_num] = (globalbatch_time, globalbatch_groups, globalbatch_results)
        if not hide_alloutput:
            print(f'=========== Success with microbatch size = {mb_num}, Total time = {globalbatch_time:.2f} ! ===========')
    else:
        if not hide_alloutput:
            print(f'\n=========== Failed microbatch size = {mb_num} ! ===========')
        result_dict[mb_num] = (float('inf'), None, None)