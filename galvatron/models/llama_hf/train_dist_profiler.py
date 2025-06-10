import os

import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
from transformers import LlamaConfig, LlamaForCausalLM

from galvatron.core import (
    RuntimeProfiler,
    clip_grad_norm,
    get_optimizer_and_param_scheduler,
    initialize_galvatron,
    set_megatron_args_for_dataset,
)
from galvatron.models.llama_hf.arguments import model_args
from galvatron.models.llama_hf.dataloader import (
    DataLoaderForLlama,
    get_batch,
    get_train_valid_test_data_iterators,
    loss_func,
)
from galvatron.models.llama_hf.LlamaModel_checkpoint import save_llama_module
from galvatron.models.llama_hf.LlamaModel_hybrid_parallel import get_llama_config, get_runtime_profiler, llama_model_hp
from galvatron.models.llama_hf.meta_configs import model_layer_configs, model_name
from galvatron.utils import distributed_dataloader, print_loss, set_seed, print_param_num
from megatron.training.arguments import _print_args


def train(args):

    # torch.cuda.memory._record_memory_history()
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    # torch.cuda.memory._record_memory_history(max_entries=80000)
    # if local_rank == 0:
    #     print("已启用CUDA内存记录，最大条目数: 80000")
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()

    config = get_llama_config(args)
    model = llama_model_hp(config, args)

    if local_rank == 0:
        print("Creating Dataset...")

    set_megatron_args_for_dataset(
        args, model, model.sp_groups_whole[0] if args.vocab_sp else model.tp_groups_whole[0], 
        model.vtp_data_group, model.cp_groups_whole[0])
    if local_rank == 0:
        _print_args("arguments", args)

    train_data_iterator, valid_data_iterator, test_data_iterator = get_train_valid_test_data_iterators()

    optimizer, opt_param_scheduler = get_optimizer_and_param_scheduler(model, args)

    path = os.path.dirname(os.path.abspath(__file__))
    profiler = get_runtime_profiler(args, path, config, start_iter=0)

    profiler.profile_memory(0, "After creating model")
    #print_param_num(model)
    if local_rank == 0:
        print("Start training...")

    # 创建输出目录 - 移到外层作用域
    profiler_dir = './galvatron_trace_logs'
    os.makedirs(profiler_dir, exist_ok=True)

    # 设置PyTorch profiler
    profiler_enabled = True
    torch_profiler = None
    
    if profiler_enabled:        
        # 创建独立的profile会话次数和文件名
        profile_iterations = 5  # 设置较小的迭代次数，确保完成
        
        torch_profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            # 使用简单的schedule，明确指定次数
            schedule=torch.profiler.schedule(
                wait=1,  # 第一次迭代用于预热
                warmup=1,  # 第二次迭代用于预热
                active=profile_iterations,  # 只记录5次迭代
                repeat=1),  # 不重复
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_modules=True
        )
        if local_rank == 0:
            print(f"启用PyTorch Profiler，将记录{profile_iterations}次迭代")
            print(f"Chrome Trace将保存到: {profiler_dir}")

    iterations_profiled = 0
    profiler_active = True
    
    if torch_profiler is not None:
        torch_profiler.__enter__()
        profiler_active = True

    for iter in range(args.iteration, args.train_iters):
        tokens, kwargs, loss_func = get_batch(train_data_iterator)
        profiler.profile_time_start(iter)
        profiler.profile_memory(iter, "Before Forward")

        input_ids = tokens
        batch = [input_ids]#为什么这里需要加？就是一种适配

        if profiler_active:
            with record_function(f"rank{rank}_model_forward_backward"):
                loss = model.forward_backward(batch, iter, profiler, loss_func=loss_func, **kwargs)
        else:
            loss = model.forward_backward(batch, iter, profiler, loss_func=loss_func, **kwargs)

        profiler.profile_memory(iter, "After Backward")

        if profiler_active:
            # 添加更细粒度的profiling来分析通信overlap
            with record_function(f"rank{rank}_clip_grads"):
                total_norm = clip_grad_norm(model, args.clip_grad)
            
            with record_function(f"rank{rank}_optimizer_step"):
                optimizer.step()
                opt_param_scheduler.step(increment=args.global_batch_size)
        else:
            total_norm = clip_grad_norm(model, args.clip_grad)#这是什么意思
            optimizer.step()
            opt_param_scheduler.step(increment=args.global_batch_size)#调整学习率

        profiler.profile_memory(iter, "After optimizer_step")

        optimizer.zero_grad()

        # print_loss(args, loss, ep, iter)

        profiler.post_profile_memory(iter)
        for param_group in optimizer.param_groups:
            learning_rate = param_group["lr"]
        profiler.profile_time_end(iter, loss, learning_rate, total_norm)

        # 在barrier前记录通信完成时间
        if profiler_active:
            with record_function(f"rank{rank}_distributed_barrier"):
                torch.distributed.barrier()
        else:
            torch.distributed.barrier()

        if args.save != None and (iter + 1) % args.save_interval == 0:
            save_llama_module(args.save, model, optimizer, opt_param_scheduler, iter + 1, args)
            
        # 更新torch profiler
        if profiler_active:
            torch_profiler.step()
            iterations_profiled += 1
            
            # 在合适的时间正确结束profiler
            if iterations_profiled >= profile_iterations + 2:  # wait(1) + warmup(1) + active(profile_iterations)
                if local_rank == 0:
                    print(f"已完成{iterations_profiled}次迭代的profiling，现在停止profiler")
                
                # 正确停止profiler
                torch_profiler.__exit__(None, None, None)
                
                # --- Begin Table Generation ---
                if torch_profiler: # 确保 profiler 对象存在
                    # 每个 rank 都会保存自己的文件，下面的打印语句仅由 local_rank 0 执行以简化控制台输出
                    if local_rank == 0: 
                        print(f"Rank {rank}: 分析 Profiler 数据并保存 key_averages 表格...")
                    
                    try:
                        # 1. 按输入形状分组，按 self CUDA time 排序
                        # row_limit 可以调整，默认为10，这里设置为30以获取更多信息
                        key_avg_table_shape_sorted_time = torch_profiler.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=10000)
                        table_shape_file_path = os.path.join(profiler_dir, f'{args.model_size}_table_shape_rank{rank}_{args.num_hidden_layers}_{args.seq_length}.txt')
                        with open(table_shape_file_path, 'w') as f:
                            f.write(f"Profiler Table: Grouped by Input Shape, Sorted by Self CUDA Time Total (Rank {rank})\\n")
                            f.write(key_avg_table_shape_sorted_time)
                        if local_rank == 0:
                            print(f"Profiler Table (按输入形状分组, 按 Self CUDA Time 排序, Rank {rank}) 已保存到: {table_shape_file_path}")

                        # 2. 按 self CUDA time 排序
                        key_avg_table_time_sorted = torch_profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=10000)
                        table_time_file_path = os.path.join(profiler_dir, f'{args.model_size}_table_time_rank{rank}_{args.num_hidden_layers}_{args.seq_length}.txt')
                        with open(table_time_file_path, 'w') as f:
                            f.write(f"Profiler Table: Sorted by Self CUDA Time Total (Rank {rank})\\n")
                            f.write(key_avg_table_time_sorted)
                        if local_rank == 0:
                            print(f"Profiler Table (按 Self CUDA Time 排序, Rank {rank}) 已保存到: {table_time_file_path}")

                        # 3. 按 self CUDA memory usage 排序
                        key_avg_table_mem_sorted = torch_profiler.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10000)
                        table_mem_file_path = os.path.join(profiler_dir, f'{args.model_size}_table_mem_rank{rank}_{args.num_hidden_layers}_{args.seq_length}.txt')
                        with open(table_mem_file_path, 'w') as f:
                            f.write(f"Profiler Table: Sorted by Self CUDA Memory Usage (Rank {rank})\\n")
                            f.write(key_avg_table_mem_sorted)
                        if local_rank == 0:
                            print(f"Profiler Table (按 Self CUDA Memory Usage 排序, Rank {rank}) 已保存到: {table_mem_file_path}")
                    
                    except Exception as e:
                        if local_rank == 0:
                            print(f"Rank {rank}: 生成或保存 Profiler 表格失败: {e}")
                # --- End Table Generation ---
                
                profiler_active = False
                
                # 导出trace文件
                trace_file = os.path.join(profiler_dir, f'{args.model_size}_trace_rank{rank}_{args.num_hidden_layers}_{args.seq_length}.json')
                try:
                    # 这时应当可以安全导出trace文件
                    torch_profiler.export_chrome_trace(trace_file)
                    if local_rank == 0:
                        print(f"Chrome Trace文件已保存到: {trace_file}")
                except Exception as e:
                    if local_rank == 0:
                        print(f"导出trace文件失败: {e}")
                
                torch_profiler = None


    


if __name__ == "__main__":
    args = initialize_galvatron(model_args, mode="train_dist")
    set_seed()
    train(args)
