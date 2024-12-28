import torch.distributed as dist

def init_dist_env():
    """Initialize distributed environment and return rank and world_size"""
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://"
        )
    return dist.get_rank(), dist.get_world_size()