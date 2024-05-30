import torch
from einops import rearrange
# from megatron.core.parallel_state import (
#     get_global_memory_buffer,
# )

def _split_along_first_dim_with_sequence_parallel(input_, group):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""
    from galvatron.core import get_args
    args = get_args()

    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    if args.sequence_parallel:
        dim_size = list(input_.size())
        dim_size[0] = dim_size[0] * world_size
        output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        # get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
        handle = torch.distributed._all_gather_base(
            output, input_, group=group
        )
    else:
        output = input_
    
    if args.shape_order == "SBH": # [s, b, h] -> [b, s, h]
        output = rearrange(output, "s b h -> b s h")
        
    # Split along first dimension.
    dim_size = output.size()[0]
    assert dim_size % world_size == 0, \
        "First dimension of the tensor should be divisible by tensor parallel size"
    local_dim_size = dim_size // world_size
    rank = torch.distributed.get_rank(group=group)
    dim_offset = rank * local_dim_size

    if args.shape_order == "SBH": # [b, s, h] -> [s, b, h]
        output = output[dim_offset:dim_offset+local_dim_size].permute(1,0,2).contiguous()
    else:
        output = output[dim_offset:dim_offset+local_dim_size].contiguous()

    # print("split"+str(torch.cuda.current_device())+str(input_.shape)+str(output.shape))
    return output

def _gather_along_first_dim_with_sequence_parallel(input_, group):
    """Gather tensors and concatinate along the first dimension."""
    from galvatron.core import get_args
    args = get_args()
    
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    if args.shape_order == "SBH": # [s, b, h] -> [b, s, h]
        input_ = rearrange(input_, "s b h -> b s h")
        
    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype,
                         device=torch.cuda.current_device())

    torch.distributed.all_gather_into_tensor(output, input_.contiguous(),
                                       group=group)
    
    if args.shape_order == "SBH": # [s, b, h] -> [b, s, h]
        output = rearrange(output, "b s h -> s b h")
    else:
        if args.sequence_parallel:
            output = rearrange(output, "b s h -> (b s) h")
    if args.sequence_parallel:
        dim_size = output.size()[0]
        assert dim_size % world_size == 0, \
            "First dimension of the tensor should be divisible by tensor parallel size"
        local_dim_size = dim_size // world_size
        rank = torch.distributed.get_rank(group=group)
        dim_offset = rank * local_dim_size
        output = output[dim_offset:dim_offset+local_dim_size].contiguous()
    # print("gather"+str(torch.cuda.current_device())+str(input_.shape)+str(output.shape))
    return output

def _split_along_first_dim(input_, group):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""

    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along first dimension.
    dim_size = input_.size()[0]
    assert dim_size % world_size == 0, \
        "First dimension of the tensor should be divisible by tensor parallel size"
    local_dim_size = dim_size // world_size
    rank = torch.distributed.get_rank(group=group)
    dim_offset = rank * local_dim_size

    output = input_[dim_offset:dim_offset+local_dim_size].contiguous()

    return output

def _gather_along_first_dim(input_, group):
    """Gather tensors and concatinate along the first dimension."""

    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype,
                         device=torch.cuda.current_device())
    torch.distributed.all_gather_into_tensor(output, input_.contiguous(),
                                       group=group)

    return output

class _Split(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    # @staticmethod
    # def symbolic(graph, input_, group):
    #     return _split_along_first_dim(input_, group)

    @staticmethod
    def forward(ctx, input_, group, is_input):
        ctx.group = group
        ctx.is_input = is_input
        if is_input is False:
            return _split_along_first_dim(input_, group)
        else:
            return _split_along_first_dim_with_sequence_parallel(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.is_input is False:
            return _gather_along_first_dim(grad_output, ctx.group), None, None
        else:
            return _gather_along_first_dim_with_sequence_parallel(grad_output, ctx.group), None, None

class _Gather(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    # @staticmethod
    # def symbolic(graph, input_):
    #     return _gather_along_first_dim(input_)
    
    @staticmethod
    def forward(ctx, input_, group, is_input):
        ctx.group = group
        ctx.is_input = is_input
        if is_input is False:
            return _gather_along_first_dim(input_, group)
        else:
            return _gather_along_first_dim_with_sequence_parallel(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.is_input is False:
            return _split_along_first_dim(grad_output, ctx.group), None, None
        else:
            return _split_along_first_dim_with_sequence_parallel(grad_output, ctx.group), None, None

def split_to_group(input_, group, is_input):
    return _Split.apply(input_, group, is_input)

def gather_from_group(input_, group, is_input):
    return _Gather.apply(input_, group, is_input)