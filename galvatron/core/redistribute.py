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

def _fused_split_allgather_along_first_dim(input_, allgather_group, split_group, fused_allgather_group, fused_split_group):
    
    if fused_split_group is not None:
        group = fused_split_group
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
        
    if fused_allgather_group is not None:
        group = fused_allgather_group
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

def _fused_split_allgather_along_first_dim_with_sequence_parallel(input_, allgather_group, split_group, fused_allgather_group, fused_split_group):
    from galvatron.core import get_args
    args = get_args()

    world_size = torch.distributed.get_world_size(group=split_group)
    # Bypass the function if we are using only 1 GPU.
    # if world_size == 1:
    #     return input_
    if args.sequence_parallel:
        dim_size = list(input_.size())
        dim_size[0] = dim_size[0] * world_size
        output_ = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        # get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
        # print(input_.shape, output_.shape,torch.cuda.current_device())
        torch.distributed.all_gather_into_tensor(
            output_, input_.contiguous(), group=split_group)
    else:
        output_ = input_.contiguous()
    
    if args.shape_order == "SBH": # [s, b, h] -> [b, s, h]
        output_ = rearrange(output_, "s b h -> b s h")
    
    if fused_split_group is not None:
        # Split along first dimension.
        world_size = torch.distributed.get_world_size(group=fused_split_group)
        dim_size = output_.size()[0]
        assert dim_size % world_size == 0, \
            "First dimension of the tensor should be divisible by tensor parallel size"
        local_dim_size = dim_size // world_size
        rank = torch.distributed.get_rank(group=fused_split_group)
        dim_offset = rank * local_dim_size

        output = output_[dim_offset:dim_offset+local_dim_size].contiguous()
        # print(rank,dim_offset,output.shape, output_.shape)

    if fused_allgather_group is not None:
        
        world_size = torch.distributed.get_world_size(group=fused_allgather_group)

        dim_size = list(output_.size())
        dim_size[0] = dim_size[0] * world_size

        output = torch.empty(dim_size, dtype=output_.dtype,
                            device=torch.cuda.current_device())
        # print(world_size,output.shape, output_.contiguous().shape,fused_allgather_group,fused_split_group)
        # print(torch.distributed.get_rank(group=fused_allgather_group),torch.cuda.current_device(),fused_allgather_group)
        # torch.distributed.barrier(group=allgather_group)
        # print("begin!",torch.cuda.current_device())
        torch.distributed.all_gather_into_tensor(output, output_.contiguous(),
                                        group=fused_allgather_group)
        # print("end!",torch.cuda.current_device())
        
    if args.shape_order == "SBH": # [b, s, h] -> [s, b, h]
        output = rearrange(output, "b s h -> s b h")
    else:
        if args.sequence_parallel:
            output = rearrange(output, "b s h -> (b s) h")
    world_size = torch.distributed.get_world_size(group=allgather_group)
    if args.sequence_parallel:
        dim_size = output.size()[0]
        assert dim_size % world_size == 0, \
            "First dimension of the tensor should be divisible by tensor parallel size"
        local_dim_size = dim_size // world_size
        rank = torch.distributed.get_rank(group=allgather_group)
        dim_offset = rank * local_dim_size
        output = output[dim_offset:dim_offset+local_dim_size].contiguous()
    # print(input_.shape, output.shape)
    return output

class _Fused_split_allgather(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_, is_input, allgather_group, split_group, fused_allgather_group, fused_split_group):
        ctx.allgather_group = allgather_group
        ctx.split_group = split_group
        ctx.fused_allgather_group = fused_allgather_group
        ctx.fused_split_group = fused_split_group
        ctx.is_input = is_input
        if is_input is False:
            return _fused_split_allgather_along_first_dim(input_, allgather_group, split_group, fused_allgather_group, fused_split_group)
        else:
            return _fused_split_allgather_along_first_dim_with_sequence_parallel(input_, allgather_group, split_group, fused_allgather_group, fused_split_group)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.is_input is False:
            return _fused_split_allgather_along_first_dim(grad_output, ctx.split_group, ctx.allgather_group, ctx.fused_split_group, ctx.fused_allgather_group), None, None, None, None, None
        else:
            return _fused_split_allgather_along_first_dim_with_sequence_parallel(grad_output, ctx.split_group, ctx.allgather_group, ctx.fused_split_group, ctx.fused_allgather_group), None, None, None, None, None

    
def fused_split_allgather(input_, is_input, allgather_group, split_group, fused_allgather_group, fused_split_group):
    return _Fused_split_allgather.apply(input_, is_input, allgather_group, split_group, fused_allgather_group, fused_split_group)