import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import functools
from typing import (
    Any,
    Callable,
    List,
    Optional,
    no_type_check
)

from torch.distributed.fsdp._common_utils import (
    HandleTrainingState,
    TrainingState,
    _FSDPState,
)

if torch.__version__ >= '2.5.0':
    from torch.distributed.fsdp._flat_param import (
        FlatParameter,
        FlatParamHandle,
        HandleShardingStrategy,
        HandleTrainingState,
        RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES,
)
else:
    from torch.distributed.fsdp.flat_param import (
        FlatParameter,
        FlatParamHandle,
        HandleShardingStrategy,
        HandleTrainingState,
        RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES,
    )

from .sp_grad_reduce import _post_backward_hook_sp as _post_backward_hook
from torch.distributed.fsdp._runtime_utils import (
    _post_backward_final_callback,
    _unshard,
)

from torch.distributed.utils import (
    _p_assert,
)

from ..utils import rgetattr, rhasattr

def _send_backward_hook(
    input_tensor_grad: List[torch.Tensor],
    position: int,
    send_backward_partial: Callable,
    check_finish_partial: Callable,
    grad_output: Any,
)-> None:
    input_tensor_grad[position] = grad_output
    if check_finish_partial():
        send_backward_partial(input_tensor_grad)

def fsdp_reduce_gradients(model):
    for m in model.modules():
        if isinstance(m, FSDP):
            m.training_state = TrainingState.FORWARD_BACKWARD
            if hasattr(m, "_handles"):
                for handle in m._handles:
                    handle._training_state = HandleTrainingState.BACKWARD_PRE
                    _unshard(m, m._handles, m._streams["unshard"], m._streams["pre_unshard"])
                    _post_backward_hook(m, handle, None)
            else:
                if m._handle != None:
                    m._handle._training_state = HandleTrainingState.BACKWARD_PRE
                    _unshard(m, m._handle, m._unshard_stream, m._pre_unshard_stream)
                    _post_backward_hook(m, m._handle, None)

    for m in model.modules():
        if isinstance(m, FSDP) and m._is_root:
            _post_backward_final_callback(m, m)
            
@torch.no_grad()
def _allreduce_word_embedding_no_pipeline(wte_model, wte_attr_name, lmhead_model, lmhead_attr_name):
    wte = rgetattr(wte_model.module, wte_attr_name)
    lmhead = rgetattr(lmhead_model.module, lmhead_attr_name)
    if hasattr(wte, "_handles"):
        for wte_handle, lmhead_handle in zip(wte._handles, lmhead._handles):
            assert wte_handle.flat_param.data is not None
            assert lmhead_handle.flat_param.data is not None
            wte_handle.flat_param.data.copy_((wte_handle.flat_param.data + lmhead_handle.flat_param.data) / 2)
            lmhead_handle.flat_param.data.copy_((wte_handle.flat_param.data + lmhead_handle.flat_param.data) / 2)
    else:
        assert wte._handle.flat_param.data is not None
        assert lmhead._handle.flat_param.data is not None
        wte._handle.flat_param.data.copy_((wte._handle.flat_param.data + lmhead._handle.flat_param.data) / 2)
        lmhead._handle.flat_param.data.copy_((wte._handle.flat_param.data + lmhead._handle.flat_param.data) / 2)
        
# For Finalization of Model Gradients
@torch.no_grad()
def _allreduce_word_embedding(module, tied_wte_attr_name, group):
    word_embedding = rgetattr(module.module, tied_wte_attr_name)
    if hasattr(word_embedding, "_handles"):
        for handle in word_embedding._handles:
            assert handle.flat_param.data is not None
            dist.all_reduce(handle.flat_param.data, group=group)
    else:
        assert word_embedding._handle.flat_param.data is not None
        dist.all_reduce(word_embedding._handle.flat_param.data, group=group)
        
@torch.no_grad()
def _allreduce_word_embedding_grads_no_pipeline(wte_model, wte_attr_name, lmhead_model, lmhead_attr_name):
    wte = rgetattr(wte_model.module, wte_attr_name)
    lmhead = rgetattr(lmhead_model.module, lmhead_attr_name)
    if hasattr(wte, "_handles"):
        for wte_handle, lmhead_handle in zip(wte._handles, lmhead._handles):
            assert wte_handle.flat_param.grad is not None
            assert lmhead_handle.flat_param.grad is not None
            wte_handle.flat_param.grad.copy_((wte_handle.flat_param.grad + lmhead_handle.flat_param.grad) / 2)
            lmhead_handle.flat_param.grad.copy_((wte_handle.flat_param.grad + lmhead_handle.flat_param.grad) / 2)
    else:
        assert wte._handle.flat_param.grad is not None
        assert lmhead._handle.flat_param.grad is not None
        wte._handle.flat_param.grad.copy_((wte._handle.flat_param.grad + lmhead._handle.flat_param.grad) / 2)
        lmhead._handle.flat_param.grad.copy_((wte._handle.flat_param.grad + lmhead._handle.flat_param.grad) / 2)
        
# For Finalization of Model Gradients
@torch.no_grad()
def _allreduce_word_embedding_grads(module, tied_wte_attr_name, group):
    word_embedding = rgetattr(module.module, tied_wte_attr_name)
    if hasattr(word_embedding, "_handles"):
        for handle in word_embedding._handles:
            assert handle.flat_param.grad is not None
            dist.all_reduce(handle.flat_param.grad, group=group)
    else:
        assert word_embedding._handle.flat_param.grad is not None
        dist.all_reduce(word_embedding._handle.flat_param.grad, group=group)

# ================ FSDP Async Reduce Gradient Utils ================
# Only Available on PyTorch 2.0
# from torch import nn
# import torch.nn.functional as F
# from typing import Any, no_type_check, Optional, List, Union
# from torch.distributed.fsdp._common_utils import (
#     _assert_in_training_states,
#     _FSDPState,
#     TrainingState,
# )
# from torch.distributed.fsdp._runtime_utils import (
#     _check_comm_hook, 
#     _low_precision_hook_enabled, 
#     _check_grad_to_accumulate, 
#     _cast_grad_to_param_dtype,
#     _should_free_in_backward,
#     _reshard,
#     _post_backward_final_callback
# )
# from torch.distributed.fsdp.flat_param import (
#     FlatParamHandle,
#     HandleShardingStrategy,
# )
# from torch.distributed.algorithms._comm_hooks import default_hooks
# from torch.distributed.fsdp._utils import (
#     _no_dispatch_record_stream,
#     p_assert
# )
# def fsdp_reduce_gradients(model):
#     for m in model.modules():
#         if isinstance(m, FSDP):
#             m.training_state = TrainingState.FORWARD_BACKWARD
#             for handle in m._handles:
#                 flat_param = handle.flat_param
#                 if flat_param.requires_grad:
#                     _fsdp_reduce_scatter_gradients(m, handle)
#     fsdp_post_backward_callback(model)

# def fsdp_post_backward_callback(model):
#     for m in model.modules():
#         if isinstance(m, FSDP) and m._is_root:
#             _post_backward_final_callback(m, model)

# def disable_post_backward_callback(model):
#     for m in model.modules():
#         if isinstance(m, FSDP):
#             if m._is_root and not m._post_backward_callback_queued:
#                 m._post_backward_callback_queued = True

def enter_no_sync_context(model):
    if isinstance(model, FSDP):
        model.no_sync_context = model.no_sync()
        model.no_sync_context.__enter__()
    elif isinstance(model, nn.Sequential):
        for block in model:
            for m in block.modules():
                if isinstance(m, FSDP):
                    m.no_sync_context = m.no_sync()
                    m.no_sync_context.__enter__()
                    break

def exit_no_sync_context(model):
    if isinstance(model, FSDP):
        model.no_sync_context.__exit__(None, None, None)
    elif isinstance(model, nn.Sequential):
        for block in model:
            for m in block.modules():
                if isinstance(m, FSDP) and hasattr(m, 'no_sync_context'):
                    m.no_sync_context.__exit__(None, None, None)
                    break

# @no_type_check
# @torch.no_grad()
# def _fsdp_reduce_scatter_gradients(
#     state: _FSDPState,
#     handle: FlatParamHandle,
#     *unused: Any,
# ):
#     """
#     Reduce-scatters the gradient of ``handle`` 's ``FlatParameter``.

#     Precondition: The ``FlatParameter`` 's ``.grad`` attribute contains the
#     unsharded gradient for the local batch.

#     Postcondition:
#     - If using ``NO_SHARD``, then the ``.grad`` attribute is the reduced
#     unsharded gradient.
#     - Otherwise, the ``_saved_grad_shard`` attribute is the reduced sharded
#     gradient (accumulating with any existing gradient).
#     """
#     flat_param = handle.flat_param
#     flat_param._post_backward_called = True
#     with torch.autograd.profiler.record_function(
#         "FullyShardedDataParallel._fsdp_reduce_scatter_gradients"
#     ):
#         # This function is only used for reduce-scattering gradients, and is called manually 
#         # after backward. So we don't need to check the training state or reshard parameters.
        
#         # _assert_in_training_states(state, [TrainingState.FORWARD_BACKWARD])
#         # # For multiple applications of reentrant AC across submodules sharing
#         # # the same `FlatParameter`, the post-backward hook may run multiple
#         # # times in one backward, in which case we permit the state to already
#         # # be in `BACKWARD_POST`.
#         # p_assert(
#         #     handle._training_state
#         #     in (HandleTrainingState.BACKWARD_PRE, HandleTrainingState.BACKWARD_POST),
#         #     f"Expects `BACKWARD_PRE` or `BACKWARD_POST` state but got {handle._training_state}",
#         # )
#         # handle._training_state = HandleTrainingState.BACKWARD_POST

#         if flat_param.grad is None:
#             return
#         if flat_param.grad.requires_grad:
#             raise RuntimeError("FSDP does not support gradients of gradients")

#         # free_unsharded_flat_param = _should_free_in_backward(state, handle)
#         # _reshard(state, [handle], [free_unsharded_flat_param])

#         # # TODO: Post-backward prefetching does not support the multiple handles
#         # # per module case since the post-backward hook runs per handle, not per
#         # # group of handles.
#         # handles_key = (handle,)
#         # _prefetch_handles(state, handles_key)

#         # if not state._sync_gradients:
#         #     if handle._use_orig_params:
#         #         handle._use_unsharded_grad_views()
#         #     return

#         # Wait for all ops in the current stream (e.g. gradient
#         # computation) to finish before reduce-scattering the gradient
#         state._streams["post_backward"].wait_stream(torch.cuda.current_stream())

#         with torch.cuda.stream(state._streams["post_backward"]):
#             autograd_computed_grad = flat_param.grad.data
#             if state._exec_order_data.is_first_iter:  # only check once
#                 _check_comm_hook(
#                     state._communication_hook, state._communication_hook_state
#                 )
#             if (
#                 not _low_precision_hook_enabled(state)
#                 and flat_param.grad.dtype != handle._reduce_dtype
#             ):
#                 flat_param.grad.data = flat_param.grad.to(handle._reduce_dtype)

#             if handle.uses_sharded_strategy:
#                 # We clear `.grad` to permit multiple backwards. This avoids a
#                 # race where the second backward pass computation precedes
#                 # ahead of the first backward pass reduction, which is possible
#                 # since the reduction is issued in a separate stream and is
#                 # async and would result in reducing the wrong gradient.
#                 unsharded_grad = flat_param.grad.data
#                 flat_param.grad = None
#                 chunks = list(unsharded_grad.chunk(state.world_size))
#                 numel_to_pad = (
#                     state.world_size * chunks[0].numel() - unsharded_grad.numel()
#                 )
#                 padded_unsharded_grad = (
#                     F.pad(unsharded_grad, [0, numel_to_pad])
#                     if numel_to_pad > 0
#                     else unsharded_grad
#                 )
#                 new_sharded_grad = torch.empty_like(chunks[0])  # padded
#                 state._communication_hook(
#                     state._communication_hook_state,
#                     padded_unsharded_grad,
#                     new_sharded_grad,
#                 )
#                 if handle._sharding_strategy in (
#                     HandleShardingStrategy.HYBRID_SHARD,
#                     HandleShardingStrategy._HYBRID_SHARD_ZERO2,
#                 ):
#                     default_hooks.allreduce_hook(
#                         state=state._inter_node_state,
#                         grad=new_sharded_grad,
#                     )
#                 _cast_grad_to_param_dtype(state, new_sharded_grad, flat_param)
#                 # Save the sharded gradient in `_saved_grad_shard` to support
#                 # gradient accumulation -- for multiple backwards, the gradient
#                 # reductions may happen in arbitrary order
#                 accumulate_grad = hasattr(flat_param, "_saved_grad_shard")
#                 if accumulate_grad:
#                     _check_grad_to_accumulate(
#                         new_sharded_grad, flat_param._saved_grad_shard
#                     )
#                     flat_param._saved_grad_shard += new_sharded_grad
#                 else:
#                     flat_param._saved_grad_shard = new_sharded_grad
#                 grad_to_offload = flat_param._saved_grad_shard
#             else:
#                 state._communication_hook(
#                     state._communication_hook_state, flat_param.grad
#                 )
#                 # For `NO_SHARD`, we can keep the low precision gradients by
#                 # simply omitting the cast altogether
#                 if not handle._keep_low_precision_grads:
#                     _cast_grad_to_param_dtype(state, flat_param.grad, flat_param)
#                 grad_to_offload = flat_param.grad.data

#             if handle._offload_params:
#                 # Offload the gradient to CPU to ensure parameters and
#                 # gradients are on the same device as required by the optimizer
#                 # TODO: Investigate why `NO_SHARD` breaks correctness when
#                 # using `non_blocking=True` here.
#                 non_blocking = handle.uses_sharded_strategy
#                 flat_param._cpu_grad.copy_(  # type: ignore[attr-defined]
#                     grad_to_offload.detach(), non_blocking=non_blocking
#                 )  # synchronized in the post-backward callback
#                 # Since the gradient being offloaded may have been produced in
#                 # the computation stream and is being consumed here in the
#                 # post-backward stream, inform the caching allocator
#                 _no_dispatch_record_stream(
#                     grad_to_offload.data,
#                     state._streams["post_backward"],
#                 )

#             # Since the unsharded gradient is produced in the computation
#             # stream and consumed in the post-backward stream, inform the
#             # caching allocator (before it goes out of scope)
#             _no_dispatch_record_stream(
#                 autograd_computed_grad, state._streams["post_backward"]
#             )

#             if handle._use_orig_params:
#                 # Since the handle's `FlatParameter` completed its gradient
#                 # computation, we should reset the gradient noneness mask
#                 handle._reset_is_grad_none()
#                 # Delay using sharded gradient views until after the
#                 # reduce-scatter instead of immediately after resharding
#                 handle._use_sharded_grad_views()
                



# ================ FSDP Sync Reduce Gradient Utils ================
# Sync gradients reduce mode, p2p has to wait for allreduce to finish.
# This mode is slow and not recommended, and will be deprecated in the future.

# import functools, weakref
# from torch.distributed.fsdp._runtime_utils import _post_backward_hook
# from torch.nn.parallel import DistributedDataParallel as DDP

# def pre_pipeline_forward(num_microbatches, idx, model):
#     if num_microbatches > 1 and idx == 0:
#         delete_ddp_backward_hook(model)
        
# def post_pipeline_forward(num_microbatches, idx, model, checkpoint_list):
#     if num_microbatches > 1:
#         if isinstance(model, FSDP):
#             model = model._fsdp_wrapped_module
#         assert(len(model)==len(checkpoint_list))
#         for module, checkpoint in zip(model, checkpoint_list):
#             if not checkpoint:
#                 if idx == num_microbatches - 1:
#                     delete_fsdp_post_backward_hook(module, save_acc_grad=True, release_param=True)
#                 else:
#                     delete_fsdp_post_backward_hook(module)
#             else:
#                 if idx == num_microbatches - 1:
#                     rewrite_fsdp_forward_no_post_backward(module)
                    
# def pre_pipeline_backward(num_microbatches, idx, model, checkpoint_list):
#     if num_microbatches > 1:
#         if isinstance(model, FSDP):
#             model = model._fsdp_wrapped_module
#         assert(len(model)==len(checkpoint_list))
#         if idx == num_microbatches - 1:
#             register_ddp_backward_hook(model)
#             for module, checkpoint in zip(model, checkpoint_list):
#                 if not checkpoint:
#                     register_fsdp_post_backward_hook(module)
#                 else:
#                     recover_fsdp_forward_with_post_backward(module)
    
# def _register_post_backward_hooks_handle(
#     state: _FSDPState,
#     handle: FlatParamHandle,
# ) -> None:
#     if not torch.is_grad_enabled():
#         return
#     flat_param = handle.flat_param
#     already_registered = hasattr(flat_param, "_post_backward_hook_state")
#     if already_registered or not flat_param.requires_grad:
#         return
#     # Get the `AccumulateGrad` object
#     acc_grad = handle.acc_grad  # type: ignore[union-attr]
#     assert acc_grad is not None
#     hook_handle = acc_grad.register_hook(
#         functools.partial(_post_backward_hook, state, handle)
#     )
#     flat_param._post_backward_hook_state = (acc_grad, hook_handle)  # type: ignore[attr-defined]

# def delete_fsdp_post_backward_hook(model, save_acc_grad=False, release_param=True):
#     for m in model.modules():
#         if isinstance(m, FSDP):
#             for handle in m._handles:
#                 flat_param = handle.flat_param
#                 if flat_param.requires_grad:
#                     if hasattr(flat_param, "_post_backward_hook_state"):
#                         if save_acc_grad:
#                             handle.acc_grad = flat_param._post_backward_hook_state[0]
#                         flat_param._post_backward_hook_state[1].remove()
#                         delattr(flat_param, "_post_backward_hook_state") # whether to reduce-scatter and release grad
#                     flat_param._post_backward_called = False
#             if not release_param and m._is_root:
#                 m._post_backward_callback_queued = True # whether to release params, trades off an allgather between param memory

# def register_fsdp_post_backward_hook(model):
#     for m in model.modules():
#         if isinstance(m, FSDP):
#             for handle in m._handles:
#                 _register_post_backward_hooks_handle(m, handle)
#             if m._is_root:
#                 m.training_state = TrainingState.IDLE
#             m._post_backward_callback_queued = False # need to wait for post backward

# def delete_ddp_backward_hook(model):
#     for m in model.modules():
#         # For DDP module, we need to disable gradient sync for accumulation, 
#         #   and set sync manually before backward of the last microbatch.
#         if isinstance(m, DDP):
#             m.require_backward_grad_sync = False

# def register_ddp_backward_hook(model):
#     for m in model.modules():
#         # For DDP module, we need to disable gradient sync for accumulation, 
#         #   and set sync manually before backward of the last microbatch.
#         if isinstance(m, DDP):
#             m.require_forward_param_sync = True
#             m.reducer.prepare_for_backward([])

# def forward_delete_backward_hook(original_forward, weak_self, *args, **kwargs):
#     module = weak_self()
#     output = original_forward(module, *args, **kwargs)
#     with torch.no_grad():
#         delete_fsdp_post_backward_hook(module, save_acc_grad=True, release_param=False)
#     return output

# def rewrite_fsdp_forward_no_post_backward(model):
#     for m in model.modules():
#         if isinstance(m, FSDP):
#             m.original_forward = m.forward
#             m.forward = functools.partial(forward_delete_backward_hook, type(m).forward, weakref.ref(m))

# def recover_fsdp_forward_with_post_backward(model):
#     for m in model.modules():
#         if isinstance(m, FSDP):
#             m.forward = m.original_forward

def _register_post_backward_hook_bf16(
    state: _FSDPState,
    handle: Optional[FlatParamHandle],
) -> None:
    """
    Registers post-backward hooks on the ``FlatParameter`` s'
    ``AccumulateGrad`` objects to reshard and to reduce-scatter gradients.

    The ``AccumulateGrad`` object represents the last function that finalizes
    the ``FlatParameter`` 's gradient, so it only runs after its entire
    gradient computation has finished.

    We register the post-backward hook only once in the *first* forward that a
    ``FlatParameter`` participates in. This relies on the ``AccumulateGrad``
    object being preserved through multiple forwards.

    NOTE: We follow this heuristic to prefer the *first* forward to target the
    parameter mixed precision case, where there are *separate*
    ``AccumulateGrad`` objects across the different forwards. (Without
    parameter mixed precision, the ``AccumulateGrad`` objects are the same.) If
    we instead prefer the *last* forward, then the hook runs early.
    """
    # If there is no gradient computation, then there is no need for
    # post-backward logic
    if not torch.is_grad_enabled():
        return
    if not handle:
        return
    flat_param = handle.flat_param
    already_registered = hasattr(flat_param, "_post_backward_hook_state")
    # if already_registered or not flat_param.requires_grad:
    #     return
    if not already_registered:
        flat_param._post_backward_hook_state = []
    # Get the `AccumulateGrad` object
    temp_flat_param = flat_param.expand_as(flat_param)
    _p_assert(
        temp_flat_param.grad_fn is not None,
        "The `grad_fn` is needed to access the `AccumulateGrad` and "
        "register the post-backward hook",
    )
    acc_grad = temp_flat_param.grad_fn.next_functions[0][0]  # type: ignore[union-attr]
    assert acc_grad is not None
    hook_handle = acc_grad.register_hook(
        functools.partial(_post_backward_hook, state, handle)
    )
    flat_param._post_backward_hook_state.append((acc_grad, hook_handle))  # type: ignore[attr-defined]

@no_type_check
def _finalize_params_bf16(
    state: _FSDPState,
) -> None:
    """Finalizes the parameters before the next iteration."""
    handle = state._handle
    if not handle:
        return
    flat_param = handle.flat_param
    if hasattr(flat_param, "_post_backward_hook_state"):
        # post_backward_hook_state_len = len(flat_param._post_backward_hook_state)
        # expected_post_backward_hook_state_len = int(flat_param.requires_grad) + 1
        # _p_assert(
        #     post_backward_hook_state_len == expected_post_backward_hook_state_len,
        #     f"Invalid: ``_post_backward_hook_state``: {flat_param._post_backward_hook_state}",
        # )
        if len(flat_param._post_backward_hook_state) > 0:
            flat_param._post_backward_hook_state[0][-1].remove()
            flat_param._post_backward_hook_state.pop(0)
        # delattr(flat_param, "_post_backward_hook_state")
    if flat_param.requires_grad:
        if not state._sync_gradients:
            # Preserve the gradient accumulation state if not synchronizing
            # gradients: `.grad` remains the unsharded gradient  from prior
            # `no_sync()` iterations, and `_saved_grad_shard` remains the
            # sharded gradient from the last synchronized iteration
            return
        if not handle._has_optim_in_backward:
            handle.prepare_gradient_for_optim()
        _p_assert(
            hasattr(flat_param, "_post_backward_called"),
            "Expects `_post_backward_called` to be set on the `FlatParameter`",
        )
        flat_param._post_backward_called = False