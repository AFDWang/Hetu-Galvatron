# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

from .mappings_group import get_tensor_model_parallel_rank_group, get_tensor_model_parallel_world_size_group

from .utils import VocabUtility


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, label_smoothing=0.0, tp_group=None):

        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        if tp_group == None:
            torch.distributed.all_reduce(
                logits_max, op=torch.distributed.ReduceOp.MAX, group=get_tensor_model_parallel_group()
            )
        else:
            torch.distributed.all_reduce(
                logits_max, op=torch.distributed.ReduceOp.MAX, group=tp_group
            )
        # Subtract the maximum value.
        vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)

        # Get the partition's vocab indecies
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        if tp_group == None:
            rank = get_tensor_model_parallel_rank()
            world_size = get_tensor_model_parallel_world_size()
        else:
            rank = get_tensor_model_parallel_rank_group(tp_group)
            world_size = get_tensor_model_parallel_world_size_group(tp_group)
        
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        if tp_group == None:
            torch.distributed.all_reduce(
                predicted_logits,
                op=torch.distributed.ReduceOp.SUM,
                group=get_tensor_model_parallel_group(),
            )
        else:
            torch.distributed.all_reduce(
                predicted_logits,
                op=torch.distributed.ReduceOp.SUM,
                group=tp_group,
            )

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        if tp_group == None:
            torch.distributed.all_reduce(
                sum_exp_logits,
                op=torch.distributed.ReduceOp.SUM,
                group=get_tensor_model_parallel_group(),
            )
        else:
            torch.distributed.all_reduce(
                sum_exp_logits,
                op=torch.distributed.ReduceOp.SUM,
                group=tp_group,
            )

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Normalize and optionally smooth logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        vocab_size = exp_logits.size(-1)
        if label_smoothing > 0:
            """
            We'd like to assign 1 / (K - 1) probability mass to every index that is not the ground truth.
            = (1 - alpha) * y_gt + alpha * mean(y_{i for i != gt})
            = (1 - alpha) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = ((K - 1) * (1 - alpha) / (K - 1)) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = (K * (1 - alpha) - 1) / (K - 1)) * y_gt  + (alpha / (K - 1)) * \sum_{i} y_i
            = (1 - (alpha * K) / (K - 1)) * y_gt + ( (alpha * K) / (K - 1) ) * \sum_{i} y_i / K
            From: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/losses/smoothed_cross_entropy.py
            """
            assert 1.0 > label_smoothing > 0.0
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)

            # Exp logits at this point are normalized probabilities. So we can just take the log to get log-probs.
            log_probs = torch.log(exp_logits)
            mean_log_probs = log_probs.mean(dim=-1)
            loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

        ctx.label_smoothing, ctx.vocab_size = label_smoothing, vocab_size

        # Store softmax, target-mask and masked-target for backward pass.
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output):

        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors
        label_smoothing, vocab_size = ctx.label_smoothing, ctx.vocab_size

        # All the inputs have softmax as thier gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)

        softmax_update = 1.0 - target_mask.view(-1).float()

        if label_smoothing > 0:
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)
            grad_2d[arange_1d, masked_target_1d] -= (1.0 - smoothing) * softmax_update
            average_grad = 1 / vocab_size
            grad_2d[arange_1d, :] -= smoothing * average_grad
        else:
            grad_2d[arange_1d, masked_target_1d] -= softmax_update

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None, None, None


def vocab_parallel_cross_entropy(vocab_parallel_logits, target, label_smoothing=0.0, tp_group = None):
    """
    Performs cross entropy loss when logits are split across tensor parallel ranks

    Args:
        vocab_parallel_logits: logits split across tensor parallel ranks
                               dimension is [sequence_length, batch_size, hidden_size]

        target: correct vocab ids of dimseion [sequence_length, micro_batch_size]

        lobal_smoothing: smoothing factor, must be in range [0.0, 1.0)
                         default is no smoothing (=0.0)
    """
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, label_smoothing, tp_group)

import torch
from packaging import version
import torch.distributed

class _VocabSequenceParallelCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_seq_parallel_logits, target, sp_group, label_smoothing=0.0):
        # vocab_seq_parallel_logits: [S/P, B, V]
        # target: [S/P, B]
        # return: [S, B]

        # Need softmax for backward
        softmax = torch.nn.functional.softmax(vocab_seq_parallel_logits, dim=-1)
        ctx.vocab_size = vocab_seq_parallel_logits.size(2)
        loss = torch.nn.functional.nll_loss(softmax.log().view(-1, ctx.vocab_size), target.view(-1), reduction='none', ignore_index=-1)
       
        ctx.seqlen = vocab_seq_parallel_logits.size(0) * torch.distributed.get_world_size(sp_group)
        batch_size = vocab_seq_parallel_logits.size(1)
        ctx.sp_group = sp_group
        loss_all = torch.empty(ctx.seqlen, batch_size, dtype=vocab_seq_parallel_logits.dtype, device=vocab_seq_parallel_logits.device)
        if version.parse(torch.__version__) >= version.parse('1.13'):
            torch.distributed.all_gather_into_tensor(loss_all, loss, group=sp_group)
        else:
            torch.distributed._all_gather_base(loss_all, loss, group=sp_group)

        ctx.save_for_backward(softmax, target)

        return loss_all

    @staticmethod
    def backward(ctx, grad_output):
        softmax, target = ctx.saved_tensors

        step_seqlen = ctx.seqlen // torch.distributed.get_world_size(ctx.sp_group)
        sp_rank = torch.distributed.get_rank(ctx.sp_group)
        grad_output_part = grad_output[step_seqlen*sp_rank:step_seqlen*(sp_rank + 1), :]

        grad_input = softmax
        grad_2d = grad_input.view(-1, ctx.vocab_size)
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0],
                                 device=grad_2d.device)

        grad_2d[arange_1d, target.view(-1)] -= 1
        grad_input.mul_(grad_output_part.unsqueeze(dim=-1))

        return grad_input, None, None, None


def vocab_sequence_parallel_cross_entropy(vocab_parallel_logits, target, sp_group, label_smoothing=0.0):
    return _VocabSequenceParallelCrossEntropy.apply(vocab_parallel_logits, target, sp_group, label_smoothing)
