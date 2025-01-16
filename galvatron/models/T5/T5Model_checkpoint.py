import torch
from einops import rearrange
import os
from galvatron.core.arguments import get_args
import torch.nn.functional as F
from megatron.core.tensor_parallel.utils import VocabUtility
import torch.distributed as dist

embedding_name = "transformer_embedding.pt"
layer_name = "transformer_h_%d.pt"
ln_f_name = "transformer_ln_f.pt"
cls_name = "transformer_embedding.pt"

@torch.no_grad()
def load_hf_checkpoint(load, tp_groups, name, submodule, module):
    raise NotImplementedError("Distributed checkpoint is not supported for T5")

@torch.no_grad()
def load_t5_module(load, tp_groups, name, submodule, module, distributed_checkpoint):
    if distributed_checkpoint:
        raise NotImplementedError("Distributed checkpoint is not supported for T5")
    else:
        load_hf_checkpoint(load, tp_groups, name, submodule, module)

@torch.no_grad()
def save_t5_module(save, model, optimizer, opt_param_scheduler, iter, args):
    raise NotImplementedError("Load checkpoint is not supported for T5")