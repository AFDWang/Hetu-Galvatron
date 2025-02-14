import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from megatron.core.tensor_parallel.utils import VocabUtility

from galvatron.core.arguments import get_args

embedding_name = "transformer_embedding.pt"
layer_name = "transformer_h_%d.pt"
ln_f_name = "transformer_ln_f.pt"
cls_name = "transformer_embedding.pt"


@torch.no_grad()
def load_hf_checkpoint(load, tp_groups, name, submodule, module):
    raise NotImplementedError("Distributed checkpoint is not supported for T5")


@torch.no_grad()
def load_swin_module(load, tp_groups, name, submodule, module, distributed_checkpoint):
    if distributed_checkpoint:
        raise NotImplementedError("Distributed checkpoint is not supported for T5")
    else:
        load_hf_checkpoint(load, tp_groups, name, submodule, module)


@torch.no_grad()
def save_swin_module(save, model, optimizer, opt_param_scheduler, iter, args):
    raise NotImplementedError("Load checkpoint is not supported for T5")
