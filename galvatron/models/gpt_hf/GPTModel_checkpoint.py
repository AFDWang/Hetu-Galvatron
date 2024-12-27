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
    world_size = dist.get_world_size(tp_groups)
    rank = dist.get_rank(tp_groups)
    if name.endswith("wte"):
        file_path = os.path.join(load, embedding_name)
        checkpoint = torch.load(file_path, mmap=True, map_location='cpu')
        args = get_args()
        vocab_size = checkpoint["wte.weight"].shape[0]
        padding_size = args.padded_vocab_size - vocab_size
        padded_weight = F.pad(checkpoint["wte.weight"].to(device = "cuda", dtype = torch.float32), (0, 0, padding_size, 0), mode='constant', value=0)
        vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
            args.padded_vocab_size, rank, world_size
        )
        submodule.weight.copy_(padded_weight[vocab_start_index:vocab_end_index])
    elif name.endswith("wpe"):
        file_path = os.path.join(load, embedding_name)
        checkpoint = torch.load(file_path, mmap=True, map_location='cpu')
        args = get_args()
        weight = checkpoint["wpe.weight"].to(device = "cuda", dtype = torch.float32)
        seq_start_index, seq_end_index = VocabUtility.vocab_range_from_global_vocab_size(
            args.seq_length, rank, world_size
        )
        submodule.weight.copy_(weight[seq_start_index:seq_end_index])
    elif name.endswith("ln_f"):
        file_path = os.path.join(load, ln_f_name)
        checkpoint = torch.load(file_path, mmap=True, map_location='cpu')
        weight = checkpoint["weight"].to(device = "cuda", dtype = torch.float32)
        bias = checkpoint["bias"].to(device = "cuda", dtype = torch.float32)
        submodule.weight.copy_(weight)
        submodule.bias.copy_(bias)
    elif name.endswith("lm_head"):
        file_path = os.path.join(load, cls_name)
        checkpoint = torch.load(file_path, mmap=True, map_location='cpu')
        args = get_args()
        vocab_size = checkpoint["wte.weight"].shape[0]
        padding_size = args.padded_vocab_size - vocab_size
        padded_weight = F.pad(checkpoint["wte.weight"].to(device = "cuda", dtype = torch.float32), (0, 0, padding_size, 0), mode='constant', value=0)
        vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
            args.padded_vocab_size, rank, world_size
        )
        submodule.weight.copy_(padded_weight[vocab_start_index:vocab_end_index].contiguous())
    else:
        file_path = os.path.join(load, layer_name%module.idx)
        checkpoint = torch.load(file_path, mmap=True, map_location='cpu')
        if name.startswith("attention"):
            if name.endswith("LayerNorm"):
                weight = checkpoint["ln_1.weight"].to(device = "cuda", dtype = torch.float32)
                bias = checkpoint["ln_1.bias"].to(device = "cuda", dtype = torch.float32)
                submodule.weight.copy_(weight)
                submodule.bias.copy_(bias)
            elif name.endswith("query_key_value"):
                args = get_args()
                weight = checkpoint["attn.c_attn.weight"].to(device = "cuda", dtype = torch.float32)
                bias = checkpoint["attn.c_attn.bias"].to(device = "cuda", dtype = torch.float32)
                # HuggingFace stores c_attn.weight as (hidden_dim, (3 nheads headdim))
                # while Megatron stores c_attn.weight as ((nheads 3 headdim), hidden_dim)
                headdim = args.hidden_size // args.num_attention_heads
                weight = rearrange(
                        weight.t(),
                        "(three nheads headdim) ... -> (nheads three headdim) ...",
                        three=3,
                        headdim=headdim,
                    )
                bias = rearrange(
                        bias,
                        "(three nheads headdim) ... -> (nheads three headdim) ...",
                        three=3,
                        headdim=headdim,
                    )
                weight_start_index, weight_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                    bias.shape[0], rank, world_size
                )
                submodule.weight.copy_(weight[weight_start_index:weight_end_index].contiguous())
                submodule.bias.copy_(bias[weight_start_index:weight_end_index].contiguous())
            elif name.endswith("dense"):
                weight = checkpoint["attn.c_proj.weight"].to(device = "cuda", dtype = torch.float32)
                bias = checkpoint["attn.c_proj.bias"].to(device = "cuda", dtype = torch.float32)
                weight_start_index, weight_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                    weight.shape[0], rank, world_size
                )
                submodule.weight.copy_(weight[weight_start_index:weight_end_index].t().contiguous())
                submodule.bias.copy_(bias.contiguous())
        elif name.startswith("mlp"):
            if name.endswith("LayerNorm"):
                weight = checkpoint["ln_2.weight"].to(device = "cuda", dtype = torch.float32)
                bias = checkpoint["ln_2.bias"].to(device = "cuda", dtype = torch.float32)
                submodule.weight.copy_(weight)
                submodule.bias.copy_(bias)
            elif name.endswith("dense_h_to_4h"):
                weight = checkpoint["mlp.c_fc.weight"].to(device = "cuda", dtype = torch.float32)
                bias = checkpoint["mlp.c_fc.bias"].to(device = "cuda", dtype = torch.float32)
                weight = weight.t()
                weight_start_index, weight_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                    weight.shape[0], rank, world_size
                )
                submodule.weight.copy_(weight[weight_start_index:weight_end_index].contiguous())
                submodule.bias.copy_(bias[weight_start_index:weight_end_index].contiguous())
            elif name.endswith("dense_4h_to_h"):
                weight = checkpoint["mlp.c_proj.weight"].to(device = "cuda", dtype = torch.float32)
                bias = checkpoint["mlp.c_proj.bias"].to(device = "cuda", dtype = torch.float32)
                weight_start_index, weight_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                    weight.shape[0], rank, world_size
                )
                submodule.weight.copy_(weight[weight_start_index:weight_end_index].t().contiguous())
                submodule.bias.copy_(bias.contiguous())

@torch.no_grad()
def load_gpt_module(load, tp_groups, name, submodule, module, distributed_checkpoint):
    if distributed_checkpoint:
        raise NotImplementedError("Distributed checkpoint is not supported for GPT")
    else:
        load_hf_checkpoint(load, tp_groups, name, submodule, module)
