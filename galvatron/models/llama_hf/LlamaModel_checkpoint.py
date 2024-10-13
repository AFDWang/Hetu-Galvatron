import torch
from einops import rearrange
import os
from galvatron.core.arguments import get_args
import torch.nn.functional as F
from megatron.core.tensor_parallel.utils import VocabUtility
import torch.distributed as dist

embedding_name = "model_embed_tokens.pt"
layer_name = "model_layers_%d.pt"
ln_f_name = "model_norm.pt"
cls_name = "lm_head.pt"

@torch.no_grad()
def load_llama_module(load, tp_groups, name, submodule, module):
    world_size = dist.get_world_size(tp_groups)
    rank = dist.get_rank(tp_groups)
    if name.endswith("embed_tokens"):
        file_path = os.path.join(load, embedding_name)
        checkpoint = torch.load(file_path, mmap=True, map_location='cpu')
        args = get_args()
        vocab_size = checkpoint["embed_tokens.weight"].shape[0]
        padding_size = args.padded_vocab_size - vocab_size
        padded_weight = F.pad(checkpoint["embed_tokens.weight"].to(device = "cuda", dtype = torch.float32), (0, 0, padding_size, 0), mode='constant', value=0)
        vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
            args.padded_vocab_size, rank, world_size
        )
        submodule.weight.copy_(padded_weight[vocab_start_index:vocab_end_index])
    elif name.endswith("norm"):
        file_path = os.path.join(load, ln_f_name)
        checkpoint = torch.load(file_path, mmap=True, map_location='cpu')
        weight = checkpoint["weight"].to(device = "cuda", dtype = torch.float32)
        submodule.weight.copy_(weight)
    elif name.endswith("lm_head"):
        file_path = os.path.join(load, cls_name)
        checkpoint = torch.load(file_path, mmap=True, map_location='cpu')
        args = get_args()
        vocab_size = checkpoint["weight"].shape[0]
        padding_size = args.padded_vocab_size - vocab_size
        padded_weight = F.pad(checkpoint["weight"].to(device = "cuda", dtype = torch.float32), (0, 0, padding_size, 0), mode='constant', value=0)
        vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
            args.padded_vocab_size, rank, world_size
        )
        submodule.weight.copy_(padded_weight[vocab_start_index:vocab_end_index].contiguous())
    else:
        file_path = os.path.join(load, layer_name%module.idx)
        checkpoint = torch.load(file_path, mmap=True, map_location='cpu')
        if name.startswith("attention"):
            if name.endswith("LayerNorm"):
                weight = checkpoint["input_layernorm.weight"].to(device = "cuda", dtype = torch.float32)
                submodule.weight.copy_(weight)
            elif name.endswith("query_key_value"):
                args = get_args()
                # q: num_heads * head_dim, hidden_size
                # k,v: num_key_value_heads * head_dim, hidden_size
                # while Megatron stores c_attn.weight as ((nheads 3 headdim), hidden_dim)
                nh = args.num_attention_heads
                ng = (args.num_query_groups if args.group_query_attention \
                    else args.num_attention_heads)
                dim = args.kv_channels
                assert nh % ng == 0
                weight = torch.cat([
                        checkpoint["self_attn.q_proj.weight"].reshape((ng, dim*nh//ng, -1)),
                        checkpoint["self_attn.k_proj.weight"].reshape((ng, dim, -1)),
                        checkpoint["self_attn.v_proj.weight"].reshape((ng, dim, -1)),
                    ], dim=1).reshape((-1, args.hidden_size))
                weight_start_index, weight_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                    weight.shape[0], rank, world_size
                )
                submodule.weight.copy_(weight[weight_start_index:weight_end_index].contiguous())
            elif name.endswith("dense"):
                # o: hidden_size, num_heads * head_dim
                weight = checkpoint["self_attn.o_proj.weight"].to(device = "cuda", dtype = torch.float32)
                weight_start_index, weight_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                    weight.shape[1], rank, world_size
                )
                submodule.weight.copy_(weight[:,weight_start_index:weight_end_index].contiguous())
        elif name.startswith("mlp"):
            if name.endswith("LayerNorm"):
                weight = checkpoint["post_attention_layernorm.weight"].to(device = "cuda", dtype = torch.float32)
                submodule.weight.copy_(weight)
            elif name.endswith("dense_h_to_4h"):
                weight_start_index, weight_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                    checkpoint["mlp.gate_proj.weight"].shape[0], rank, world_size
                )
                weight = (torch.cat([
                    checkpoint["mlp.gate_proj.weight"][weight_start_index:weight_end_index].contiguous(),
                    checkpoint["mlp.up_proj.weight"][weight_start_index:weight_end_index].contiguous(),
                ], dim=0))
                
                submodule.weight.copy_(weight.contiguous())
            elif name.endswith("dense_4h_to_h"):
                weight = checkpoint["mlp.down_proj.weight"].to(device = "cuda", dtype = torch.float32)
                weight_start_index, weight_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                    weight.shape[1], rank, world_size
                )
                submodule.weight.copy_(weight[:,weight_start_index:weight_end_index].contiguous())
        