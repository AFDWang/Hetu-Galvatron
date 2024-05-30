import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.arguments import core_transformer_config_from_args
from galvatron.core import get_args
from galvatron.core.tensor_parallel import ParallelMLP, ParallelAttention
from galvatron.core.tensor_parallel import AttnMaskType, AttnType

class GPTAttention_tp(nn.Module):
    def __init__(self, config, layer_number, tp_group = None):
        super().__init__()
        megatron_config = core_transformer_config_from_args(get_args())
        self.tp_group = tp_group.group if tp_group is not None else None
        self.attention = ParallelAttention(megatron_config, layer_number,
                                        attention_type=AttnType.self_attn,
                                        attn_mask_type=AttnMaskType.causal,
                                        tp_group = self.tp_group)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)


    def forward(self, hidden_states, attention_mask):
        input_tensor = hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states, bias = self.attention(hidden_states, attention_mask)
        hidden_states = hidden_states + input_tensor
        return hidden_states

class GPTMLP_tp(nn.Module):
    def __init__(self, config, tp_group = None):
        super().__init__()
        megatron_config = core_transformer_config_from_args(get_args())
        self.tp_group = tp_group.group if tp_group is not None else None
        self.mlp = ParallelMLP(megatron_config, tp_group = self.tp_group)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states):
        input_tensor = hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states, bias = self.mlp(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states

class GPTLayer_tp(nn.Module):
    def __init__(self, config, layer_number, tp_group = None):
        super().__init__()
        self.attention = GPTAttention_tp(config, layer_number, tp_group)
        self.mlp = GPTMLP_tp(config, tp_group)

    def forward(
        self,
        hidden_states,
        attention_mask = None,
    ):
        attention_output = self.attention(
            hidden_states,
            attention_mask,
        )
        layer_output = self.mlp(attention_output)
        return layer_output
    
def construct_tensor_parallel_model(model, config, tp_groups_enc):
    layers_tp = nn.ModuleList([GPTLayer_tp(config, i, tp_group = tp_groups_enc[i]) for i in range(config.num_hidden_layers)])
    setattr(model.transformer, 'h', layers_tp)
    return model