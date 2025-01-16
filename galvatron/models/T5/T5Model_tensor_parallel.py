import torch
from torch import nn
from megatron.core.tensor_parallel import VocabParallelEmbedding, ColumnParallelLinear
from megatron.training.arguments import core_transformer_config_from_args
from galvatron.core import get_args
from galvatron.core.tensor_parallel import ParallelMLP, ParallelAttention
from galvatron.core.tensor_parallel import AttnMaskType, AttnType
from flash_attn.ops.rms_norm import RMSNorm as T5LayerNorm
# from megatron.model.fused_layer_norm import MixedFusedLayerNorm as LayerNorm

# TODO: support relative attention bias but flash attn does not support it
# TODO: support padding from flash attn(padding -> varlen)
class T5Attention_tp(nn.Module):
    def __init__(self, config, layer_number, attention_type = AttnType.self_attn, attn_mask_type = AttnMaskType.padding, tp_group = None, sp_group = None):
        super().__init__()
        args = get_args()
        self.use_ulysses = sp_group.size > 1
        megatron_config = core_transformer_config_from_args(args)
        self.tp_group = tp_group.group if tp_group is not None else None
        self.sp_group = sp_group.group if sp_group is not None else None
        self.attention = ParallelAttention(megatron_config, layer_number,
                                        attention_type=attention_type,
                                        attn_mask_type=attn_mask_type,
                                        tp_group = self.tp_group,
                                        sp_group = self.sp_group,
                                        use_ulysses = self.use_ulysses)
        self.LayerNorm = T5LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.Dropout = nn.Dropout(megatron_config.attention_dropout)

    def forward(self, hidden_states, attention_mask, encoder_output = None):
        input_tensor = hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states, bias = self.attention(hidden_states, attention_mask, encoder_output)
        if bias is not None:
            hidden_states = hidden_states + bias
        hidden_states = self.Dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states

class T5MLP_tp(nn.Module):
    def __init__(self, config, tp_group = None):
        super().__init__()
        megatron_config = core_transformer_config_from_args(get_args())
        self.tp_group = tp_group.group if tp_group is not None else None
        self.mlp = ParallelMLP(megatron_config, tp_group = self.tp_group)
        self.LayerNorm = T5LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.Dropout = nn.Dropout(megatron_config.hidden_dropout)
        
    def forward(self, hidden_states):
        input_tensor = hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states, bias = self.mlp(hidden_states)
        if bias is not None:
            hidden_states = hidden_states + bias
        hidden_states = self.Dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states

class T5EncoderLayer_tp(nn.Module):
    def __init__(self, config, layer_number, tp_group = None, sp_group = None):
        super().__init__()
        self.attention = T5Attention_tp(config, layer_number, AttnType.self_attn, AttnMaskType.padding, tp_group, sp_group)
        self.mlp = T5MLP_tp(config, tp_group)
        self.idx = layer_number

    def forward(
        self,
        hidden_states,
        enc_attn_mask = None,
    ):
        attention_output = self.attention(
            hidden_states,
            enc_attn_mask,
        )
        layer_output = self.mlp(attention_output)
        return layer_output

class T5DecoderLayer_tp(nn.Module):
    def __init__(self, config, layer_number, tp_group = None, sp_group = None):
        super().__init__()
        self.attention = T5Attention_tp(config, layer_number, AttnType.self_attn, AttnMaskType.causal, tp_group, sp_group)
        self.cross_attention = T5Attention_tp(config, layer_number, AttnType.cross_attn, AttnMaskType.padding, tp_group, sp_group)
        self.mlp = T5MLP_tp(config, tp_group)
        self.idx = layer_number

    def forward(
        self,
        enc_hidden_states,
        dec_hidden_states, 
        dec_attn_mask, 
        enc_dec_attn_mask
    ):
        attention_output = self.attention(
            dec_hidden_states,
            dec_attn_mask,
        )
        attention_output = self.cross_attention(
            attention_output,
            enc_dec_attn_mask,
            encoder_output = enc_hidden_states
        )
        layer_output = self.mlp(attention_output)
        return enc_hidden_states, layer_output
     
def construct_tensor_parallel_model(model, config, tp_groups_enc, sp_groups_enc):
    layers_tp = nn.ModuleList([T5EncoderLayer_tp(config, i, tp_group = tp_groups_enc[i + 1], sp_group = sp_groups_enc[i + 1]) for i in range(config.num_layers)])
    layers_tp_dec = nn.ModuleList([T5DecoderLayer_tp(config, i, tp_group = tp_groups_enc[config.num_layers + i + 3], sp_group = sp_groups_enc[config.num_layers + i + 3]) for i in range(config.num_decoder_layers)])
    setattr(model, 'encoder', layers_tp)
    setattr(model, 'decoder', layers_tp_dec)
    args = get_args()
    megatron_config = core_transformer_config_from_args(get_args())
    setattr(model, 'shared', VocabParallelEmbedding(
            args.padded_vocab_size, megatron_config.hidden_size, config = megatron_config, init_method = megatron_config.init_method, tp_group = tp_groups_enc[0].group, sp_group = sp_groups_enc[0].group))
    setattr(model, 'dec_shared',  VocabParallelEmbedding(
            args.padded_vocab_size, megatron_config.hidden_size, config = megatron_config, init_method = megatron_config.init_method, tp_group = tp_groups_enc[0].group, sp_group = sp_groups_enc[0].group))
    setattr(model, 'lm_head', ColumnParallelLinear(
            megatron_config.hidden_size, args.padded_vocab_size, config = megatron_config, init_method = megatron_config.init_method, bias=False, tp_group = tp_groups_enc[-1].group, sp_group = sp_groups_enc[-1].group))
    
    return model