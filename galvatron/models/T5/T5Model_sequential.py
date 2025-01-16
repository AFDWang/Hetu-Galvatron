import torch.distributed
import torch.nn as nn
import torch
from galvatron.core.pipeline import PipeSequential
from galvatron.core import mixed_precision_dtype, ModelInfo
from galvatron.core import get_args
from megatron.core import tensor_parallel
from galvatron.core.tensor_parallel import colummn_row_reset_parameters
from megatron.core.tensor_parallel.utils import VocabUtility
from megatron.core.tensor_parallel.mappings_group import get_tensor_model_parallel_world_size_group
from flash_attn.ops.rms_norm import RMSNorm as T5LayerNorm

# wrap inputs: [dec_tokens, enc_attn_mask, dec_attn_mask, enc_dec_attn_mask, dec_labels]
class T5EncoderEmbeddings_(nn.Module):
    def __init__(self, model):
        super().__init__()
        args = get_args()
        self.embeddings = model.shared
        self.drop = torch.nn.Dropout(args.hidden_dropout)
        self.sequence_parallel = args.sequence_parallel
        self.clone_scatter_output_in_embedding = args.clone_scatter_output_in_embedding
        self.tp_group = self.embeddings.tp_group
        self.sp_group = self.embeddings.sp_group
        self.vocab_sp = args.vocab_sp
        if self.vocab_sp:
            self.seq_start_index, self.seq_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                    args.encoder_seq_length, torch.distributed.get_rank(self.sp_group), torch.distributed.get_world_size(self.sp_group)
                )
        
    def forward(self, enc_tokens, dec_tokens, enc_attn_mask, dec_attn_mask, enc_dec_attn_mask, dec_labels):
        if self.vocab_sp:
            enc_tokens = enc_tokens[:, self.seq_start_index:self.seq_end_index].contiguous()

        hidden_states = self.embeddings(enc_tokens)
        # hidden_states = self.drop(hidden_states)
        # [b, s, h] -> [s, b, h]
        hidden_states = hidden_states.transpose(0, 1).contiguous()
        if self.sequence_parallel:
            hidden_states = tensor_parallel.scatter_to_sequence_parallel_region_group(hidden_states, self.tp_group)
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            if self.clone_scatter_output_in_embedding:
                hidden_states = hidden_states.clone()
            with tensor_parallel.get_cuda_rng_tracker().fork():
                hidden_states = self.drop(hidden_states)
                
        else:
            hidden_states = self.drop(hidden_states)
            
        return hidden_states

class T5DecoderEmbeddings_(nn.Module):
    def __init__(self, model):
        super().__init__()
        args = get_args()
        self.embeddings = model.dec_shared
        self.drop = torch.nn.Dropout(args.hidden_dropout)
        self.sequence_parallel = args.sequence_parallel
        self.clone_scatter_output_in_embedding = args.clone_scatter_output_in_embedding
        self.tp_group = self.embeddings.tp_group
        self.sp_group = self.embeddings.sp_group
        self.vocab_sp = args.vocab_sp
        if self.vocab_sp:
            self.seq_start_index, self.seq_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                    args.decoder_seq_length, torch.distributed.get_rank(self.sp_group), torch.distributed.get_world_size(self.sp_group)
                )
        
    def forward(self, enc_hidden_states, dec_tokens, enc_attn_mask, dec_attn_mask, enc_dec_attn_mask, dec_labels):
        if self.vocab_sp:
            dec_tokens = dec_tokens[:, self.seq_start_index:self.seq_end_index].contiguous()

        hidden_states = self.embeddings(dec_tokens)
        # hidden_states = self.drop(hidden_states)
        # [b, s, h] -> [s, b, h]
        hidden_states = hidden_states.transpose(0, 1).contiguous()
        if self.sequence_parallel:
            hidden_states = tensor_parallel.scatter_to_sequence_parallel_region_group(hidden_states, self.tp_group)
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            if self.clone_scatter_output_in_embedding:
                hidden_states = hidden_states.clone()
            with tensor_parallel.get_cuda_rng_tracker().fork():
                hidden_states = self.drop(hidden_states)
                
        else:
            hidden_states = self.drop(hidden_states)
            
        return enc_hidden_states, hidden_states

class T5EncoderLayers_(nn.Module):
    def __init__(self, model, layer_idx):
        super().__init__()
        self.layer = model.encoder[layer_idx]

    def forward(self, hidden_states, dec_tokens, enc_attn_mask, dec_attn_mask, enc_dec_attn_mask, dec_labels):
        # attention_mask = get_ltor_masks_and_position_ids(input_ids)
        hidden_states = self.layer(hidden_states, enc_attn_mask = enc_attn_mask)
        return hidden_states

class T5DecoderLayers_(nn.Module):
    def __init__(self, model, layer_idx):
        super().__init__()
        self.layer = model.decoder[layer_idx]

    def forward(self, enc_hidden_states, dec_hidden_states, dec_tokens, enc_attn_mask, dec_attn_mask, enc_dec_attn_mask, dec_labels):
        enc_hidden_states, dec_hidden_states = self.layer(enc_hidden_states, dec_hidden_states, dec_attn_mask = dec_attn_mask, enc_dec_attn_mask = enc_dec_attn_mask)
        return enc_hidden_states, dec_hidden_states

class T5EncoderPreNorm_(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.LayerNorm = T5LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.Dropout = nn.Dropout(config.dropout_rate)

    def forward(self, enc_hidden_states, dec_tokens, enc_attn_mask, dec_attn_mask, enc_dec_attn_mask, dec_labels):
        hidden_states = self.LayerNorm(enc_hidden_states)
        hidden_states = self.Dropout(hidden_states)
        return hidden_states

class T5DecoderPreNorm_(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.LayerNorm = T5LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.Dropout = nn.Dropout(config.dropout_rate)

    def forward(self, enc_hidden_states, dec_hidden_states, dec_tokens, enc_attn_mask, dec_attn_mask, enc_dec_attn_mask, dec_labels):
        hidden_states = self.LayerNorm(dec_hidden_states)
        hidden_states = self.Dropout(hidden_states)
        return enc_hidden_states, hidden_states

class T5Loss_(nn.Module):
    def __init__(self, weight, sequence_parallel, tp_group):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
        self.sequence_parallel = sequence_parallel
        self.tp_group = tp_group
        world_size = get_tensor_model_parallel_world_size_group(tp_group)
        if self.sequence_parallel and world_size <= 1:
            self.sequence_parallel = False
            # disable sp to avoid global buffer
    
    def forward(self, hidden_states):
        logits_parallel = tensor_parallel.linear_with_grad_accumulation_and_async_allreduce(
            input=hidden_states,
            weight=self.weight,
            bias=None,
            gradient_accumulation_fusion=False,
            async_grad_allreduce=False,
            sequence_parallel=self.sequence_parallel,
            tp_group=self.tp_group)
        return logits_parallel

class T5Cls_(nn.Module):
    def __init__(self, model, parallel_loss = True, half_entropy = True):
        super().__init__()
        self.sequence_parallel = get_args().sequence_parallel
        self.tp_group = model.lm_head.tp_group
        self.sp_group = model.lm_head.sp_group
        self.lm_head = T5Loss_(model.lm_head.weight, self.sequence_parallel, self.tp_group)
        self.clone_scatter_output_in_embedding = get_args().clone_scatter_output_in_embedding
        self.parallel_loss = parallel_loss
        self.half_entropy = half_entropy
        args = get_args()
        if args.entropy_in_fp32:
            self.half_entropy = False
        self.seq_length = args.decoder_seq_length
        self.vocab_sp = args.vocab_sp
        if self.vocab_sp:
            self.seq_start_index, self.seq_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                    self.seq_length, torch.distributed.get_rank(self.sp_group), torch.distributed.get_world_size(self.sp_group)
                )

    def forward(self, enc_hidden_states, dec_hidden_states, dec_tokens, enc_attn_mask, dec_attn_mask, enc_dec_attn_mask, dec_labels):
        if self.vocab_sp:
            labels = dec_labels[:, self.seq_start_index:self.seq_end_index].contiguous()
        else:
            labels = dec_labels
        if not self.sequence_parallel:
            hidden_states = tensor_parallel.copy_to_tensor_model_parallel_region_group(dec_hidden_states, self.tp_group)
        else:
            hidden_states = dec_hidden_states
        logits_parallel = self.lm_head(hidden_states)
        
        # [b s] -> [s b]
        labels = labels.transpose(0,1).contiguous()
        
        # loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), input_ids)
        if not self.parallel_loss:
            output = tensor_parallel.gather_from_tensor_model_parallel_region_group(logits_parallel, self.tp_group)
            if not self.half_entropy:
                logits = output.float()
            else:
                logits = output
            loss = None
            # Shift so that tokens < n predict n
            shift_logits = logits.contiguous() # logits[:-1, ..., :].contiguous()
            shift_labels = labels.contiguous() # input_ids[1:, ...].contiguous()
            # Flatten the tokens
            from torch.nn import CrossEntropyLoss
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        else:
            if not self.vocab_sp:
                if not self.half_entropy:
                    loss = tensor_parallel.vocab_parallel_cross_entropy(logits_parallel.float(), labels, tp_group = self.tp_group)
                else:
                    loss = tensor_parallel.vocab_parallel_cross_entropy(logits_parallel, labels, tp_group = self.tp_group)
            else:
                if not self.half_entropy:
                    loss = tensor_parallel.vocab_sequence_parallel_cross_entropy(logits_parallel.float(), labels, self.sp_group)
                else:
                    loss = tensor_parallel.vocab_sequence_parallel_cross_entropy(logits_parallel, labels, self.sp_group)
            # loss = loss.mean()
        loss = loss.transpose(0,1).contiguous()
        return loss

def construct_sequential_model(model, config):
    model_ = PipeSequential()
    model_.add_module('encoder_embeddings', T5EncoderEmbeddings_(model))
    for i in range(config.num_layers):
        enc = T5EncoderLayers_(model, i)
        model_.add_module('encoder_layer_%d'%i, enc)
    model_.add_module('encoder_pre_norm', T5EncoderPreNorm_(config))
    model_.add_module('decoder_embeddings', T5DecoderEmbeddings_(model))
    for i in range(config.num_decoder_layers):
        dec = T5DecoderLayers_(model, i)
        model_.add_module('decoder_layer_%d'%i, dec)
    model_.add_module('decoder_pre_norm', T5DecoderPreNorm_(config))
    model_.add_module('cls', T5Cls_(model))
    T5Loss_.reset_parameters = colummn_row_reset_parameters
    return model_

class T5ModelInfo(ModelInfo):
    def __init__(self, config, args):
        super(T5ModelInfo, self).__init__()
        layernum_list = [config.num_layers, config.num_decoder_layers]
        enc_seq_len, hidden_size = config.n_positions, config.hidden_size
        dec_seq_len, hidden_size = config.n_decoder_positions, config.hidden_size
        mixed_precision = mixed_precision_dtype(args.mixed_precision)
        if args.shape_order == "SBH":
            layer_shapes_list = [
                [[enc_seq_len,-1,hidden_size]],
                [[enc_seq_len,-1,hidden_size], [dec_seq_len,-1,hidden_size]]
            ]
        else:
            layer_shapes_list = [
                [[-1,enc_seq_len,hidden_size]],
                [[-1,enc_seq_len,hidden_size], [-1,dec_seq_len,hidden_size]]
            ]
        layer_dtypes_list = [
            [mixed_precision],
            [mixed_precision, mixed_precision]
        ]
        module_types = ['embed_1'] + ['t5_enc']*config.num_layers + ['pre_norm_1'] + ['embed_2'] + ['t5_dec']*config.num_decoder_layers + ['pre_norm_2'] + ['cls']
        self.set_layernums(layernum_list)
        self.set_shapes(layer_shapes_list)
        self.set_dtypes(layer_dtypes_list)
        self.set_module_types(module_types)