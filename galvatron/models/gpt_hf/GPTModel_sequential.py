import torch.distributed
import torch.nn as nn
import torch
from galvatron.core.pipeline import PipeSequential
from galvatron.core import mixed_precision_dtype, ModelInfo
from galvatron.core import get_args
from megatron.core import tensor_parallel

def get_ltor_masks_and_position_ids(data):
    """Build masks and position id for left to right model."""
    micro_batch_size, seq_length = data.size()
    att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)
    
    # position_ids = torch.arange(seq_length, dtype=torch.long,
    #                             device=data.device)
    # position_ids = position_ids.unsqueeze(0).expand_as(data)
    attention_mask = (attention_mask < 0.5)

    return attention_mask# , position_ids


class GPTEmbeddings_(nn.Module):
    def __init__(self, model):
        super().__init__()
        model = model.transformer
        self.wte = model.wte
        self.wps = model.wpe
        args = get_args()
        self.drop = torch.nn.Dropout(args.hidden_dropout)# model.drop
        self.sequence_parallel = args.sequence_parallel
        self.clone_scatter_output_in_embedding = args.clone_scatter_output_in_embedding
        self.tp_group = self.wte.tp_group
        
    def forward(self, input_ids):

        tokens = input_ids[:, :-1].contiguous()
        labels = input_ids[:, 1:].contiguous()
        
        position_ids = torch.arange(0, tokens.size(-1), dtype=torch.long, device=tokens.device)
        position_ids = position_ids.unsqueeze(0)
        inputs_embeds = self.wte(tokens)
        position_embeds = self.wps(position_ids)
        hidden_states = inputs_embeds + position_embeds
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
            
        labels = labels.clone()
        
        return hidden_states, labels

class GPTLayers_(nn.Module):
    def __init__(self, model, layer_idx):
        super().__init__()
        model = model.transformer
        self.layer = model.h[layer_idx]

    def forward(self, hidden_states, input_ids):
        attention_mask = get_ltor_masks_and_position_ids(input_ids)
        hidden_states = self.layer(hidden_states, attention_mask = attention_mask) # , position_ids = position_ids)
        input_ids = input_ids.clone()
        return hidden_states, input_ids

class GPTPreNorm_(nn.Module):
    def __init__(self, model):
        super().__init__()
        model = model.transformer
        self.ln_f = model.ln_f

    def forward(self, hidden_states, input_ids):
        hidden_states = self.ln_f(hidden_states)
        input_ids = input_ids.clone()
        return hidden_states, input_ids
    
class GPTLoss_(nn.Module):
    def __init__(self, weight, sequence_parallel, tp_group):
        super().__init__()
        self.weight = weight
        self.sequence_parallel = sequence_parallel
        self.tp_group = tp_group
    
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


class GPTCls_(nn.Module):
    def __init__(self, model, parallel_loss = True, half_entorpy = True):
        super().__init__()
        self.sequence_parallel = get_args().sequence_parallel
        self.tp_group = model.lm_head.tp_group
        self.lm_head = GPTLoss_(model.lm_head.weight, self.sequence_parallel, self.tp_group)
        self.clone_scatter_output_in_embedding = get_args().clone_scatter_output_in_embedding
        self.parallel_loss = parallel_loss
        self.half_entorpy = half_entorpy

    def forward(self, hidden_states, input_ids):

        if not self.sequence_parallel:
            hidden_states = tensor_parallel.copy_to_tensor_model_parallel_region_group(hidden_states, self.tp_group)
        
        logits_parallel = self.lm_head(hidden_states)
    
        # [b s] -> [s b]
        input_ids = input_ids.transpose(0,1).contiguous()
        
        # loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), input_ids)
        if not self.parallel_loss:
            output = tensor_parallel.gather_from_tensor_model_parallel_region_group(logits_parallel, self.tp_group)
            if not self.half_entorpy:
                logits = output.float()
            else:
                logits = output
            loss = None
            # Shift so that tokens < n predict n
            shift_logits = logits.contiguous() # logits[:-1, ..., :].contiguous()
            shift_labels = input_ids.contiguous() # input_ids[1:, ...].contiguous()
            # Flatten the tokens
            from torch.nn import CrossEntropyLoss
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        else:
            if not self.half_entorpy:
                loss = tensor_parallel.vocab_parallel_cross_entropy(logits_parallel.float(), input_ids, tp_group = self.tp_group)
            else:
                loss = tensor_parallel.vocab_parallel_cross_entropy(logits_parallel, input_ids, tp_group = self.tp_group)
            loss = loss.mean()
        return loss

def construct_sequential_model(model, config):
    model_ = PipeSequential()
    model_.add_module('embeddings', GPTEmbeddings_(model))
    for i in range(config.num_hidden_layers):
        enc = GPTLayers_(model, i)
        model_.add_module('layer_%d'%i, enc)
    model_.add_module('prenorm', GPTPreNorm_(model))
    model_.add_module('cls', GPTCls_(model))
    return model_

class GPTModelInfo(ModelInfo):
    def __init__(self, config, args):
        super(GPTModelInfo, self).__init__()
        layernum_list = [config.num_hidden_layers]
        seq_len, hidden_size = config.max_position_embeddings, config.hidden_size
        mixed_precision = mixed_precision_dtype(args.mixed_precision)
        if args.shape_order == "SBH":
            layer_shapes_list = [[[seq_len,-1,hidden_size], [-1,seq_len]]]
        else:
            # TODO: fix fa tensor shape
            layer_shapes_list = [[[-1,seq_len,hidden_size], [-1,seq_len]]]
        layer_dtypes_list = [[mixed_precision, torch.long]]
        module_types = ['embed'] + ['gpt_dec']*config.num_hidden_layers + ['norm', 'cls']
        self.set_layernums(layernum_list)
        self.set_shapes(layer_shapes_list)
        self.set_dtypes(layer_dtypes_list)
        self.set_module_types(module_types)