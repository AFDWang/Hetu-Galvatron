import torch.nn as nn
import torch
from galvatron.core.pipeline import PipeSequential
from galvatron.core import mixed_precision_dtype, ModelInfo
from galvatron.core import get_args

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
        self.drop = model.drop
        # self.sequence_parallel = get_args().sequence_parallel
        
    def forward(self, input_ids):
        position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0)
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wps(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        # [b, s, h] -> [s, b, h]
        hidden_states = hidden_states.permute(1, 0, 2).contiguous()
        input_ids = input_ids.clone()
        return hidden_states, input_ids

class GPTLayers_(nn.Module):
    def __init__(self, model, layer_idx_start, layer_idx_end):
        super().__init__()
        model = model.transformer
        self.layers = model.h[layer_idx_start:layer_idx_end]

    def forward(self, hidden_states, input_ids):
        attention_mask = get_ltor_masks_and_position_ids(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask = attention_mask) # , position_ids = position_ids)
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

class GPTCls_(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.lm_head = model.lm_head

    def forward(self, hidden_states, input_ids):
        # [s, b, h] -> [b, s, h]
        hidden_states = hidden_states.permute(1, 0, 2).contiguous()
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        loss = None
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        # Flatten the tokens
        from torch.nn import CrossEntropyLoss
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        
        # print(shift_logits.shape, shift_labels.shape)
        loss = loss_fct(shift_logits, shift_labels)
        return loss

def construct_sequential_model(model, config):
    model_ = PipeSequential()
    model_.add_module('embeddings', GPTEmbeddings_(model))
    for i in range(config.num_hidden_layers):
        enc = GPTLayers_(model, i, i + 1)
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