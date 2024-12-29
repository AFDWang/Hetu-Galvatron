import torch.nn as nn
import torch
from galvatron.core.pipeline import PipeSequential
from galvatron.core import mixed_precision_dtype, ModelInfo
from einops import rearrange

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    dropout_add_layer_norm = None

try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm_parallel_residual
except ImportError:
    dropout_add_layer_norm_parallel_residual = None
    
try:
    from flash_attn.ops.rms_norm import RMSNorm, dropout_add_rms_norm
except ImportError:
    RMSNorm, dropout_add_rms_norm = None, None

try:
    from flash_attn.ops.rms_norm import dropout_add_rms_norm_parallel_residual
except ImportError:
    dropout_add_rms_norm_parallel_residual = None


class GPTEmbeddings_(nn.Module):
    def __init__(self, model):
        super().__init__()
        model = model.transformer
        attrs = ['embeddings', 'sequence_parallel']
        for key in attrs:
            setattr(self, key, getattr(model, key))
        setattr(self, 'process_group', getattr(self.embeddings, "process_group"))

    def forward(self, tokens, position_ids=None, attention_mask=None, labels=None):
        # tokens = input_ids[:, :-1].contiguous()
        # labels = input_ids[:, 1:].contiguous()
        embedding_kwargs = ({'combine_batch_seqlen_dim': True}
                            if self.process_group is not None and self.sequence_parallel else {})
        if position_ids == None:
            position_ids = torch.arange(0, tokens.size(-1), dtype=torch.long, device=tokens.device)
            position_ids = position_ids.unsqueeze(0)
        hidden_states = self.embeddings(tokens, position_ids=position_ids, **embedding_kwargs)
        return hidden_states

class GPTLayers_(nn.Module):
    def __init__(self, model, layer_idx):
        super().__init__()
        model = model.transformer
        self.layer = model.layers[layer_idx]
        
        attrs = ['prenorm', 'parallel_block', 'sequence_parallel']
        for key in attrs:
            setattr(self, key, getattr(model, key))
        setattr(self, 'process_group', getattr(self.layer, "process_group"))
        if self.prenorm:
            self.dropout1 = self.layer.dropout1
            self.layer.dropout1 = nn.Identity()
        
    def forward(self, hidden_states, position_ids=None, attention_mask=None, labels=None):
        mixer_kwargs = ({'seqlen': labels.shape[1]}
                        if self.process_group is not None and self.sequence_parallel else {})
        if self.prenorm:
            assert(not self.parallel_block)
            hidden_states, residual = self.layer(hidden_states,
                                            mixer_kwargs=mixer_kwargs)
            hidden_states = self.dropout1(hidden_states)
            hidden_states = hidden_states + residual
        else:
            hidden_states = self.layer(hidden_states, mixer_kwargs=mixer_kwargs)
        return hidden_states

class GPTPreNorm_(nn.Module):
    def __init__(self, model):
        super().__init__()
        model = model.transformer
        attrs = ['ln_f', 'prenorm']
        for key in attrs:
            setattr(self, key, getattr(model, key))

    def forward(self, hidden_states, position_ids=None, attention_mask=None, labels=None):
        if self.prenorm:
            hidden_states = self.ln_f(hidden_states)
        return hidden_states

class GPTCls_(nn.Module):
    def __init__(self, model):
        super().__init__()
        attrs = ['lm_head', 'config', 'project_out']
        for key in attrs:
            setattr(self, key, getattr(model, key))
        self.sequence_parallel = self.lm_head.sequence_parallel
        self.process_group = self.lm_head.process_group

    def forward(self, hidden_states, position_ids=None, attention_mask=None, labels=None):
        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)
        lm_logits = self.lm_head(hidden_states)
        if not self.sequence_parallel:
            lm_logits = rearrange(lm_logits, "b s d -> (b s) d")
        from flash_attn.losses.cross_entropy import CrossEntropyLoss
        loss_fn = CrossEntropyLoss(inplace_backward=True, process_group = self.process_group)
        loss = loss_fn(lm_logits, labels.view(-1).long())
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
        layer_shapes_list = [[[-1,seq_len,hidden_size]]]
        layer_dtypes_list = [[mixed_precision]]
        module_types = ['embed'] + ['gpt_dec']*config.num_hidden_layers + ['norm', 'cls']
        self.set_layernums(layernum_list)
        self.set_shapes(layer_shapes_list)
        self.set_dtypes(layer_dtypes_list)
        self.set_module_types(module_types)