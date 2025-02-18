import torch.distributed
import torch.nn as nn
import torch
from galvatron.core.pipeline import PipeSequential
from galvatron.core import mixed_precision_dtype, ModelInfo
from galvatron.core import get_args
from megatron.core import tensor_parallel
from galvatron.core.tensor_parallel import colummn_row_reset_parameters
import math

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
        self.sp_group = self.wte.sp_group
    def forward(self, input_ids, cu_seqlens = None):
        args = get_args()
        if args.use_flexSP:
            self.sp_group = args.sp_group
        if not args.use_packing:
            tokens = input_ids
            labels = input_ids.clone()
            # tokens = input_ids[:, :-1].contiguous()
            # labels = input_ids[:, 1:].contiguous()
            position_ids = torch.arange(0, tokens.size(-1), dtype=torch.long, device=tokens.device)
        else:
            real_seqlen = cu_seqlens[-1] - (len(cu_seqlens) - 1)
            world_size = torch.distributed.get_world_size()
            target_len = ((real_seqlen - 1) // world_size + 1) * world_size
            # tokens = torch.cat([input_ids[cu_seqlens[i]:(cu_seqlens[i+1] -1)] for i in range(len(cu_seqlens) - 1)]).contiguous().cuda()
            # labels = torch.cat([input_ids[cu_seqlens[i]+1: cu_seqlens[i+1]] for i in range(len(cu_seqlens) - 1)]).contiguous().cuda()
            tokens = input_ids.clone()
            labels = input_ids.clone()
            cu_seqlens = cu_seqlens.cuda()
            # cu_seqlens = cu_seqlens - torch.arange(len(cu_seqlens), device = cu_seqlens.device)
            # tokens = torch.cat((tokens, torch.zeros(target_len - cu_seqlens[-1], dtype = tokens.dtype, device = tokens.device)))
            # labels = torch.cat((labels, torch.zeros(target_len - cu_seqlens[-1], dtype = tokens.dtype, device = tokens.device)))
            # cu_seqlens[-1] = target_len
            position_ids = torch.cat([torch.arange(cu_seqlens[i+1] - cu_seqlens[i]) for i in range(len(cu_seqlens) - 1)]).contiguous().cuda()
        tot_seq_len = tokens.size(-1)
        seq_partit = max(torch.distributed.get_world_size(self.tp_group), torch.distributed.get_world_size(self.sp_group))
        seq_partit_rank = max(torch.distributed.get_rank(self.tp_group), torch.distributed.get_rank(self.sp_group))
        ds_sequence_parallel = torch.distributed.get_world_size(self.sp_group) > 1
        avg_seq_len = tot_seq_len // seq_partit
        if ds_sequence_parallel:
            position_ids = position_ids[avg_seq_len * seq_partit_rank : avg_seq_len * (seq_partit_rank + 1)]
            if not args.use_packing:
                tokens = tokens[: , avg_seq_len * seq_partit_rank : avg_seq_len * (seq_partit_rank + 1)]
            else:
                tokens = tokens[avg_seq_len * seq_partit_rank : avg_seq_len * (seq_partit_rank + 1)]
        
        # position_ids = position_ids.unsqueeze(0) #[1, S, d]
        inputs_embeds = self.wte(tokens)
        position_embeds = self.wps(position_ids)
        position_embeds = position_embeds.unsqueeze(1) #[S, 1, d]
        # hidden_states = inputs_embeds + position_embeds
        hidden_states = inputs_embeds
        # hidden_states = self.drop(hidden_states)
        if not args.use_packing:
            # [b, s, h] -> [s, b, h]
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        else:
            # [s, h] -> [s, 1, h]
            hidden_states = hidden_states.unsqueeze(1)
        if ds_sequence_parallel:
            if self.clone_scatter_output_in_embedding:
                hidden_states = hidden_states.clone()
            hidden_states += position_embeds
            with tensor_parallel.get_cuda_rng_tracker().fork():
                hidden_states = self.drop(hidden_states)
        elif self.sequence_parallel:
            hidden_states = tensor_parallel.scatter_to_sequence_parallel_region_group(hidden_states, self.tp_group)
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            if self.clone_scatter_output_in_embedding:
                hidden_states = hidden_states.clone()
            hidden_states += position_embeds
            with tensor_parallel.get_cuda_rng_tracker().fork():
                hidden_states = self.drop(hidden_states)
        else:
            hidden_states = self.drop(hidden_states)
        labels = labels.clone()
        return hidden_states, labels, cu_seqlens

class GPTLayers_(nn.Module):
    def __init__(self, model, layer_idx):
        super().__init__()
        model = model.transformer
        self.layer = model.h[layer_idx]
        self.tp_group = self.layer.tp_group
        self.sp_group = self.layer.sp_group

    def forward(self, hidden_states, input_ids, cu_seqlens = None):
        #flash-attn doesn't need attention_mask
        attention_mask = get_ltor_masks_and_position_ids(input_ids) if not get_args().use_flash_attn else None
        max_seqlen = torch.max(cu_seqlens[1:] - cu_seqlens[:-1]).item() if cu_seqlens != None else None
        hidden_states = self.layer(hidden_states, attention_mask = attention_mask, 
                                        cu_seqlens = cu_seqlens, max_seqlen = max_seqlen) # , position_ids = position_ids)
        input_ids = input_ids.clone()
        return hidden_states, input_ids, cu_seqlens

class GPTPreNorm_(nn.Module):
    def __init__(self, model):
        super().__init__()
        model = model.transformer
        self.ln_f = model.ln_f

    def forward(self, hidden_states, input_ids, cu_seqlens = None):
        hidden_states = self.ln_f(hidden_states)
        input_ids = input_ids.clone()
        return hidden_states, input_ids, cu_seqlens
    
class GPTLoss_(nn.Module):
    def __init__(self, weight, sequence_parallel, tp_group):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())
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
        self.sp_group = model.lm_head.sp_group
        self.seq_data_group = model.lm_head.seq_data_group
        
    def forward(self, hidden_states, input_ids, cu_seqlens = None):
        sp_group = self.sp_group if not get_args().use_flexSP else get_args().sp_group
        sp_worldsize = torch.distributed.get_world_size(sp_group)
        ds_sequence_parallel = sp_worldsize > 1
        if cu_seqlens is None:
            sub_seq_len = input_ids.shape[1] // sp_worldsize
        else:
            sub_seq_len = input_ids.shape[0] // sp_worldsize
        sp_rank = torch.distributed.get_rank(sp_group)
        local_bsz = input_ids.shape[0] if cu_seqlens is None else cu_seqlens.shape[0]-1

        global_bsz = get_args().global_train_batch_size
        if ds_sequence_parallel:
            if cu_seqlens is None:
                input_ids = input_ids[: , sub_seq_len * sp_rank : sub_seq_len * (sp_rank + 1)]
            else:
                input_ids = input_ids[sub_seq_len * sp_rank : sub_seq_len * (sp_rank + 1)]
        from torch.nn import CrossEntropyLoss
        if ds_sequence_parallel:
            from megatron.core import sequence_parallel
        if not self.sequence_parallel:
            hidden_states = tensor_parallel.copy_to_tensor_model_parallel_region_group(hidden_states, self.tp_group)
        
        logits_parallel = self.lm_head(hidden_states)
        # [b s] -> [s b]
        if cu_seqlens is None:
            input_ids = input_ids.transpose(0,1).contiguous()
        else:
            input_ids = input_ids.unsqueeze(1)
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
            # loss_fct = CrossEntropyLoss()
            loss_fct = sequence_parallel.vocab_sequence_parallel_cross_entropy if ds_sequence_parallel else CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            if ds_sequence_parallel:
                loss = loss_fct(shift_logits, shift_labels, sp_group).mean()
            else:
                loss = loss_fct(shift_logits, shift_labels)
        else:
            if not ds_sequence_parallel:
                if not self.half_entorpy:
                    loss = tensor_parallel.vocab_parallel_cross_entropy(logits_parallel.float(), input_ids, tp_group = self.tp_group)
                else:
                    loss = tensor_parallel.vocab_parallel_cross_entropy(logits_parallel, input_ids, tp_group = self.tp_group)
                loss = loss.mean()
            else:
                if not self.half_entorpy:
                    loss = sequence_parallel.vocab_sequence_parallel_cross_entropy(logits_parallel.float(), input_ids, sp_group)
                else:
                    loss = sequence_parallel.vocab_sequence_parallel_cross_entropy(logits_parallel, input_ids, sp_group)
                loss = loss.mean()
        # if sp_rank == 0:            
        #     print(f"from rank {torch.distributed.get_rank()}: global_bsz {global_bsz}, local_bsz {local_bsz}") 
        if get_args().use_packing:
            loss /= local_bsz
        loss  = (loss * local_bsz * torch.distributed.get_world_size(self.seq_data_group)) / global_bsz #grad balance
        return loss

def construct_sequential_model(model, config):
    model_ = PipeSequential()
    model_.add_module('embeddings', GPTEmbeddings_(model))
    for i in range(config.num_hidden_layers):
        enc = GPTLayers_(model, i)
        model_.add_module('layer_%d'%i, enc)
    model_.add_module('prenorm', GPTPreNorm_(model))
    model_.add_module('cls', GPTCls_(model))
    GPTLoss_.reset_parameters = colummn_row_reset_parameters
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