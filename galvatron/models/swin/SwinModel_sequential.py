import torch
import torch.distributed
import torch.nn as nn
import einops

from megatron.core import tensor_parallel
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.mappings_group import get_tensor_model_parallel_world_size_group
from megatron.core.tensor_parallel.utils import VocabUtility

from galvatron.core import get_args
from galvatron.core.runtime import ModelInfo, mixed_precision_dtype
from galvatron.core.runtime.pipeline import PipeSequential
from galvatron.core.runtime.tensor_parallel import colummn_row_reset_parameters

class SwinPatchEmbedding_(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.patch_embeddings = model.swin.embeddings.patch_embeddings

    def forward(self, tokens):
        hidden_states, _ = self.patch_embeddings(tokens)
        return hidden_states


class SwinPositionEmbedding_(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.position_embeddings = model.swin.embeddings.position_embeddings

    def forward(self, position_ids):
        return self.position_embeddings(position_ids)

class SwinEmeddings_LayerNorm_(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.layernorm = model.swin.embeddings.norm

    def forward(self, hidden_states):
        return self.layernorm(hidden_states)

class SwinEmbeddings_(nn.Module):
    def __init__(self, model):
        super().__init__()
        args = get_args()
        self.patch_dim = model.swin.config.patch_size
        self.patch_embeddings = SwinPatchEmbedding_(model)
        if model.swin.config.use_absolute_embeddings:
            self.position_embeddings = SwinPositionEmbedding_(model)
        else:
            self.position_embeddings = None
        self.norm = SwinEmeddings_LayerNorm_(model)
        self.drop = torch.nn.Dropout(args.hidden_dropout)
        self.sequence_parallel = args.sequence_parallel
        self.clone_scatter_output_in_embedding = args.clone_scatter_output_in_embedding
        self.tp_group = self.patch_embeddings.patch_embeddings.tp_group
        self.sp_group = self.patch_embeddings.patch_embeddings.sp_group
        self.vocab_sp = args.vocab_sp
        if self.vocab_sp:
            self.seq_start_index, self.seq_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                args.seq_length,
                torch.distributed.get_rank(self.sp_group),
                torch.distributed.get_world_size(self.sp_group),
            )

    def forward(self, tokens, labels=None):
        tokens = einops.rearrange(
            tokens,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_dim,
            p2=self.patch_dim,
        )
        if self.vocab_sp:
            tokens = tokens[:, self.seq_start_index : self.seq_end_index, :].contiguous()

        inputs_embeds = self.patch_embeddings(tokens)
        if self.position_embeddings is not None:
            position_embeds = self.position_embeddings(tokens)
            hidden_states = inputs_embeds + position_embeds
        else:
            hidden_states = inputs_embeds
        hidden_states = self.norm(hidden_states)
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

class SwinBlocks_(nn.Module):
    def __init__(self, model, block_idx, layer_idx, has_downsample=False):
        super().__init__()
        self.layer = model.swin.encoder.layers[block_idx].blocks[layer_idx]
        self.has_downsample = has_downsample
    def forward(self, hidden_states, labels=None):
        # attention_mask = get_ltor_masks_and_position_ids(input_ids)
        hidden_states = self.layer(hidden_states)
        return hidden_states

class SwinDownsample_(nn.Module):
    def __init__(self, model, block_idx):
        super().__init__()
        self.downsample = model.swin.encoder.layers[block_idx].downsample

    def forward(self, hidden_states, labels=None):
        hidden_states, _ = self.downsample(hidden_states)
        return hidden_states
    
class SwinPreNorm_(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.LayerNorm = model.swin.layernorm

    def forward(self, hidden_states, labels=None):
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class SwinLoss_(nn.Module):
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
            tp_group=self.tp_group,
        )
        return logits_parallel


class SwinCls_(nn.Module):
    def __init__(self, model, parallel_loss=True, half_entropy=True):
        super().__init__()
        self.sequence_parallel = get_args().sequence_parallel
        self.tp_group = model.classifier.tp_group
        self.sp_group = model.classifier.sp_group
        self.pooler = model.swin.pooler
        self.lm_head = SwinLoss_(model.classifier.weight, self.sequence_parallel, self.tp_group)
        self.clone_scatter_output_in_embedding = get_args().clone_scatter_output_in_embedding
        self.parallel_loss = parallel_loss
        self.half_entropy = half_entropy
        args = get_args()
        if args.entropy_in_fp32:
            self.half_entropy = False
        self.seq_length = args.decoder_seq_length
        self.vocab_sp = args.vocab_sp

    def forward(
        self,
        hidden_states,
        labels,
    ):
        # hidden states shape: [s, b, h] -> [b, h, s] -> [b, h, 1] -> [h, b]
        logits_parallel = self.pooler(hidden_states.permute(1, 2, 0).contiguous()).squeeze(-1).permute(1, 0).contiguous()
        # if self.sequence_parallel:
        #     logits_parallel = tensor_parallel.reduce_scatter_to_sequence_parallel_region_group(hidden_states, self.tp_group)
        #     logits_parallel /= get_tensor_model_parallel_world_size_group(self.tp_group)

        # [h, b] -> [1, b, h]
        logits_parallel = logits_parallel.transpose(0, 1).unsqueeze(0).contiguous()
        logits_parallel = self.lm_head(logits_parallel)

        # [b 1] -> [1 b]
        labels = labels.transpose(0, 1).contiguous()

        # loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), input_ids)
        if not self.parallel_loss:
            output = tensor_parallel.gather_from_tensor_model_parallel_region_group(logits_parallel, self.tp_group)
            if not self.half_entropy:
                logits = output.float()
            else:
                logits = output
            loss = None
            # Shift so that tokens < n predict n
            shift_logits = logits.contiguous()  # logits[:-1, ..., :].contiguous()
            shift_labels = labels.contiguous()  # input_ids[1:, ...].contiguous()
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
                    loss = tensor_parallel.vocab_parallel_cross_entropy(
                        logits_parallel.float(), labels, tp_group=self.tp_group
                    )
                else:
                    loss = tensor_parallel.vocab_parallel_cross_entropy(logits_parallel, labels, tp_group=self.tp_group)
            else:
                if not self.half_entropy:
                    loss = tensor_parallel.vocab_sequence_parallel_cross_entropy(
                        logits_parallel.float(), labels, self.sp_group
                    )
                else:
                    loss = tensor_parallel.vocab_sequence_parallel_cross_entropy(logits_parallel, labels, self.sp_group)
            # loss = loss.mean()
        loss = loss.transpose(0, 1).contiguous()
        return loss


def construct_sequential_model(model, config):
    model_ = PipeSequential()
    model_.add_module("embeddings", SwinEmbeddings_(model))
    for i, d in enumerate(config.depths):
        for j in range(d):
            model_.add_module('encoder_%d_%d'%(i, j), SwinBlocks_(model, i, j, j==d-1 and i!=len(config.depths)-1))
        if i != len(config.depths) - 1:
            model_.add_module('downsample_%d'%(i), SwinDownsample_(model, i))
    model_.add_module("pre_norm", SwinPreNorm_(model))
    model_.add_module("cls", SwinCls_(model))
    SwinLoss_.reset_parameters = colummn_row_reset_parameters
    return model_


class SwinModelInfo(ModelInfo):
    def __init__(self, config, args):
        super(SwinModelInfo, self).__init__()
        layernum_list = []
        layer_shapes_list = []
        layer_dtypes_list = []
        module_types = []
        mixed_precision = mixed_precision_dtype(args.mixed_precision)
        for i in range(len(config.depths)):
            seq_len, hidden_size = (config.image_size // config.patch_size // (2 ** i)) ** 2, config.embed_dim * (2 ** i)
            layer_shapes_list += [[[seq_len, -1, hidden_size]]]
            layer_dtypes_list += [[mixed_precision]]
            # if i < len(config.depths) - 1: # downsample
            #     layernum_list += [config.depths[i] - 1, 1]
            #     layer_shapes_list += [[[-1, seq_len // 4, hidden_size * 2]]]
            #     layer_dtypes_list += [[mixed_precision]]
            # else:
            layernum_list += [config.depths[i]]
        module_types = ['embed'] 
        for i in range(len(config.depths)):
            module_types += ['swin_enc']*config.depths[i]
            if i < len(config.depths) - 1:
                module_types += ['swin_downsample']
        module_types += ['pooler', 'cls']
        
        self.set_layernums(layernum_list)
        self.set_shapes(layer_shapes_list)
        self.set_dtypes(layer_dtypes_list)
        self.set_module_types(module_types)
