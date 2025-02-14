import torch
from transformers.models.swin.modeling_swin import SwinDropPath
from megatron.core.tensor_parallel import ColumnParallelLinear, VocabParallelEmbedding
from megatron.training.arguments import core_transformer_config_from_args
from torch import nn
from torch.nn import LayerNorm

from galvatron.core import get_args
from galvatron.core.runtime.tensor_parallel import AttnMaskType, AttnType, ParallelAttention, ParallelMLP

# from megatron.model.fused_layer_norm import MixedFusedLayerNorm as LayerNorm


def window_partition(input_feature, window_size, batch_dim = 0):
    """
    Partitions the given input into windows.
    """
    if batch_dim == 0:
        batch_size, height, width, num_channels = input_feature.shape
        input_feature = input_feature.view(
            batch_size, height // window_size, window_size, width // window_size, window_size, num_channels
        )
        windows = input_feature.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, num_channels)
    else:
        height, width, batch_size, num_channels = input_feature.shape
        input_feature = input_feature.view(
            height // window_size, window_size, width // window_size, window_size, batch_size, num_channels
        )
        windows = input_feature.permute(0, 2, 1, 3, 4, 5).contiguous().view(window_size, window_size, -1, num_channels)
    return windows


def window_reverse(windows, window_size, height, width):
    """
    Merges windows to produce higher resolution features.
    """
    num_channels = windows.shape[-1]
    windows = windows.view(height // window_size, width // window_size, window_size, window_size, -1, num_channels)
    windows = windows.permute(0, 2, 1, 3, 4, 5).contiguous().view(height, width, -1, num_channels)
    return windows


class SwinAttention_tp(nn.Module):
    def __init__(self, config, layer_number, dim, num_attention_head, tp_group=None, sp_group=None):
        super().__init__()
        args=get_args()
        args.num_attention_heads = num_attention_head
        args.hidden_size = dim
        args.kv_channels = dim // num_attention_head
        self.use_ulysses = sp_group.size > 1
        megatron_config = core_transformer_config_from_args(args)
        self.tp_group = tp_group.group if tp_group is not None else None
        self.sp_group = sp_group.group if sp_group is not None else None
        self.attention = ParallelAttention(
            megatron_config,
            layer_number,
            attention_type=AttnType.self_attn,
            attn_mask_type=AttnMaskType.padding,
            tp_group=self.tp_group,
            sp_group=self.sp_group,
            use_ulysses=self.use_ulysses,
        )
        self.drop_path = SwinDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
    
    def forward(self, hidden_states, attention_mask, batch_size=None):
        seq_len, bsz, hs = hidden_states.shape
        if attention_mask is None:
            attention_mask = torch.zeros((bsz, 1, seq_len, seq_len), dtype=torch.bool, device=hidden_states.device)
        else:
            attention_mask = attention_mask.repeat(batch_size, 1, 1, 1).to(hidden_states.device)

        input_tensor = hidden_states
        hidden_states, bias = self.attention(hidden_states, attention_mask)
        if bias is not None:
            hidden_states = hidden_states + bias
        hidden_states = self.drop_path(hidden_states)
        hidden_states = hidden_states + input_tensor

        return hidden_states


class SwinMlp_tp(nn.Module):
    def __init__(self, config, dim, tp_group=None):
        super().__init__()
        args=get_args()
        args.hidden_size = dim
        args.ffn_hidden_size = dim * args.mlp_ratio
        megatron_config = core_transformer_config_from_args(get_args())
        self.tp_group = tp_group.group if tp_group is not None else None
        self.mlp = ParallelMLP(megatron_config, tp_group=self.tp_group)

    def forward(self, hidden_states):
        input_tensor = hidden_states
        hidden_states, bias = self.mlp(hidden_states)
        if bias is not None:
            hidden_states = hidden_states + bias
        hidden_states = hidden_states + input_tensor
        return hidden_states


class SwinBlock_tp(nn.Module):
    def __init__(self, config, layer_number, dim, input_resolution, num_heads, shift_size=0, tp_group=None, sp_group=None):
        super().__init__()
        self.shift_size = shift_size
        self.window_size = config.window_size
        self.input_resolution = input_resolution

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.attention = SwinAttention_tp(config, layer_number, dim, num_heads, tp_group=tp_group, sp_group=sp_group)
        self.intermediate = SwinMlp_tp(config, dim, tp_group=tp_group)
        
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            height, width = self.input_resolution
            img_mask = torch.zeros((1, height, width, 1))
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            mask_windows = window_partition(img_mask, self.window_size, batch_dim=0)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size) # win_num, ws^2
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # win_num, ws^2, ws^2
            attn_mask = attn_mask.unsqueeze(1).type(torch.bool) # win_num, 1, ws^2, ws^2
        else:
            attn_mask = None
        
        self.attention_mask = attn_mask
        self.layernorm_before = LayerNorm(dim, eps=config.layer_norm_eps)
        self.layernorm_after = LayerNorm(dim, eps=config.layer_norm_eps)
    
    # TODO: support sequence parallel
    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        height, width = self.input_resolution

        dim, batch_size, channels = hidden_states.size()
        shortcut = hidden_states

        hidden_states = self.layernorm_before(hidden_states)
        hidden_states = hidden_states.view(height, width, batch_size, channels)

        # cyclic shift
        if self.shift_size > 0:
            shifted_hidden_states = torch.roll(hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(0, 1))
        else:
            shifted_hidden_states = hidden_states

        # partition windows
        hidden_states_windows = window_partition(shifted_hidden_states, self.window_size, batch_dim=1)
        hidden_states_windows = hidden_states_windows.view(self.window_size * self.window_size, -1, channels)

        attention_output = self.attention(
            hidden_states_windows,
            self.attention_mask,
            batch_size
        )

        attention_windows = attention_output.view(self.window_size, self.window_size, -1, channels)
        shifted_windows = window_reverse(attention_windows, self.window_size, height, width)

        # reverse cyclic shift
        if self.shift_size > 0:
            attention_windows = torch.roll(shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(0, 1))
        else:
            attention_windows = shifted_windows

        attention_windows = attention_windows.view(height * width, batch_size, channels)

        hidden_states = shortcut + attention_windows

        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = hidden_states + layer_output
        # print('output:', layer_output.size())
        return layer_output


class SwinDownsample_tp(nn.Module):
    def __init__(self, config, dim, input_resolution, tp_group=None, sp_group=None):
        super().__init__()
        self.input_resolution = input_resolution
        megatron_config = core_transformer_config_from_args(get_args())
        self.reduction = ColumnParallelLinear(
            dim * 4,
            dim * 2,
            config=megatron_config,
            init_method=megatron_config.init_method,
            gather_output=True,
            bias=False,
            tp_group=tp_group.group,
            sp_group=sp_group.group,
        )
        self.layernorm = LayerNorm(dim * 4, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        height, width = self.input_resolution
        height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
        dim, batch_size, channels = hidden_states.shape
        hidden_states = hidden_states.view(height, width, batch_size, channels)
        hidden_states_0 = hidden_states[0::2, 0::2, :, :]
        hidden_states_1 = hidden_states[1::2, 0::2, :, :]
        hidden_states_2 = hidden_states[0::2, 1::2, :, :]
        hidden_states_3 = hidden_states[1::2, 1::2, :, :]
        hidden_states = torch.cat([hidden_states_0, hidden_states_1, hidden_states_2, hidden_states_3], dim=-1)
        hidden_states = hidden_states.view(height_downsampled * width_downsampled, batch_size, channels * 4)
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.reduction(hidden_states)
        return hidden_states

def build_swinblock_list(config, dim, input_resolution, depth, num_heads, tp_gen=None, sp_gen=None, layer_num=0):
    return nn.ModuleList(
        [
            SwinBlock_tp(
                config=config,
                layer_number=layer_num,
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                shift_size=0 if (i % 2 == 0) else config.window_size // 2,
                tp_group=tp_gen.__next__() if tp_gen is not None else None,
                sp_group=sp_gen.__next__() if sp_gen is not None else None,
            )
            for i in range(depth)
        ]
    )


def construct_tensor_parallel_model(model, config, tp_groups_enc, sp_groups_enc):
    tp_gen = tp_groups_enc.__iter__()
    sp_gen = sp_groups_enc.__iter__()
    layer_num = 0
    tp_gen.__next__()
    sp_gen.__next__()

    for i, swinlayer in enumerate(model.swin.encoder.layers):
        new_layers = build_swinblock_list(swinlayer.config, swinlayer.dim, swinlayer.blocks[0].input_resolution, 
            swinlayer.config.depths[i], swinlayer.config.num_heads[i], tp_gen, sp_gen, layer_num)
        layer_num += swinlayer.config.depths[i]
        setattr(swinlayer, 'blocks', new_layers)
        if i != len(config.depths) - 1:
            setattr(swinlayer, 'downsample', SwinDownsample_tp(swinlayer.config, swinlayer.dim, swinlayer.blocks[0].input_resolution, tp_gen.__next__(), sp_gen.__next__()))
    args = get_args()
    args.hidden_size = config.embed_dim
    megatron_config = core_transformer_config_from_args(get_args())
    setattr(
        model.swin.embeddings,
        "patch_embeddings",
        ColumnParallelLinear(
            args.patch_size * args.patch_size * config.num_channels,
            megatron_config.hidden_size,
            config=megatron_config,
            gather_output=True,
            bias=True,
            init_method=megatron_config.init_method,
            tp_group=tp_groups_enc[0].group,
            sp_group=sp_groups_enc[0].group,
        ),
    )
    if config.use_absolute_embeddings:
        setattr(
            model.swin.embeddings,
            "position_embeddings",
            VocabParallelEmbedding(
                args.seq_length,
                megatron_config.hidden_size,
                config=megatron_config,
                init_method=megatron_config.init_method,
                tp_group=tp_groups_enc[0].group,
                sp_group=sp_groups_enc[0].group,
            ),
        )
    setattr(
        model,
        "classifier",
        ColumnParallelLinear(
            megatron_config.hidden_size * (2 ** (len(config.depths) - 1)),
            args.num_classes,
            config=megatron_config,
            init_method=megatron_config.init_method,
            bias=True,
            tp_group=tp_groups_enc[-1].group,
            sp_group=sp_groups_enc[-1].group,
        ),
    )

    return model
