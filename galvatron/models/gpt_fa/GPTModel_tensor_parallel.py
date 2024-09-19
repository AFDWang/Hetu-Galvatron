from galvatron.core import get_args
from flash_attn.models.gpt import create_mixer_cls, create_mlp_cls
from flash_attn.modules.embedding import ParallelGPT2Embeddings
try:
    from flash_attn.ops.fused_dense import ColumnParallelLinear
except ImportError:
    ColumnParallelLinear = None
    
def construct_tensor_parallel_model(model, config, tp_groups_enc):
    args=get_args()
    factory_kwargs = {
        'device': 'meta' if hasattr(args, 'initialize_on_meta') and args.initialize_on_meta else 'cpu',
        'dtype': None
    }
    model.transformer.embeddings = ParallelGPT2Embeddings(
                config.hidden_size,
                args.padded_vocab_size,
                config.max_position_embeddings,
                process_group=tp_groups_enc[0].group,
                sequence_parallel=args.sequence_parallel,
                **factory_kwargs,
            )
    setattr(model.transformer.embeddings, "process_group", tp_groups_enc[0].group)
    
    if ColumnParallelLinear is None:
        raise ImportError("fused_dense_lib is not installed")
    model.lm_head = ColumnParallelLinear(
                config.hidden_size,
                args.padded_vocab_size,
                tp_groups_enc[-1].group,
                bias=getattr(config, "lm_head_bias", False),
                sequence_parallel=getattr(config, "sequence_parallel", True),
                **factory_kwargs,
            )
    for i in range(config.num_hidden_layers):
        layer = model.transformer.layers[i]
        setattr(layer, 'mixer', create_mixer_cls(config, layer_idx=i, process_group=tp_groups_enc[i+1].group, **factory_kwargs)(config.hidden_size))
        setattr(layer, 'mlp', create_mlp_cls(config, layer_idx=i, process_group=tp_groups_enc[i+1].group, **factory_kwargs)(config.hidden_size))
        setattr(layer, 'process_group', tp_groups_enc[i+1].group)
    return model