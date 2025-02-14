from torch.nn import LayerNorm
from transformers import SwinForImageClassification

from galvatron.core import (
    RuntimeProfiler,
    construct_hybrid_parallel_model_api,
    get_args,
    get_hybrid_parallel_configs_api,
)
from galvatron.core.runtime.initialize import init_empty_weights
from galvatron.models.swin.meta_configs import config_from_meta, model_layer_configs, model_name, set_model_config
from galvatron.models.swin.SwinModel_checkpoint import load_swin_module
from galvatron.models.swin.SwinModel_sequential import (
    SwinCls_,
    SwinPatchEmbedding_,
    SwinPositionEmbedding_,
    SwinEmeddings_LayerNorm_,
    SwinPreNorm_,
    SwinModelInfo,
    construct_sequential_model,
)
from galvatron.models.swin.SwinModel_tensor_parallel import (
    SwinBlock_tp,
    construct_tensor_parallel_model,
    SwinDownsample_tp,
)


def get_hybrid_parallel_configs(model_config, training_args):
    hybrid_parallel_configs = get_hybrid_parallel_configs_api(model_config, training_args, SwinModelInfo)
    return hybrid_parallel_configs


def construct_hybrid_parallel_model(model, model_config, training_args, hybrid_parallel_configs):
    args = get_args()
    wrap_block_name = [SwinBlock_tp]
    wrap_other_block_name = [SwinPatchEmbedding_, SwinPositionEmbedding_, SwinEmeddings_LayerNorm_, SwinDownsample_tp, SwinPreNorm_, SwinCls_]
    wrap_checkpoint_block_name = [SwinBlock_tp]
    all_block_name = [
        SwinPatchEmbedding_,
        SwinPositionEmbedding_,
        SwinEmeddings_LayerNorm_,
        SwinBlock_tp,
        SwinDownsample_tp,
        SwinPreNorm_,
        SwinCls_,
    ]
    hp_model = construct_hybrid_parallel_model_api(
        model,
        model_config,
        training_args,
        hybrid_parallel_configs,
        SwinModelInfo,
        construct_sequential_model,
        construct_tensor_parallel_model,
        wrap_block_name=wrap_block_name,
        wrap_checkpoint_block_name=wrap_checkpoint_block_name,
        wrap_other_block_name=wrap_other_block_name,
        tied_wte_attr_names=None,
        layernorm_name=["layernorm", "layernorm_before", "layernorm_after"],
        all_block_name=all_block_name,
        load_module_func=load_swin_module,
        meta_init_buffer=False,
    )
    return hp_model


def get_swin_config(args, overwrite_args=True):
    config = config_from_meta(args.model_size)
    config = set_model_config(config, args, overwrite_args)
    if hasattr(args, "local_rank") and args.local_rank == 0:
        print(config)
    return config


def swin_model_hp(config, args):
    if args.sequence_parallel:
        assert False, "Sequence parallel is not supported for Swin"
    if args.use_flash_attn:
        assert False, "Flash attention is not supported for Swin"
    hybrid_parallel_configs = get_hybrid_parallel_configs(model_config=config, training_args=args)
    if args.local_rank == 0:
        print("Creating Model...")
    if args.initialize_on_meta:
        with init_empty_weights():
            swin_model = SwinForImageClassification(config)
    else:
        swin_model = SwinForImageClassification(config)

    model = construct_hybrid_parallel_model(
        model=swin_model, model_config=config, training_args=args, hybrid_parallel_configs=hybrid_parallel_configs,
    )
    return model


def get_runtime_profiler(args, path, config, start_iter=10, end_iter=20):
    profiler = RuntimeProfiler(args)
    profiler.set_profiler_dist(
        path, model_layer_configs(config), model_name(config), start_iter=start_iter, end_iter=end_iter
    )
    return profiler
