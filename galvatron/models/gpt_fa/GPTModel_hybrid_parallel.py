from flash_attn.models.gpt import GPTLMHeadModel
from flash_attn.modules.block import Block
from flash_attn.modules.embedding import ColumnParallelEmbedding, VocabParallelEmbedding

from galvatron.core import (
    RuntimeProfiler,
    construct_hybrid_parallel_model_api,
    get_args,
    get_hybrid_parallel_configs_api,
)
from galvatron.models.gpt_fa.GPTModel_sequential import (
    GPTCls_,
    GPTEmbeddings_,
    GPTModelInfo,
    GPTPreNorm_,
    construct_sequential_model,
)
from galvatron.models.gpt_fa.GPTModel_tensor_parallel import construct_tensor_parallel_model
from galvatron.models.gpt_fa.meta_configs import config_from_meta, model_layer_configs, model_name, set_model_config


def get_hybrid_parallel_configs(model_config, training_args):
    hybrid_parallel_configs = get_hybrid_parallel_configs_api(model_config, training_args, GPTModelInfo)
    return hybrid_parallel_configs


def construct_hybrid_parallel_model(model, model_config, training_args, hybrid_parallel_configs):
    args = get_args()
    wrap_block_name = [Block]
    wrap_other_block_name = [VocabParallelEmbedding, ColumnParallelEmbedding, GPTPreNorm_, GPTCls_]
    wrap_checkpoint_block_name = [Block]
    all_block_name = [VocabParallelEmbedding, ColumnParallelEmbedding, Block, GPTPreNorm_, GPTCls_]
    hp_model = construct_hybrid_parallel_model_api(
        model,
        model_config,
        training_args,
        hybrid_parallel_configs,
        GPTModelInfo,
        construct_sequential_model,
        construct_tensor_parallel_model,
        wrap_block_name=wrap_block_name,
        wrap_checkpoint_block_name=wrap_checkpoint_block_name,
        wrap_other_block_name=wrap_other_block_name,
        tied_wte_attr_names=(
            ["embeddings.word_embeddings", ""] if not args.untie_embeddings_and_output_weights else None
        ),
        layernorm_name=["norm1", "norm2", "ln_f"],
        all_block_name=all_block_name,
    )
    return hp_model


def get_gpt_config(args, overwrite_args=True):
    config = config_from_meta(args.model_size)
    config = set_model_config(config, args, overwrite_args)
    if hasattr(args, "local_rank") and args.local_rank == 0:
        print(config)
    return config


def gpt_model_hp(config, args):
    hybrid_parallel_configs = get_hybrid_parallel_configs(model_config=config, training_args=args)
    if args.local_rank == 0:
        print("Creating Model...")
    gpt_model = GPTLMHeadModel(config, device="meta" if args.initialize_on_meta else "cpu")
    model = construct_hybrid_parallel_model(
        model=gpt_model, model_config=config, training_args=args, hybrid_parallel_configs=hybrid_parallel_configs
    )
    return model


def get_runtime_profiler(args, path, config, start_iter=10, end_iter=20):
    profiler = RuntimeProfiler(args)
    profiler.set_profiler_dist(
        path, model_layer_configs(config), model_name(config), start_iter=start_iter, end_iter=end_iter
    )
    return profiler
