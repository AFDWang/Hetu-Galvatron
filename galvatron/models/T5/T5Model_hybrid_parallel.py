from torch.nn import LayerNorm
from transformers import T5ForConditionalGeneration
from galvatron.core import construct_hybrid_parallel_model_api, get_hybrid_parallel_configs_api
from galvatron.models.T5.T5Model_sequential import T5ModelInfo, construct_sequential_model, T5EncoderEmbeddings_, T5DecoderEmbeddings_, T5EncoderPreNorm_, T5DecoderPreNorm_, T5Cls_
from galvatron.models.T5.T5Model_tensor_parallel import construct_tensor_parallel_model, T5EncoderLayer_tp, T5DecoderLayer_tp
from galvatron.core.tensor_parallel import ParallelMLP, ParallelAttention
from galvatron.models.T5.T5Model_checkpoint import load_t5_module
from galvatron.core import get_args
from galvatron.core.initialize import init_empty_weights
from galvatron.models.T5.meta_configs import config_from_meta, set_model_config

def get_hybrid_parallel_configs(model_config, training_args):
    hybrid_parallel_configs = get_hybrid_parallel_configs_api(model_config, training_args, T5ModelInfo)
    return hybrid_parallel_configs

def construct_hybrid_parallel_model(model, model_config, training_args, hybrid_parallel_configs):
    args = get_args()
    wrap_block_name = [T5EncoderLayer_tp, T5DecoderLayer_tp]
    wrap_other_block_name = [T5EncoderEmbeddings_, T5DecoderEmbeddings_, T5EncoderPreNorm_, T5DecoderPreNorm_, T5Cls_]
    wrap_checkpoint_block_name=[T5EncoderLayer_tp, T5DecoderLayer_tp]
    all_block_name=[T5EncoderEmbeddings_, T5DecoderEmbeddings_, T5EncoderLayer_tp, T5EncoderPreNorm_, T5DecoderLayer_tp, T5DecoderPreNorm_, T5Cls_]
    hp_model = construct_hybrid_parallel_model_api(
        model,
        model_config,
        training_args,
        hybrid_parallel_configs,
        T5ModelInfo,
        construct_sequential_model,
        construct_tensor_parallel_model,
        wrap_block_name=wrap_block_name,
        wrap_checkpoint_block_name=wrap_checkpoint_block_name,
        wrap_other_block_name=wrap_other_block_name,
        # TODO: sync decoder embeddings 
        tied_wte_attr_names=['', ''] if not args.untie_embeddings_and_output_weights else None,
        layernorm_name=['LayerNorm'],
        all_block_name=all_block_name,
        load_module_func=load_t5_module,
    )
    return hp_model

def get_t5_config(args, overwrite_args=True):
    config = config_from_meta(args.model_size)
    config = set_model_config(config, args, overwrite_args)
    if hasattr(args, 'local_rank') and args.local_rank == 0:
        print(config)
    return config

def t5_model_hp(config, args):
    hybrid_parallel_configs = get_hybrid_parallel_configs(model_config=config, training_args=args)
    if args.local_rank == 0:
        print("Creating Model...")
    if args.initialize_on_meta:
        with init_empty_weights():
            t5_model = T5ForConditionalGeneration(config)
    else:
        t5_model = T5ForConditionalGeneration(config)

    model = construct_hybrid_parallel_model(
        model=t5_model, 
        model_config=config, 
        training_args=args, 
        hybrid_parallel_configs=hybrid_parallel_configs
    )
    return model
