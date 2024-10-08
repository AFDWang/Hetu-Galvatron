from torch.nn import LayerNorm
from transformers import GPT2LMHeadModel
from galvatron.core import construct_hybrid_parallel_model_api, get_hybrid_parallel_configs_api
from galvatron.models.gpt_hf.GPTModel_sequential import GPTModelInfo, construct_sequential_model, GPTEmbeddings_, GPTPreNorm_, GPTCls_
from galvatron.models.gpt_hf.GPTModel_tensor_parallel import construct_tensor_parallel_model, GPTLayer_tp
from galvatron.core.tensor_parallel import ParallelMLP, ParallelAttention
from galvatron.models.gpt_hf.GPTModel_checkpoint import load_gpt_module

def get_hybrid_parallel_configs(model_config, training_args):
    hybrid_parallel_configs = get_hybrid_parallel_configs_api(model_config, training_args, GPTModelInfo)
    return hybrid_parallel_configs

def construct_hybrid_parallel_model(model, model_config, training_args, hybrid_parallel_configs):
    wrap_block_name = [GPTLayer_tp]
    wrap_checkpoint_block_name=[GPTLayer_tp]
    all_block_name=[GPTEmbeddings_, GPTLayer_tp, GPTPreNorm_, GPTCls_]
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
        wrap_other_block_name=['wte','wpe','lm_head'],
        # tied_wte_attr_names=['wte', 'lm_head'],
        layernorm_name=['LayerNorm', "ln_f"],
        all_block_name=all_block_name,
        load_module_func=load_gpt_module,
    )
    return hp_model

def gpt_model_hp(config, args):
    hybrid_parallel_configs = get_hybrid_parallel_configs(model_config=config, training_args=args)
    if args.local_rank == 0:
        print("Creating Model...")
    gpt_model = GPT2LMHeadModel(config)
    model = construct_hybrid_parallel_model(
        model=gpt_model, 
        model_config=config, 
        training_args=args, 
        hybrid_parallel_configs=hybrid_parallel_configs
    )
    return model
