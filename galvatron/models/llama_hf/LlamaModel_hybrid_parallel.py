from transformers import LlamaForCausalLM
from galvatron.core import construct_hybrid_parallel_model_api, get_hybrid_parallel_configs_api
from galvatron.models.llama_hf.LlamaModel_sequential import LlamaModelInfo, construct_sequential_model, LlamaEmbeddings_, LlamaPreNorm_, LlamaCls_
from galvatron.models.llama_hf.LlamaModel_tensor_parallel import construct_tensor_parallel_model, LlamaLayer_tp
from galvatron.models.llama_hf.LlamaModel_checkpoint import load_llama_module

# from megatron.model.rms_norm import RMSNorm as LlamaRMSNorm

def get_hybrid_parallel_configs(model_config, training_args):
    hybrid_parallel_configs = get_hybrid_parallel_configs_api(model_config, training_args, LlamaModelInfo)
    return hybrid_parallel_configs

def construct_hybrid_parallel_model(model, model_config, training_args, hybrid_parallel_configs):
    wrap_block_name = [LlamaLayer_tp]
    wrap_checkpoint_block_name=[LlamaLayer_tp]
    wrap_other_block_name = [LlamaEmbeddings_, LlamaPreNorm_, LlamaCls_]
    all_block_name = [LlamaEmbeddings_, LlamaLayer_tp, LlamaPreNorm_, LlamaCls_]
    hp_model = construct_hybrid_parallel_model_api(
        model,
        model_config,
        training_args,
        hybrid_parallel_configs,
        LlamaModelInfo,
        construct_sequential_model,
        construct_tensor_parallel_model,
        wrap_block_name=wrap_block_name,
        wrap_checkpoint_block_name=wrap_checkpoint_block_name,
        wrap_other_block_name=wrap_other_block_name,
        # tied_wte_attr_names=['embed_tokens', 'lm_head'],
        layernorm_name = ['LayerNorm', "norm"],
        all_block_name = all_block_name,
        load_module_func=load_llama_module,
    )
    return hp_model

def llama_model_hp(config, args):
    hybrid_parallel_configs = get_hybrid_parallel_configs(model_config=config, training_args=args)
    if args.local_rank == 0:
        print("Creating Model...")
    llama_model = LlamaForCausalLM(config)
    model = construct_hybrid_parallel_model(
        model=llama_model, 
        model_config=config, 
        training_args=args, 
        hybrid_parallel_configs=hybrid_parallel_configs
    )
    return model