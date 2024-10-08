from flash_attn.models.gpt import GPTLMHeadModel
from flash_attn.modules.block import Block
from galvatron.core import construct_hybrid_parallel_model_api, get_hybrid_parallel_configs_api
from galvatron.models.llama_fa.LlamaModel_sequential import LlamaModelInfo, construct_sequential_model, LlamaEmbeddings_, LlamaPreNorm_, LlamaCls_
from galvatron.models.llama_fa.LlamaModel_tensor_parallel import construct_tensor_parallel_model

def get_hybrid_parallel_configs(model_config, training_args):
    hybrid_parallel_configs = get_hybrid_parallel_configs_api(model_config, training_args, LlamaModelInfo)
    return hybrid_parallel_configs

def construct_hybrid_parallel_model(model, model_config, training_args, hybrid_parallel_configs):
    wrap_block_name = [Block]
    all_block_name=[LlamaEmbeddings_, Block, LlamaPreNorm_, LlamaCls_]
    hp_model = construct_hybrid_parallel_model_api(
        model,
        model_config,
        training_args,
        hybrid_parallel_configs,
        LlamaModelInfo,
        construct_sequential_model,
        construct_tensor_parallel_model,
        wrap_block_name = wrap_block_name,
        all_block_name = all_block_name,
        # tied_wte_attr_names = ["embeddings.word_embeddings", "lm_head"],
        layernorm_name = ["norm1", "norm2", "ln_f"]
    )
    return hp_model

def llama_model_hp(config, args):
    hybrid_parallel_configs = get_hybrid_parallel_configs(model_config=config, training_args=args)
    if args.local_rank == 0:
        print("Creating Model...")
    llama_model = GPTLMHeadModel(config, device='meta' if args.initialize_on_meta else 'cpu')
    model = construct_hybrid_parallel_model(
        model=llama_model, 
        model_config=config, 
        training_args=args, 
        hybrid_parallel_configs=hybrid_parallel_configs
    )
    return model