from flash_attn.models.gpt import GPTLMHeadModel
from flash_attn.modules.block import Block
from galvatron.core import construct_hybrid_parallel_model_api, get_hybrid_parallel_configs_api
from galvatron.models.gpt_fa.GPTModel_sequential import GPTModelInfo, construct_sequential_model, GPTEmbeddings_, GPTPreNorm_, GPTCls_
from galvatron.models.gpt_fa.GPTModel_tensor_parallel import construct_tensor_parallel_model
from flash_attn.modules.embedding import VocabParallelEmbedding, ColumnParallelEmbedding
from galvatron.core import get_args

def get_hybrid_parallel_configs(model_config, training_args):
    hybrid_parallel_configs = get_hybrid_parallel_configs_api(model_config, training_args, GPTModelInfo)
    return hybrid_parallel_configs

def construct_hybrid_parallel_model(model, model_config, training_args, hybrid_parallel_configs):
    args = get_args()
    wrap_block_name = [Block]
    wrap_other_block_name = [VocabParallelEmbedding, ColumnParallelEmbedding, GPTPreNorm_, GPTCls_]
    wrap_checkpoint_block_name=[Block]
    all_block_name=[VocabParallelEmbedding, ColumnParallelEmbedding, Block, GPTPreNorm_, GPTCls_]
    hp_model = construct_hybrid_parallel_model_api(
        model,
        model_config,
        training_args,
        hybrid_parallel_configs,
        GPTModelInfo,
        construct_sequential_model,
        construct_tensor_parallel_model,
        wrap_block_name = wrap_block_name,
        wrap_checkpoint_block_name=wrap_checkpoint_block_name,
        wrap_other_block_name=wrap_other_block_name,
        tied_wte_attr_names = ["embeddings.word_embeddings", ""] if not args.untie_embeddings_and_output_weights else None,
        layernorm_name = ["norm1", "norm2", "ln_f"],
        all_block_name = all_block_name,
    )
    return hp_model

def gpt_model_hp(config, args):
    hybrid_parallel_configs = get_hybrid_parallel_configs(model_config=config, training_args=args)
    if args.local_rank == 0:
        print("Creating Model...")
    gpt_model = GPTLMHeadModel(config)
    model = construct_hybrid_parallel_model(
        model=gpt_model, 
        model_config=config, 
        training_args=args, 
        hybrid_parallel_configs=hybrid_parallel_configs
    )
    return model