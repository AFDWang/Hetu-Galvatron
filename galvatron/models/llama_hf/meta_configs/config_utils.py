import os, json
from transformers import GPT2Config, LlamaConfig
from galvatron.utils import dict_join_dirname

# ============= Meta AI Model Config Paths =============
path_dict =  {
    'llama-0.3b': 'llama-0.3b.json',
    'llama-7b': 'llama-7b.json',
    'llama-13b': 'llama-13b.json',
    'llama-30b': 'llama-30b.json',
}

def config_from_meta(model_type) -> LlamaConfig:
    global path_dict
    path_dict = dict_join_dirname(path_dict, os.path.dirname(__file__))
    with open(path_dict[model_type]) as f:
        params = json.load(f)
    return LlamaConfig(
        hidden_size=params['dim'], intermediate_size= (params['dim'] * 8 // 3 + params['multiple_of'] - 1) // params['multiple_of'] * params['multiple_of'],
        num_attention_heads=params['n_heads'],
        num_hidden_layers=params['n_layers'],
        rms_norm_eps=params['norm_eps']
    )
    
# ============= Set Model Config and Arguments =============
def set_model_config(config, args, overwrite_args=True):
    # ======= Arguments --> Model Config ======
    # Overwrite all model configs by manually set arguments
    if args.set_model_config_manually:
        config.vocab_size = args.vocab_size
        config.hidden_size = args.hidden_size
        config.intermediate_size = 4 * args.hidden_size
        config.num_hidden_layers = args.num_hidden_layers
        config.num_attention_heads = args.num_attention_heads
        config.max_position_embeddings = args.seq_length
    # Overwrite layer number only
    elif args.set_layernum_manually:
        config.num_hidden_layers = args.num_hidden_layers
    
    # ======= Model Config --> Arguments ======
    # This step is necessary that maintains the consistency of model config and arguments.
    # Overwrite the model arguments with the model config
    overwrite_model_args(config, args)
    
    if overwrite_args: # Overwrite necessary Megatron-LM arguments with the model config
        overwrite_megatron_args(config, args)
    return config

def overwrite_megatron_args(config, args):
    args.hidden_size = config.hidden_size
    args.num_layers = config.num_hidden_layers
    args.num_attention_heads = config.num_attention_heads
    args.max_position_embeddings = config.max_position_embeddings
    args.use_cpu_initialization = True
    args.swiglu = True

# Need to overwrite the arguments with the model config
def overwrite_model_args(config, args):
    args.hidden_size = config.hidden_size
    args.seq_length = config.max_position_embeddings
    args.num_hidden_layers = config.num_hidden_layers
    args.vocab_size = config.vocab_size
    args.num_attention_heads = config.num_attention_heads
    args.kv_channels = args.hidden_size // args.num_attention_heads

# ============= Get Model Name and Layer Configs =============
def model_name(config, args=None):
    return 'hidden%d_head%d_seqlen%d'%(config.hidden_size, config.num_attention_heads, config.max_position_embeddings)

def model_layer_configs(config):
    return [
        {
            'hidden_size': config.hidden_size,
            'seq_len': config.max_position_embeddings,
            'layer_num': config.num_hidden_layers
        }
    ]