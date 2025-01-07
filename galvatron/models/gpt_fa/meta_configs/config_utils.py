import os, json
from transformers import GPT2Config
from galvatron.utils import dict_join_dirname

# ============= Meta AI Model Config Paths =============
path_dict =  {
    'gpt-0.3b': 'gpt-0.3b.json',
    'gpt-1.5b': 'gpt-1.5b.json',
    'gpt-2.7b': 'gpt-2.7b.json',
    'gpt-6.7b': 'gpt-6.7b.json',
}

def config_from_meta(model_type) -> GPT2Config:
    if isinstance(model_type, str):
        global path_dict
        path_dict = dict_join_dirname(path_dict, os.path.dirname(__file__))
        with open(path_dict[model_type]) as f:
            params = json.load(f)
    else:
        assert isinstance(model_type, dict), "model_type must be a string or a dictionary"
        params = model_type
    return GPT2Config(**params)

# ============= Set Model Config and Arguments =============
def set_model_config(config, args, overwrite_args=True):
    config.use_cache = False
    config.fused_bias_fc = True
    config.sequence_parallel = hasattr(args, 'sequence_parallel') and args.sequence_parallel
    config.use_flash_attn = hasattr(args, 'use_flash_attn') and args.use_flash_attn
    assert getattr(config, 'fused_dropout_add_ln', False) == False
    # ======= Arguments --> Model Config ======
    # Overwrite all model configs by manually set arguments
    if args.set_model_config_manually:
        config.vocab_size = args.vocab_size
        config.hidden_size = args.hidden_size
        config.num_hidden_layers = args.num_hidden_layers
        config.num_attention_heads = args.num_attention_heads
        config.max_position_embeddings = args.seq_length
        config.resid_pdrop = args.dropout_prob
        config.embd_pdrop = args.dropout_prob
        config.attn_pdrop = args.dropout_prob
    # Overwrite layer number only
    else:
        if args.set_layernum_manually:
            config.num_hidden_layers = args.num_hidden_layers
        if args.set_seqlen_manually:
            config.max_position_embeddings = args.seq_length
    
    # ======= Model Config --> Arguments ======
    overwrite_model_args(config, args)
    # This step is necessary that maintains the consistency of model config and arguments.
    if overwrite_args: # Overwrite necessary Megatron-LM arguments with the model config
        overwrite_megatron_args(config, args)
    return config

def overwrite_model_args(config, args):
    args.hidden_size = config.hidden_size
    args.num_hidden_layers = config.num_hidden_layers
    args.num_attention_heads = config.num_attention_heads
    args.seq_length = config.max_position_embeddings
    args.vocab_size = config.vocab_size

def overwrite_megatron_args(config, args):
    args.num_layers = config.num_hidden_layers
    args.hidden_size = config.hidden_size
    args.ffn_hidden_size = args.hidden_size * 4
    args.seq_length = config.max_position_embeddings
    args.vocab_size = config.vocab_size
    args.num_attention_heads = config.num_attention_heads
    args.kv_channels = args.hidden_size // args.num_attention_heads
    assert abs(config.resid_pdrop - config.embd_pdrop) <= 1e-3, "resid_pdrop should be equal to embd_pdrop"
    args.hidden_dropout = config.resid_pdrop
    args.attention_dropout = config.attn_pdrop
    if getattr(args, "padded_vocab_size", None) is None:
        args.padded_vocab_size = (config.vocab_size + args.make_vocab_size_divisible_by - 1) // args.make_vocab_size_divisible_by * args.make_vocab_size_divisible_by

# ============= Get Model Name and Layer Configs =============
def model_name(config, args=None):
    if hasattr(args,"profile_mode"):
        if args.profile_mode != "sequence":
            return 'hidden%d_head%d_seqlen%d'%(config.hidden_size, config.num_attention_heads, config.max_position_embeddings)
    return 'hidden%d_head%d'%(config.hidden_size, config.num_attention_heads)


def model_layer_configs(config):
    return [
        {
            'hidden_size': config.hidden_size,
            'seq_len': config.max_position_embeddings,
            'layer_num': config.num_hidden_layers
        }
    ]