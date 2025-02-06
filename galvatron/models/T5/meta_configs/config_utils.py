import json
import os

from transformers import T5Config

from galvatron.utils import dict_join_dirname

# ============= Meta AI Model Config Paths =============
path_dict = {"t5-small": "t5-small.json", "t5-base": "t5-base.json", "t5-large": "t5-large.json", "t5-3B": "t5-3B.json"}


def config_from_meta(model_type) -> T5Config:
    if isinstance(model_type, str):
        global path_dict
        path_dict = dict_join_dirname(path_dict, os.path.dirname(__file__))
        with open(path_dict[model_type]) as f:
            params = json.load(f)
    else:
        assert isinstance(model_type, dict), "model_type must be a string or a dictionary"
        params = model_type
    if "num_decoder_layers" not in params:
        params["num_decoder_layers"] = params["num_layers"]
    if "n_decoder_positions" not in params:
        params["n_decoder_positions"] = params["n_positions"]
    return T5Config(**params)


# ============= Set Model Config and Arguments =============
def set_model_config(config, args, overwrite_args=True):
    config.use_cache = False
    config.model_name = args.model_size
    # ======= Arguments --> Model Config ======
    # Overwrite all model configs by manually set arguments
    if args.set_model_config_manually:
        config.vocab_size = args.vocab_size
        config.d_model = args.hidden_size
        config.d_ff = args.hidden_size * 4
        config.num_heads = args.num_attention_heads
        config.d_kv = args.hidden_size // args.num_attention_heads
        config.num_layers = args.num_encoder_layers
        config.num_decoder_layers = args.num_decoder_layers
        config.n_positions = args.encoder_seq_length
        config.n_decoder_positions = args.decoder_seq_length
        config.dropout_rate = args.hidden_dropout
    # Overwrite layer number only
    else:
        if args.set_layernum_manually:
            config.num_layers = args.num_encoder_layers
            config.num_decoder_layers = args.num_decoder_layers
        if args.set_seqlen_manually:
            config.n_positions = args.encoder_seq_length
            config.n_decoder_positions = args.decoder_seq_length

    # ======= Model Config --> Arguments ======
    overwrite_model_args(config, args)
    # This step is necessary that maintains the consistency of model config and arguments.
    if overwrite_args:  # Overwrite necessary Megatron-LM arguments with the model config
        overwrite_megatron_args(config, args)
    return config


def overwrite_model_args(config, args):
    args.hidden_size = config.hidden_size
    args.num_encoder_layers = config.num_layers
    args.num_decoder_layers = config.num_decoder_layers
    args.num_attention_heads = config.num_attention_heads
    args.encoder_seq_length = config.n_positions
    args.decoder_seq_length = config.n_decoder_positions
    args.vocab_size = config.vocab_size


def overwrite_megatron_args(config, args):
    args.num_encoder_layers = config.num_layers
    args.num_decoder_layers = config.num_decoder_layers
    args.hidden_size = config.d_model
    args.ffn_hidden_size = config.d_ff
    args.encoder_seq_length = config.n_positions
    args.decoder_seq_length = config.n_decoder_positions
    args.vocab_size = config.vocab_size
    args.num_attention_heads = config.num_heads
    args.kv_channels = config.d_kv
    args.norm_epsilon = config.layer_norm_epsilon
    args.hidden_dropout = config.dropout_rate
    args.attention_dropout = config.dropout_rate
    args.vocab_extra_ids = 100
    if getattr(args, "padded_vocab_size", None) is None:
        args.padded_vocab_size = config.vocab_size
        # args.padded_vocab_size = (config.vocab_size + args.make_vocab_size_divisible_by - 1 // args.make_vocab_size_divisible_by * args.make_vocab_size_divisible_by)


# ============= Get Model Name and Layer Configs =============
def model_name(config, args=None):
    if hasattr(args, "profile_mode"):
        if args.profile_mode != "sequence":
            return "%s_seqlen[%d,%d]" % (config.model_name, config.n_positions, config.n_decoder_positions)
    return "%s" % (config.model_name)


def model_layer_configs(config):
    return [
        {"hidden_size": config.hidden_size, "seq_len": config.n_positions, "layer_num": config.num_layers},
        {
            "hidden_size": config.hidden_size,
            "seq_len": config.n_decoder_positions,
            "layer_num": config.num_decoder_layers,
        },
    ]
