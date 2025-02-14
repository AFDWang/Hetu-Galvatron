import json
import os

from transformers import SwinConfig

from galvatron.utils import dict_join_dirname

# ============= Meta AI Model Config Paths =============
path_dict = {"swin-large": "swin-large-patch4-window12-384.json",
             "swin-huge": "swin-huge-patch4-window7-224.json"}


def config_from_meta(model_type) -> SwinConfig:
    if isinstance(model_type, str):
        global path_dict
        path_dict = dict_join_dirname(path_dict, os.path.dirname(__file__))
        with open(path_dict[model_type]) as f:
            params = json.load(f)
    else:
        assert isinstance(model_type, dict), "model_type must be a string or a dictionary"
        params = model_type
    params["ffn_dim"] = params["embed_dim"] * params["mlp_ratio"]
    # params["n_positions"] = (params["image_size"] // params["patch_size"]) ** 2 * params["num_channels"]
    return SwinConfig(**params)


# ============= Set Model Config and Arguments =============
def set_model_config(config, args, overwrite_args=True):
    config.use_cache = False
    config.model_name = args.model_size
    # ======= Arguments --> Model Config ======
    # Overwrite all model configs by manually set arguments
    if args.set_model_config_manually:
        config.image_size = args.image_size
        config.patch_size = args.patch_size
        config.num_channels = args.num_channels
        config.embed_dim = args.embed_dim
        config.mlp_ratio = 4
        config.depths = args.depths
        config.num_heads = args.num_heads
        config.window_size = args.window_size
        config.num_labels = args.num_classes
        # config.n_positions = args.seq_length
        config.drop_path_rate = args.drop_path_rate
    # Overwrite layer number only
    else:
        if args.set_layernum_manually:
            config.depths = args.depths
            config.num_heads = args.num_heads
        if args.set_seqlen_manually:
            config.image_size = args.image_size
            config.patch_size = args.patch_size
            config.num_channels = args.num_channels
            # config.n_positions = args.seq_length

    # ======= Model Config --> Arguments ======
    overwrite_model_args(config, args)
    # This step is necessary that maintains the consistency of model config and arguments.
    if overwrite_args:  # Overwrite necessary Megatron-LM arguments with the model config
        overwrite_megatron_args(config, args)
    return config


def overwrite_model_args(config, args):
    args.embed_dim = config.embed_dim
    args.depths = config.depths
    args.num_heads = config.num_heads
    args.window_size = config.window_size
    args.num_channels = config.num_channels
    args.image_size = config.image_size
    args.patch_size = config.patch_size
    args.num_classes = config.num_labels
    args.drop_path_rate = config.drop_path_rate
    args.mlp_ratio = config.mlp_ratio
    # if config.n_positions is not None:
    #     assert config.n_positions == (config.image_size // config.patch_size) ** 2 * config.num_channels
    #     args.seq_length = config.n_positions
    # else:
    #     config.n_positions = (config.image_size // config.patch_size) ** 2 * config.num_channels
    #     args.seq_length = config.n_positions


def overwrite_megatron_args(config, args):
    args.hidden_size = config.embed_dim
    args.num_layers = sum(config.depths)
    args.num_attention_heads = config.num_heads
    args.ffn_hidden_size = config.ffn_dim
    # args.seq_length = config.n_positions
    # args.kv_channels = config.d_kv
    args.norm_epsilon = config.layer_norm_eps
    args.hidden_dropout = config.hidden_dropout_prob
    args.attention_dropout = config.attention_probs_dropout_prob

# ============= Get Model Name and Layer Configs =============
def model_name(config, args=None):
    if hasattr(args, "profile_mode"):
        if args.profile_mode != "sequence":
            return "%s_seqlen[%d,%d]" % (config.model_name, config.n_positions, config.n_decoder_positions)
    return "%s" % (config.model_name)


def model_layer_configs(config):
    return [
        {
            'hidden_size': config.embed_dim * (2**i),
            'seq_len': (config.image_size // config.patch_size // (2**i)) ** 2,
            'layer_num': config.depths[i]
        } for i in range(len(config.depths))
    ]
