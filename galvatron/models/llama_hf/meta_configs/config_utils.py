import json
import os

from transformers import GPT2Config, LlamaConfig

from galvatron.utils import dict_join_dirname

# ============= Meta AI Model Config Paths =============
path_dict = {
    "llama-0.3b": "llama-0.3b.json",
    "llama-7b": "llama-7b.json",
    "llama-13b": "llama-13b.json",
    "llama-30b": "llama-30b.json",
    "llama2-70b": "llama2-70b.json",
    "qwen2.5-7b": "qwen2.5-7b.json",
    "qwen2.5-72b": "qwen2.5-72b.json",
    "qwen2.5-1.5b": "qwen2.5-1.5b.json",
    "qwen2.5-3b": "qwen2.5-3b.json",
}


def config_from_meta(model_type) -> LlamaConfig:
    if isinstance(model_type, str):
        global path_dict
        path_dict = dict_join_dirname(path_dict, os.path.dirname(__file__))
        with open(path_dict[model_type]) as f:
            params = json.load(f)
    else:
        assert isinstance(model_type, dict), "model_type must be a string or a dictionary"
        params = model_type
    if "n_kv_heads" not in params:
        params["n_kv_heads"] = None
    # 只有在ffn_dim不存在时才设置默认值
    if "ffn_dim" not in params:
        if model_type.startswith("qwen"):
            # 对于Qwen模型，可能需要特殊的计算方式
            params["ffn_dim"] = params["dim"] * 5.5  # Qwen模型的ffn_dim约为hidden_size的5.5倍左右
        else:
            # 对于其他模型，使用原来的计算方式
            params["ffn_dim"] = (
                (params["dim"] * 8 // 3 + params["multiple_of"] - 1) // params["multiple_of"] * params["multiple_of"]
            )
    return LlamaConfig(
        hidden_size=params["dim"],
        intermediate_size=params["ffn_dim"],
        num_attention_heads=params["n_heads"],
        num_hidden_layers=params["n_layers"],
        rms_norm_eps=params["norm_eps"],
        num_key_value_heads=params["n_kv_heads"],
        max_position_embeddings=params["n_positions"],
        vocab_size=params["vocab_size"],
    )


# ============= Set Model Config and Arguments =============
def set_model_config(config, args, overwrite_args=True):
    config.use_cache = False
    config.model_name = args.model_size
    # ======= Arguments --> Model Config ======
    # Overwrite all model configs by manually set arguments
    if args.set_model_config_manually:
        config.vocab_size = args.vocab_size
        config.hidden_size = args.hidden_size
        # 根据模型类型使用不同的ffn尺寸计算方式
        if args.model_size.startswith("qwen"):
            # 对于Qwen模型，使用固定的ffn_dim值
            config.intermediate_size = args.ffn_hidden_size if args.ffn_hidden_size is not None else args.hidden_size * 8 // 3
        else:
            # 对于其他模型，使用原来的计算方式
            config.intermediate_size = args.hidden_size * 8 // 3
        config.num_hidden_layers = args.num_hidden_layers
        config.num_attention_heads = args.num_attention_heads
        config.max_position_embeddings = args.seq_length
    # Overwrite layer number only
    else:
        if args.set_layernum_manually:
            config.num_hidden_layers = args.num_hidden_layers
        if args.set_seqlen_manually:
            config.max_position_embeddings = args.seq_length

    # ======= Model Config --> Arguments ======
    overwrite_model_args(config, args)
    # This step is necessary that maintains the consistency of model config and arguments.
    if overwrite_args:  # Overwrite necessary Megatron-LM arguments with the model config
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
    args.ffn_hidden_size = config.intermediate_size
    args.seq_length = config.max_position_embeddings
    args.vocab_size = config.vocab_size
    args.num_attention_heads = config.num_attention_heads
    args.kv_channels = args.hidden_size // args.num_attention_heads
    args.norm_epsilon = config.rms_norm_eps
    args.hidden_dropout = 0.0
    args.attention_dropout = 0.0
    args.add_bias_linear = False
    args.swiglu = True
    if getattr(args, "padded_vocab_size", None) is None:
        args.padded_vocab_size = config.vocab_size
        # args.padded_vocab_size = (config.vocab_size + args.make_vocab_size_divisible_by - 1 // args.make_vocab_size_divisible_by * args.make_vocab_size_divisible_by)
    if config.num_key_value_heads != config.num_attention_heads:
        args.group_query_attention = True
        args.num_query_groups = config.num_key_value_heads


# ============= Get Model Name and Layer Configs =============
def model_name(config, args=None):
    if hasattr(args, "profile_mode"):
        if args.profile_mode != "sequence":
            return "%s_seqlen%d" % (config.model_name, config.max_position_embeddings)
    return "%s" % (config.model_name)


def model_layer_configs(config):
    return [
        {
            "hidden_size": config.hidden_size,
            "seq_len": config.max_position_embeddings,
            "layer_num": config.num_hidden_layers,
        }
    ]
