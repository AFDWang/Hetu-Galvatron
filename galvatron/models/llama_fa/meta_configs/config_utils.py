import json
import os

from transformers import GPT2Config, LlamaConfig

from galvatron.utils import dict_join_dirname

# ============= Meta AI Model Config Paths =============
path_dict = {
    "llama-7b": "llama-7b.json",
    "llama-13b": "llama-13b.json",
    "llama-30b": "llama-30b.json",
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
    if "ffn_dim" not in params:
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
    )


def llama_config_to_gpt2_config(llama_config, args) -> GPT2Config:
    return GPT2Config(
        vocab_size=llama_config.vocab_size,
        n_positions=0,  # No absolute position embedding
        n_embd=llama_config.hidden_size,
        n_layer=llama_config.num_hidden_layers,
        n_head=llama_config.num_attention_heads,
        n_inner=llama_config.intermediate_size,
        activation_function="swiglu",  # Hardcode since HF calls it 'silu'
        # Llama doesn't have dropout, idk if it's because they only release the inference code
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=llama_config.rms_norm_eps,
        initializer_range=llama_config.initializer_range,
        bos_token_id=llama_config.bos_token_id,
        eos_token_id=llama_config.eos_token_id,
        # These are new arguments not in the original GPT2Config
        pad_token_id=llama_config.pad_token_id,  # Idk if this does anything
        rms_norm=True,
        rotary_emb_fraction=1.0,
        rotary_emb_interleaved=True,
        tie_word_embeddings=False,
        qkv_proj_bias=False,
        out_proj_bias=False,
        mlp_fc1_bias=False,
        mlp_fc2_bias=False,
        rotary_emb_base=getattr(llama_config, "rotary_emb_base", 10000.0),
        n_head_kv=llama_config.num_key_value_heads,
        use_cache=False,
        fused_bias_fc=True,
        sequence_parallel=hasattr(args, "sequence_parallel") and args.sequence_parallel,
        use_flash_attn=hasattr(args, "use_flash_attn") and args.use_flash_attn,
        max_position_embeddings_data=llama_config.max_position_embeddings,
    )


# ============= Set Model Config and Arguments =============
def set_model_config(config, args, overwrite_args=True):
    config.use_cache = False
    config.fused_bias_fc = True
    config.sequence_parallel = hasattr(args, "sequence_parallel") and args.sequence_parallel
    config.use_flash_attn = hasattr(args, "use_flash_attn") and args.use_flash_attn
    config.resid_pdrop = 0.0
    config.embd_pdrop = 0.0
    config.attn_pdrop = 0.0
    # ======= Arguments --> Model Config ======
    # Overwrite all model configs by manually set arguments
    if args.set_model_config_manually:
        config.vocab_size = args.vocab_size
        config.hidden_size = args.hidden_size
        config.num_hidden_layers = args.num_hidden_layers
        config.num_attention_heads = args.num_attention_heads
        config.max_position_embeddings_data = args.seq_length
    # Overwrite layer number only
    else:
        if args.set_layernum_manually:
            config.num_hidden_layers = args.num_hidden_layers
        if args.set_seqlen_manually:
            config.max_position_embeddings_data = args.seq_length

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
    args.seq_length = config.max_position_embeddings_data
    args.vocab_size = config.vocab_size


def overwrite_megatron_args(config, args):
    args.num_layers = config.num_hidden_layers
    args.hidden_size = config.hidden_size
    args.ffn_hidden_size = config.n_inner
    args.seq_length = config.max_position_embeddings_data
    args.vocab_size = config.vocab_size
    args.num_attention_heads = config.num_attention_heads
    args.kv_channels = args.hidden_size // args.num_attention_heads
    assert abs(config.resid_pdrop - config.embd_pdrop) <= 1e-3, "resid_pdrop should be equal to embd_pdrop"
    args.hidden_dropout = config.resid_pdrop
    args.attention_dropout = config.attn_pdrop
    if getattr(args, "padded_vocab_size", None) is None:
        args.padded_vocab_size = (
            (config.vocab_size + args.make_vocab_size_divisible_by - 1)
            // args.make_vocab_size_divisible_by
            * args.make_vocab_size_divisible_by
        )


# ============= Get Model Name and Layer Configs =============
def model_name(config, args=None):
    if hasattr(args, "profile_mode"):
        if args.profile_mode != "sequence":
            return "hidden%d_head%d_seqlen%d" % (
                config.hidden_size,
                config.num_attention_heads,
                config.max_position_embeddings,
            )
    return "hidden%d_head%d" % (config.hidden_size, config.num_attention_heads)


def model_layer_configs(config):
    return [
        {
            "hidden_size": config.hidden_size,
            "seq_len": config.max_position_embeddings_data,
            "layer_num": config.num_hidden_layers,
        }
    ]
