import torch

class RuntimeArgs:
    def __init__(self, model_type, rank, checkpoint_dir=None, backend="hf"):
        self.use_flash_attn = False
        self.global_train_batch_size = 16
        self.default_dp_type = "ddp"
        self.chunks = 2
        self.pp_deg = 1
        self.global_tp_deg = 1
        self.global_tp_consec = 1
        self.sdp = 0
        self.embed_sdp = 0
        self.global_checkpoint = 0
        self.vocab_tp = 1
        self.vocap_sp = 0
        self.pipeline_type = "gpipe"
        self.untie_embeddings_and_output_weights = False
        self.mixed_precision = "fp32"
        self.adam_weight_decay = 0.01
        self.learning_rate = 1e-5
        self.model_size = model_type + "_test"
        self.set_model_config_manually = False
        self.set_layernum_manually = False
        self.set_seqlen_manually = False
        self.make_vocab_size_divisible_by = 1
        self.local_rank = rank
        self.galvatron_config_path = None
        self.use_ulysses = False
        self.distributed_checkpoint = False
        self.initialize_on_meta = True
        if checkpoint_dir is not None:  
            self.load = checkpoint_dir["converted"]
        else:
            self.load = None
        self.load_iteration = 0
        self.lr = 1e-5
        self.adam_weight_decay = 0.01
        self.sequence_parallel = False
        self.no_persist_layer_norm = False
        self.apply_layernorm_1p = False
        self.params_dtype = torch.float
        self.overlap_p2p_comm = False
        self.num_experts = None
        self.rotary_interleaved = False
        self.bias_gelu_fusion = False # TODO: check if this is correct
        self.bias_swiglu_fusion = False # TODO: check if this is correct
        self.squared_relu = False
        self.init_method_xavier_uniform = False 
        self.group_query_attention = False
        self.num_query_groups = None
        self.add_qkv_bias = False
        self.add_bias_linear = True
        self.rotary_percent = 1.0
        self.rotary_seq_len_interpolation_factor = None
        self.openai_gelu = False
        self.onnx_safe = False
        self.swiglu = False
        self.clone_scatter_output_in_embedding = True
        self.entropy_in_fp32 = True
        self.async_grad_reduce = True
        self.reduce_in_fp32 = True
        self.profile_forward = 0
        self.norm_epsilon = 1e-5

        if backend == "hf":
            self.shape_order = "SBH"
        else:
            self.shape_order = "BSH"