from dataclasses import dataclass

@dataclass
class SearchArgs:
    """Mock search arguments for testing"""
    def __init__(self):
        # Model config settings
        self.set_model_config_manually: int = 0
        self.set_layernum_manually: int = 0
        self.set_seqlen_manually: int = 0
        
        # Cluster settings
        self.num_nodes: int = 1
        self.num_gpus_per_node: int = 8
        self.memory_constraint: int = 24
    
        # Batch size settings
        self.min_bsz: int = 8
        self.max_bsz: int = 10240
        self.recommend_min_bsz: int = 0
        self.settle_bsz: int = -1
        self.settle_chunk: int = -1
        self.bsz_scale: int = 8
    
        # Search space settings
        self.search_space: str = "full"
        self.sp_space: str = "tp"
        
        # Disable flags
        self.disable_dp: int = 0
        self.disable_tp: int = 0
        self.disable_vtp: int = 0
        self.disable_pp: int = 0
        self.disable_sdp: int = 0
        self.disable_ckpt: int = 0
        self.disable_tp_consec: int = 0
    
        # Parallel degree limits
        self.max_tp_deg: int = 8
        self.max_pp_deg: int = 8
        
        # Parallel settings
        self.default_dp_type: str = "ddp"
        self.embed_sdp: int = 0
        self.mixed_precision: str = "bf16"
        self.pipeline_type: str = "gpipe"
    
        # Cost model settings
        self.use_pipeline_costmodel: int = 1
        self.costmodel_coe: float = 1.0
    
        # Sequence parallel settings
        self.sequence_parallel: bool = False
        self.global_memory_buffer: bool = True
        self.async_grad_reduce: bool = True
        
        # Vocab settings
        self.make_vocab_size_divisible_by: int = 128
        
        # Search mode settings
        self.fine_grained_mode: int = 1
        self.time_profile_mode: str = "static"
        self.memory_profile_mode: str = "static"

        # Path
        self.memory_profiling_path: str = None
        self.time_profiling_path: str = None
        self.allreduce_bandwidth_config_path: str = None
        self.p2p_bandwidth_config_path: str = None
        self.overlap_coe_path: str = None
        self.sp_time_path: str = None
        self.output_config_path: str = None


