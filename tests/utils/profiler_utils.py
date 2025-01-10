from galvatron.core import GalvatronProfiler
from tests.utils.model_utils import ModelFactory
from tests.models.configs.get_config_json import ConfigFactory

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

@dataclass
class ModelProfileArgs:
    """Arguments for Galvatron Profiling"""
    
    # Profile type and mode
    profile_type: str = "memory"
    profile_mode: str = "static"
    
    # Manual configuration flags
    set_model_config_manually: bool = False
    set_layernum_manually: bool = False
    set_seqlen_manually: bool = False
    
    # Batch size configuration
    profile_batch_size: Optional[int] = None
    profile_min_batch_size: Optional[int] = None
    profile_max_batch_size: Optional[int] = None
    profile_batch_size_step: int = 1
    
    # Sequence length configuration
    profile_min_seq_length: Optional[int] = None
    profile_max_seq_length: Optional[int] = None
    profile_seq_length_step: int = 128
    
    # Layer configuration
    layernum_min: int = 1
    layernum_max: int = 2
    
    # Parallel configuration
    max_tp_deg: int = 8
    profile_dp_type: str = "zero3"
    sequence_parallel: bool = False
    
    # Precision and optimization
    mixed_precision: str = "bf16"
    use_flash_attn: bool = False
    
    # Model configuration
    make_vocab_size_divisible_by: int = 128
    shape_order: str = "SBH"
    
    # Extra arguments
    extra_args_str: str = ""

    def with_updates(self, **kwargs) -> 'ModelProfileArgs':
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    def validate(self):
        """Validate profile arguments"""
        if self.profile_mode == "static":
            if self.profile_batch_size is None:
                raise ValueError("profile_batch_size must be set for static mode")
                
        elif self.profile_mode == "batch":
            if any(x is None for x in [self.profile_min_batch_size, self.profile_max_batch_size]):
                raise ValueError("profile_min_batch_size and profile_max_batch_size must be set for batch mode")
            if self.profile_min_batch_size > self.profile_max_batch_size:
                raise ValueError("profile_min_batch_size must be <= profile_max_batch_size")
                
        elif self.profile_mode == "sequence":
            if any(x is None for x in [self.profile_min_seq_length, self.profile_max_seq_length]):
                raise ValueError("profile_min_seq_length and profile_max_seq_length must be set for sequence mode")
            if self.profile_min_seq_length > self.profile_max_seq_length:
                raise ValueError("profile_min_seq_length must be <= profile_max_seq_length")

        if self.layernum_min > self.layernum_max:
            raise ValueError("layernum_min must be <= layernum_max")

        return True

    def _get_kwargs(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}

def initialize_model_profile_profiler(profiler_model_configs_dir, model_type, backend, **kwargs):
    """Initialize profiler"""

    # Setup search engine
    args = ModelProfileArgs()
    args.with_updates(**kwargs)
    layernum_arg_names = ModelFactory.get_layernum_arg_names(model_type, backend)
    config_json = ConfigFactory.get_config_json(model_type)
    args.model_size = config_json
    config = ModelFactory.create_config(model_type, backend, args, False)

    # Initialize profiler
    profiler = GalvatronProfiler(args)
    profiler.set_profiler_launcher(profiler_model_configs_dir.parent, layernum_arg_names(), model_type)

    return profiler

class HardwareProfileArgs:
    """Arguments for Hardware Profiling"""
    num_nodes: int = 1
    num_gpus_per_node: int = 8
    master_addr: str = '$MASTER_ADDR'
    master_port: str = '$MASTER_PORT'
    node_rank: str = '$RANK'
    
    # Parallel configuration
    max_tp_size: int = 8
    max_pp_deg: int = 8
    
    # Environment configuration
    envs: List[str] = []
    backend: str = 'nccl'
    
    # NCCL test configuration
    nccl_test_dir: str = 'nccl-tests'
    mpi_path: str = '/usr/local/mpi/'
    start_mb: int = 16
    end_mb: int = 512
    scale: int = 2
    hostfile: str = 'hostfile'
    
    # Profiling options
    avg_or_min_or_first: str = 'first'
    overlap_time_multiply: int = 4

def initialize_hardware_profile_profiler(profiler_hardware_configs_dir):
    """Initialize profiler"""

    # Setup search engine
    args = HardwareProfileArgs()
    profiler = GalvatronProfiler(args)
    profiler.set_path(profiler_hardware_configs_dir)
    return profiler

def initialize_runtime_profile_profiler(profiler_model_configs_dir, model_type, backend, **kwargs):
    """Initialize profiler"""

    # Setup search engine
    class DummyArgs:
        def __init__(self):
            self.profile = True
            self.mixed_precision = 'bf16'
            self.set_model_config_manually = False
            self.set_layernum_manually = False
            self.set_seqlen_manually = False
    args = DummyArgs()
    model_layer_configs, model_name = ModelFactory.get_meta_configs(model_type, backend)
    config_json = ConfigFactory.get_config_json(model_type)
    args.model_size = config_json
    config = ModelFactory.create_config(model_type, backend, args, False)
    # Initialize profiler
    profiler = GalvatronProfiler(args)
    profiler.set_profiler_dist(profiler_model_configs_dir.parent, model_layer_configs(config), model_type, rank = 0, profile_ranks = [0], **kwargs)
    
    return profiler