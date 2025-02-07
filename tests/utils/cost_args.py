from dataclasses import dataclass, asdict
from typing import Dict, Any, Callable, Optional
from tests.utils.search_configs import (
    create_static_memory_config,
    create_static_time_config,
    create_batch_time_config,
    create_hardware_configs
)
from galvatron.core.search_engine.search_engine import optimal_chunk_func_default
from galvatron.utils.config_utils import read_allreduce_bandwidth_config, read_p2p_bandwidth_config, remap_config

@dataclass
class MemoryModelArgs:
    parameter_size: float
    tp_activation_per_bsz_dict: Dict[str, float]
    other_memory_pp_off: Dict[str, Dict[str, Dict[str, float]]]
    other_memory_pp_on: Dict[str, Dict[str, Dict[str, float]]]
    pipeline_type: str = 'gpipe'
    mixed_precision: bool = True
    use_zero2_for_dp: int = 0
    use_zero3_for_embed: int = 0
    disable_vtp: int = 0
    max_tp_deg: int = 8
    gpu_num: int = 8
    vsp: int = 0
    optimal_chunk_func: Callable = optimal_chunk_func_default

    @staticmethod
    def convert_keys_to_int(d):
        if isinstance(d, dict):
            new_dict = {}
            for k, v in d.items():
                if isinstance(k, str) and k.isdigit():
                    new_dict[int(k)] = MemoryModelArgs.convert_keys_to_int(v)
                else:
                    new_dict[k] = MemoryModelArgs.convert_keys_to_int(v)
            return new_dict
        return d
    
    def with_updates(self, **kwargs) -> 'MemoryModelArgs':
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    @classmethod
    def from_mock_config(cls) -> 'MemoryModelArgs':
        memory_config = create_static_memory_config()
        memory_config = cls.convert_keys_to_int(memory_config)
        return cls(
            parameter_size=memory_config['layertype_0'][4096]['parameter_size'],
            tp_activation_per_bsz_dict=memory_config['layertype_0'][4096]['tp_activation_per_bsz_dict'],
            other_memory_pp_off={
                'model_states': memory_config['other_memory_pp_off'][4096]['model_states'],
                'activation': memory_config['other_memory_pp_off'][4096]['activation']
            },
            other_memory_pp_on={
                'first_stage': {
                    'model_states': memory_config['other_memory_pp_on_first'][4096]['model_states'],
                    'activation': memory_config['other_memory_pp_on_first'][4096]['activation']
                },
                'last_stage': {
                    'model_states': memory_config['other_memory_pp_on_last'][4096]['model_states'],
                    'activation': memory_config['other_memory_pp_on_last'][4096]['activation']
                }
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class TimeModelArgs:
    parameter_size: float = 48
    microbatch: bool = False
    optimal_chunk_func: Callable = optimal_chunk_func_default
    sequence_length: int = 512
    hidden_size: int = 1024
    forward_computation_time: float = 35 / 24
    bct_fct_coe: float = 2
    extra_overhead: float = 0
    comm_coe_dict: Dict[str, float] = None
    dp_overlap_coe: float = 1.3
    bct_overlap_coe: float = 1.3
    p2p_comm_coe_dict: Optional[Dict[str, float]] = None
    layer_num: Optional[int] = None
    use_zero2_for_dp: int = 0
    mixed_precision: bool = False
    no_comm: bool = False
    costmodel_coe: float = 1.0
    async_grad_reduce: bool = True
    allreduce_dict: Optional[Dict[int, float]] = None
    all2all_dict: Optional[Dict[int, float]] = None
    sp_space: str = 'tp'

    def with_updates(self, **kwargs) -> 'MemoryModelArgs':
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self
    
    @classmethod
    def from_mock_config(cls) -> 'TimeModelArgs':
        static_time = create_static_time_config()
        hardware = create_hardware_configs()
        
        return cls(
            forward_computation_time=static_time['layertype_0_bsz8_seq4096'],
            comm_coe_dict=read_allreduce_bandwidth_config(hardware['allreduce'], 8)[1],
            p2p_comm_coe_dict=read_p2p_bandwidth_config(hardware['p2p'])[1],
            allreduce_dict=remap_config(hardware['sp'], 'allreduce'),
            all2all_dict=remap_config(hardware['sp'], 'all2all'),
            dp_overlap_coe=hardware['overlap']['overlap_coe'],
            bct_overlap_coe=hardware['overlap']['overlap_coe']
        )


    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
