import pytest
import json
from unittest.mock import patch
from tests.utils.profiler_utils import initialize_model_profile_profiler
from tests.utils.profiler_configs import save_profiler_configs
from tests.utils.search_configs import create_static_time_config, create_batch_time_config, create_sequence_time_config, create_static_memory_config, create_static_memory_config_sp, create_sequence_memory_config_sp
@pytest.fixture
def base_profiler(profiler_model_configs_dir):
    """Create base profiler instance"""
    profiler = initialize_model_profile_profiler(profiler_model_configs_dir, "llama_search", "hf")
    return profiler

@pytest.mark.profiler
@pytest.mark.parametrize("mode,expected_seq_list,config", [
    ("static", [4096], {"profile_batch_size": 32}),
    ("sequence", [128, 256, 384, 512], {
        "profile_min_seq_length": 128,
        "profile_max_seq_length": 512,
        "profile_seq_length_step": 128
    })
])
def test_get_seq_list(base_profiler, mode, expected_seq_list, config):
    """Test sequence list generation in different modes"""
    base_profiler.args.profile_mode = mode
    for key, value in config.items():
        setattr(base_profiler.args, key, value)
    
    assert base_profiler.get_seq_list() == expected_seq_list

@pytest.mark.profiler
@pytest.mark.parametrize("mode,expected_bsz_list,config", [
    ("static", [32], {"profile_batch_size": 32}),
    ("batch", [16, 32, 48, 64], {
        "profile_min_batch_size": 16,
        "profile_max_batch_size": 64,
        "profile_batch_size_step": 16
    }),
])
def test_get_bsz_list(base_profiler, mode, expected_bsz_list, config):
    """Test batch size list generation in different modes"""
    base_profiler.args.profile_mode = mode
    for key, value in config.items():
        setattr(base_profiler.args, key, value)
    
    assert base_profiler.get_bsz_list() == expected_bsz_list

@pytest.mark.profiler
@pytest.mark.parametrize("profile_type,profile_mode,expected_calls", [
    # Memory profiling with static mode
    ("memory", "static", {
        "cmd_count": 24,  # Expected number of os.system calls
    }),
    # Memory profiling with sequence mode
    ("memory", "sequence", {
        "cmd_count": 18,  # Reduced because max_tp_deg=1 in sequence mode, sequence length is 128, 256, 512 (different with computation mode)
    }),
    # Computation profiling
    ("computation", "static", {
        "cmd_count": 2,  # 2 layernum_lists * 2 batch_sizes
    }),
    ("computation", "batch", {
        "cmd_count": 4,  # 2 layernum_lists * 2 batch_sizes
    }),
    ("computation", "sequence", {
        "cmd_count": 8,  # 2 layernum_lists * 4 seq_lengths
    })
    
])
def test_launch_profiling_scripts(base_profiler, profile_type, profile_mode, expected_calls):
    """Test launch_profiling_scripts with different configurations"""
    
    # Configure profiler args
    base_profiler.args.profile_type = profile_type
    base_profiler.args.profile_mode = profile_mode
    
    if profile_type == "computation":
        if profile_mode == "static":
            base_profiler.args.profile_batch_size = 32
        elif profile_mode == "batch":
            base_profiler.args.profile_min_batch_size = 16
            base_profiler.args.profile_max_batch_size = 32
            base_profiler.args.profile_batch_size_step = 16
        elif profile_mode == "sequence":
            base_profiler.args.profile_min_seq_length = 128
            base_profiler.args.profile_max_seq_length = 512
            base_profiler.args.profile_seq_length_step = 128
    elif profile_mode == "sequence":
        if profile_mode == "static":
            base_profiler.args.seq_length = 4096
        else:
            base_profiler.args.profile_min_seq_length = 128
            base_profiler.args.profile_max_seq_length = 512
            base_profiler.args.profile_seq_length_step = 128
    
    # Mock all required methods
    with patch('os.system') as mock_system:
        
        # Execute the function
        base_profiler.launch_profiling_scripts()
        
        # Verify number of calls
        assert mock_system.call_count == expected_calls["cmd_count"]

@pytest.mark.profiler
@pytest.mark.parametrize("mode,config", [
    ("static", {"profile_batch_size": 8, "layernum_min": 2, "layernum_max": 4}),
    ("batch", {"profile_min_batch_size": 1, "profile_max_batch_size": 10, "profile_batch_size_step": 1, "layernum_min": 2, "layernum_max": 4,}),
    ("sequence", {"profile_batch_size": 1, "profile_min_seq_length": 4096, "profile_max_seq_length": 32768, "profile_seq_length_step": 4096, "layernum_min": 1, "layernum_max": 2,})
])
def test_process_computation_profiled_data(base_profiler, profiler_model_configs_dir, mode, config):
    """Test processing of computation profiled data"""
    # Create initial profiling data
    base_profiler.args.mixed_precision = "bf16"
    base_profiler.args.profile_mode = mode
    base_profiler.args.profile_type = "computation"
    save_profiler_configs(
        profiler_model_configs_dir,
        type="computation",
        mode=mode,
        mixed_precision=base_profiler.args.mixed_precision,
        model_name=base_profiler.model_name
    )

    # Configure profiler
    base_profiler.args.profile_mode = mode
    for key, value in config.items():
        setattr(base_profiler.args, key, value)
    
    # Process data
    base_profiler.process_profiled_data()
    
    # Verify results
    config_path = profiler_model_configs_dir / f"computation_profiling_{base_profiler.args.mixed_precision}_{base_profiler.model_name}.json"
    assert config_path.exists()
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Check processed data format
    if mode == "static":
        result = create_static_time_config()
    elif mode == "batch":
        result = create_batch_time_config()
    elif mode == "sequence":
        result = create_sequence_time_config()
    
    for key, value in result.items():
        assert abs(config[key] - value) < 1e-6

@pytest.mark.profiler
@pytest.mark.parametrize("mode,config", [
    ("static", {"profile_batch_size": 8, "layernum_min": 1, "layernum_max": 2, "sequence_parallel": False}),
    ("static", {"profile_batch_size": 8, "layernum_min": 1, "layernum_max": 2, "sequence_parallel": True}),
    ("sequence", {"profile_batch_size": 8, "profile_min_seq_length": 512, "profile_max_seq_length": 8192, "layernum_min": 1, "layernum_max": 2, "sequence_parallel": True}),
])
def test_process_memory_profiled_data(base_profiler, profiler_model_configs_dir, mode, config):
    """Test processing of memory profiled data"""
    # Create initial profiling data
    base_profiler.args.mixed_precision = "bf16"
    base_profiler.args.profile_mode = mode
    base_profiler.args.profile_type = "memory"
    save_profiler_configs(
        profiler_model_configs_dir,
        type="memory",
        mode=mode,
        mixed_precision=base_profiler.args.mixed_precision,
        model_name=base_profiler.model_name,
        sp_mode=config["sequence_parallel"]
    )

    # Configure profiler
    base_profiler.args.profile_mode = mode
    for key, value in config.items():
        setattr(base_profiler.args, key, value)
    
    # Process data
    base_profiler.process_profiled_data()
    
    # Verify results
    config_path = profiler_model_configs_dir / f"memory_profiling_{base_profiler.args.mixed_precision}_{base_profiler.model_name}.json"
    assert config_path.exists()
    
    with open(config_path) as f:
        calc_config = json.load(f)
    
    # Check processed data format
    if mode == "static" and config["sequence_parallel"] == False:
        result = create_static_memory_config()
    elif mode == "static" and config["sequence_parallel"] == True:
        result = create_static_memory_config_sp()
    elif mode == "sequence" and config["sequence_parallel"] == True:
        result = create_sequence_memory_config_sp()
    
    def cmp(calc_config, result):
        if isinstance(calc_config, dict):
            for key, value in result.items():
                cmp(calc_config[key], value)
        else:
            assert abs(calc_config - result) < 1e-6
    cmp(calc_config, result)
