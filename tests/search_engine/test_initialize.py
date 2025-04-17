import pytest
from tests.utils.search_configs import (
    initialize_search_engine
)

@pytest.mark.search_engine
@pytest.mark.parametrize("model_type,backend", [
    ("llama_search", "hf"),
])
@pytest.mark.parametrize("time_mode,memory_mode,sp_enabled", [
    ("static", "static", False),
    ("batch", "static", True),
    ("sequence", "sequence", True),
])
def test_set_cost_models(base_config_dirs, base_log_dirs, model_type, backend, time_mode, memory_mode, sp_enabled):
    """Test setting both time and memory cost models"""
    search_engine = initialize_search_engine(base_config_dirs, base_log_dirs, model_type, backend, time_mode, memory_mode, sp_enabled)

    # Verify time cost models
    assert hasattr(search_engine, 'model_args_list')
    assert hasattr(search_engine, 'train_args_list')
    assert hasattr(search_engine, 'parallel_args_list')
    assert hasattr(search_engine, 'profile_model_args_list')
    assert hasattr(search_engine, 'profile_hardware_args_list')
    assert len(search_engine.model_args_list) == search_engine.num_layertype
    assert len(search_engine.train_args_list) == search_engine.num_layertype
    assert len(search_engine.parallel_args_list) == search_engine.num_layertype
    assert len(search_engine.profile_model_args_list) == search_engine.num_layertype
    assert len(search_engine.profile_hardware_args_list) == search_engine.num_layertype
    # Verify specific time cost model properties
    assert search_engine.model_args_list[0].seq_length == 4096
    assert search_engine.train_args_list[0].mixed_precision == True
    assert search_engine.parallel_args_list[0].sequence_parallel == sp_enabled
