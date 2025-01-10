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
def test_set_cost_models(base_config_dirs, model_type, backend, time_mode, memory_mode, sp_enabled):
    """Test setting both time and memory cost models"""
    search_engine = initialize_search_engine(base_config_dirs, model_type, backend, time_mode, memory_mode, sp_enabled)

    # Verify time cost models
    assert hasattr(search_engine, 'timecost_model_args_list')
    assert len(search_engine.timecost_model_args_list) == search_engine.num_layertype
    
    # Verify specific time cost model properties
    assert search_engine.timecost_model_args_list[0]['sequence_length'] == 4096
    assert len(search_engine.timecost_model_args_list[0]) == 21
    
    # Verify memory cost models
    assert hasattr(search_engine, 'memcost_model_args_list')
    assert len(search_engine.memcost_model_args_list) == search_engine.num_layertype
    
    # Verify specific memory cost model properties
    assert search_engine.memcost_model_args_list[0]['sequence_parallel'] == sp_enabled
    assert len(search_engine.memcost_model_args_list[0]) == 17
