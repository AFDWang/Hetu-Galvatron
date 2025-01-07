import pytest
import os
import glob
import json
from tests.utils.mock_configs import (
    initialize_search_engine
)

@pytest.mark.search_engine
@pytest.mark.parametrize("idx, model_type,backend,time_mode,memory_mode,sp_enabled,settle_bsz, settle_chunk, memory_constraint, seqlen_list, sp_space, fine_grained_mode", [
    (0, "llama_search", "hf", "sequence", "sequence", True, 64, 32, 36, [8192], "tp+sp", 1),
    (1, "llama_search", "hf", "sequence", "sequence", True, 64, 8, 36, [8192], "tp+sp", 0),
])
def test_basic_search_flow(base_config_dirs, idx, model_type, backend, time_mode, memory_mode, sp_enabled, settle_bsz, settle_chunk, memory_constraint, seqlen_list, sp_space, fine_grained_mode):
    
    kwargs = {
        "settle_bsz": settle_bsz,
        "settle_chunk": settle_chunk,
        "sp_space": sp_space,
        "memory_constraint": memory_constraint,
        "default_dp_type": "zero2",
        "pipeline_type": "pipedream_flush",
        "async_grad_reduce": False,
        "sequence_parallel": True,
        "fine_grained_mode": fine_grained_mode,
    }

    search_engine = initialize_search_engine(base_config_dirs, model_type, backend, time_mode, memory_mode, sp_enabled, seqlen_list, **kwargs)
    

    
    throughput = search_engine.parallelism_optimization()

    if idx == 0:
        assert abs(throughput - 2.5578320019881313) < 1e-6

        output_dir = base_config_dirs[2]
        json_files = glob.glob(os.path.join(output_dir, '*.json'))
        assert len(json_files) == 1, f"Expected exactly one JSON file, found {len(json_files)}"
        output_file = json_files[0]
        
        filename = os.path.basename(output_file)
        assert filename.startswith('galvatron_config_')
        assert filename.endswith('.json')

        with open(output_file, 'r') as f:
            config = json.load(f)

        expected_fields = [
                "pp_deg", "tp_sizes_enc", "tp_consecutive_flags", 
                "dp_types_enc", "use_sp", "checkpoint", "global_bsz",
                "chunks", "pp_division", "pipeline_type", 
                "default_dp_type", "vtp", "vsp"
            ]
        for field in expected_fields:
            assert field in config, f"Missing field: {field}"

        assert config["pp_deg"] == 2
        assert config["global_bsz"] == 64
        assert config["chunks"] == 32
        assert config["pp_division"] == "14,14"
        assert config["pipeline_type"] == "pipedream_flush"
        assert config["default_dp_type"] == "zero2"
        assert config["vtp"] == 2
        assert config["vsp"] == 0

        array_fields = {
            "tp_sizes_enc": "4",
            "tp_consecutive_flags": "1",
            "dp_types_enc": "0",
            "use_sp": "0"
        }
        
        for field, expected_value in array_fields.items():
            values = config[field].split(',')
            assert len(values) == 28, f"Expected 28 values in {field}, got {len(values)}"
            assert all(v == expected_value for v in values), f"Unexpected values in {field}"

        checkpoint_values = config["checkpoint"].split(',')
        assert len(checkpoint_values) == 28
        expected_checkpoint = "1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0"
        assert config["checkpoint"] == expected_checkpoint
    else:
        assert abs(throughput - 2.3714885035864284) < 1e-6

        output_dir = base_config_dirs[2]
        json_files = glob.glob(os.path.join(output_dir, '*.json'))
        assert len(json_files) == 1, f"Expected exactly one JSON file, found {len(json_files)}"
        output_file = json_files[0]
        
        filename = os.path.basename(output_file)
        assert filename.startswith('galvatron_config_')
        assert filename.endswith('.json')

        with open(output_file, 'r') as f:
            config = json.load(f)

        expected_fields = [
                "pp_deg", "tp_sizes_enc", "tp_consecutive_flags", 
                "dp_types_enc", "use_sp", "checkpoint", "global_bsz",
                "chunks", "pp_division", "pipeline_type", 
                "default_dp_type", "vtp", "vsp"
            ]
        for field in expected_fields:
            assert field in config, f"Missing field: {field}"

        assert config["pp_deg"] == 1
        assert config["global_bsz"] == 64
        assert config["chunks"] == 8
        assert config["pp_division"] == "28"
        assert config["pipeline_type"] == "pipedream_flush"
        assert config["default_dp_type"] == "zero2"
        assert config["vtp"] == 2
        assert config["vsp"] == 0

        array_fields = {
            "tp_sizes_enc": "2",
            "tp_consecutive_flags": "1",
            "dp_types_enc": "1",
            "use_sp": "0",
            "checkpoint": "1",
        }
        
        for field, expected_value in array_fields.items():
            values = config[field].split(',')
            assert len(values) == 28, f"Expected 28 values in {field}, got {len(values)}"