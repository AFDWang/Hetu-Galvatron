import pytest
import json
from unittest.mock import patch, mock_open
import os
from tests.utils.profiler_utils import initialize_hardware_profile_profiler
from tests.utils.profiler_configs import save_profiler_configs
from tests.utils.search_configs import create_static_time_config, create_batch_time_config, create_sequence_time_config, create_static_memory_config, create_static_memory_config_sp, create_sequence_memory_config_sp

@pytest.fixture
def base_profiler(profiler_hardware_configs_dir):
    """Create base profiler instance"""
    profiler = initialize_hardware_profile_profiler(profiler_hardware_configs_dir)
    return profiler

@pytest.mark.profiler
@pytest.mark.parametrize("num_nodes,num_gpus_per_node,expected_calls,expected_sp_calls", [
    # 11 = 1 + 5 + 1 + 3 + 1 build + ar + build + a2a + rm
    # 9 = 1 + 3 + 1 + 3 + 1 build + ar + build + a2a + rm
    (1, 8, 11, 9), 
    (2, 8, 13, 11),
])
def test_nccl_hardware_profile(base_profiler, num_nodes, num_gpus_per_node, expected_calls, expected_sp_calls):
    """Test NCCL hardware profile"""
    base_profiler.args.num_nodes = num_nodes
    base_profiler.args.num_gpus_per_node = num_gpus_per_node
    
    path = base_profiler.path
    output = [
        "# nThread 1 nGpus 8 minBytes 1048576 maxBytes 1073741824 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0\n",
        "#\n",
        "# Using devices\n",
        "#  Rank  0 Group  0 Pid 1341852 on job-83e1033f-9636-44b3-bf8b-2b627707b95f-worker-0 device  0 [0x07] NVIDIA A100-SXM4-40GB\n",
        "#  Rank  1 Group  0 Pid 1341852 on job-83e1033f-9636-44b3-bf8b-2b627707b95f-worker-0 device  1 [0x0f] NVIDIA A100-SXM4-40GB\n",
        "#  Rank  2 Group  0 Pid 1341852 on job-83e1033f-9636-44b3-bf8b-2b627707b95f-worker-0 device  2 [0x47] NVIDIA A100-SXM4-40GB\n",
        "#  Rank  3 Group  0 Pid 1341852 on job-83e1033f-9636-44b3-bf8b-2b627707b95f-worker-0 device  3 [0x4e] NVIDIA A100-SXM4-40GB\n",
        "#  Rank  4 Group  0 Pid 1341852 on job-83e1033f-9636-44b3-bf8b-2b627707b95f-worker-0 device  4 [0x87] NVIDIA A100-SXM4-40GB\n",
        "#  Rank  5 Group  0 Pid 1341852 on job-83e1033f-9636-44b3-bf8b-2b627707b95f-worker-0 device  5 [0x90] NVIDIA A100-SXM4-40GB\n",
        "#  Rank  6 Group  0 Pid 1341852 on job-83e1033f-9636-44b3-bf8b-2b627707b95f-worker-0 device  6 [0xb7] NVIDIA A100-SXM4-40GB\n",
        "#  Rank  7 Group  0 Pid 1341852 on job-83e1033f-9636-44b3-bf8b-2b627707b95f-worker-0 device  7 [0xbd] NVIDIA A100-SXM4-40GB\n",
        "#\n",
        "#                                                              out-of-place                       in-place          \n",
        "#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong\n",
        "#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       \n",
        "     1048576        262144     float     sum      -1    78.77   13.31   23.29      0    81.31   12.90   22.57      0\n",
        "     2097152        524288     float     sum      -1    108.8   19.28   33.73      0    108.2   19.38   33.91      0\n",
        "     4194304       1048576     float     sum      -1    131.1   31.98   55.97      0    131.0   32.01   56.02      0\n",
        "     8388608       2097152     float     sum      -1    183.4   45.74   80.04      0    180.7   46.44   81.26      0\n",
        "    16777216       4194304     float     sum      -1    293.6   57.14   99.99      0    289.4   57.98  101.46      0\n",
        "    33554432       8388608     float     sum      -1    415.7   80.73  141.27      0    414.5   80.95  141.65      0\n",
        "    67108864      16777216     float     sum      -1    657.2  102.11  178.69      0    657.7  102.04  178.57      0\n",
        "   134217728      33554432     float     sum      -1   1256.7  106.80  186.91      0   1262.3  106.33  186.07      0\n",
        "   268435456      67108864     float     sum      -1   2338.9  114.77  200.85      0   2334.0  115.01  201.27      0\n",
        "   536870912     134217728     float     sum      -1   4651.2  115.43  202.00      0   4517.0  118.85  208.00      0\n",
        "  1073741824     268435456     float     sum      -1   8139.9  131.91  230.84      0   8201.6  130.92  229.11      0\n",
        "# Out of bounds values : 0 OK\n",
        "# Avg bus bandwidth    : 130.613\n",
        "#\n",
    ]
    with open(os.path.join(path, base_profiler.args.hostfile), 'w') as f:
        if num_nodes == 1:
            f.write('node1\n')
        else:
            f.write('node1\nnode2\n')
    
    real_open = open
    with patch('os.system') as mock_system, \
         patch('builtins.open', mock_open()) as mock_file:
        def selective_mock(filename, mode='r', *args, **kwargs):
            if filename == 'nccl_log/1/rank.0/stdout' and mode == 'r':
                mock = mock_open().return_value
                mock.readlines.return_value = output
                return mock
            else:
                return real_open(filename, mode, *args, **kwargs)
            
        mock_file.side_effect = selective_mock
            
        base_profiler.profile_bandwidth(backend="nccl")
        
        # Verify number of calls
        assert mock_system.call_count == expected_calls
    
    real_open = open
    with patch('os.system') as mock_system, \
         patch('builtins.open', mock_open()) as mock_file:
        def selective_mock(filename, mode='r', *args, **kwargs):
            if filename == 'nccl_log/1/rank.0/stdout' and mode == 'r':
                mock = mock_open().return_value
                mock.readlines.return_value = output
                return mock
            else:
                return real_open(filename, mode, *args, **kwargs)
            
        mock_file.side_effect = selective_mock
            
        base_profiler.profile_sp_bandwidth(backend="nccl")
        
        # Verify number of calls
        assert mock_system.call_count == expected_sp_calls

@pytest.mark.profiler
@pytest.mark.parametrize("num_nodes,num_gpus_per_node,expected_ar_calls,expected_p2p_calls,expected_ar_sp_calls,expected_a2a_sp_calls", [
    (1, 4, 3, 2, 22, 22),
    (1, 8, 5, 3, 33, 33), 
    (2, 8, 7, 3, 33, 33),
])
def test_torch_hardware_profile(base_profiler, num_nodes, num_gpus_per_node, expected_ar_calls, expected_p2p_calls, expected_ar_sp_calls, expected_a2a_sp_calls):
    """Test Torch hardware profile"""
    base_profiler.args.num_nodes = num_nodes
    base_profiler.args.num_gpus_per_node = num_gpus_per_node

    
    path = base_profiler.path
    base_profiler.profile_bandwidth(backend="torch")

    count = 0
    with open(os.path.join(path, "scripts/profile_allreduce.sh"), 'r') as f:
        lines = f.readlines()

        for line in lines:
            if line.startswith("python -m torch.distributed.launch"):
                assert line.count("profile_allreduce.py") == 1
                count += 1
        assert count == expected_ar_calls
    
    count = 0
    with open(os.path.join(path, "scripts/profile_p2p.sh"), 'r') as f:
        lines = f.readlines()

        for line in lines:
            if line.startswith("python -m torch.distributed.launch"):
                assert line.count("profile_p2p.py") == 1
                count += 1
        assert count == expected_p2p_calls

    base_profiler.profile_sp_bandwidth(backend="torch")
    
    count = 0
    with open(os.path.join(path, "scripts/profile_allreduce_sp.sh"), 'r') as f:
        lines = f.readlines()

        for line in lines:
            if line.startswith("python -m torch.distributed.launch"):
                assert line.count("profile_allreduce.py") == 1
                count += 1
        assert count == expected_ar_sp_calls

    count = 0
    with open(os.path.join(path, "scripts/profile_all2all_sp.sh"), 'r') as f:
        lines = f.readlines()

        for line in lines:
            if line.startswith("python -m torch.distributed.launch"):
                assert line.count("profile_all2all.py") == 1
                count += 1
        assert count == expected_a2a_sp_calls
