import pytest
import json
import time
from unittest.mock import patch, MagicMock
from tests.utils.profiler_utils import initialize_runtime_profile_profiler

@pytest.fixture(autouse=True)
def mock_distributed():
    """Mock torch.distributed functions"""
    with patch('torch.distributed.is_initialized', return_value=True), \
         patch('torch.distributed.get_world_size', return_value=8), \
         patch('torch.distributed.get_rank', return_value=0):
        yield

@pytest.fixture
def base_profiler(profiler_model_configs_dir):
    """Create base profiler instance"""
    profiler = initialize_runtime_profile_profiler(profiler_model_configs_dir, "llama_search", "hf")
    return profiler

@pytest.mark.profiler
@pytest.mark.parametrize("stage,expected_keys", [
    ("Before Forward", ["iter_1_before_forward"]),
    ("After Forward", ["iter_1_after_forward"]),
    ("After Backward", ["iter_1_after_backward", "iter_1_after_backward_max"]),
    ("After optimzer_step", [])
])
def test_profile_memory_stages(base_profiler, stage, expected_keys):
    """Test memory profiling at different stages"""
    base_profiler.set_memory_profiler(rank=0, profile_ranks=[0])
    
    with patch('torch.cuda.reset_peak_memory_stats') as mock_reset, \
        patch('torch.cuda.max_memory_allocated', return_value=1024 * 2**20), \
        patch('torch.cuda.memory_allocated', return_value=512 * 2**20), \
        patch('torch.cuda.max_memory_reserved', return_value=2048 * 2**20), \
        patch('torch.cuda.memory_reserved', return_value=1024 * 2**20):
             
        base_profiler.profile_memory(iter=1, stage=stage)
        
        # Verify reset_peak_memory_stats is called only for Before Forward
        if stage == "Before Forward":
            mock_reset.assert_called_once_with(0)
        else:
            mock_reset.assert_not_called()
        
        # Verify memory dictionary keys
        for key in expected_keys:
            assert key in base_profiler.mem_dict

@pytest.mark.profiler
@pytest.mark.parametrize("pipeline_type,expected_keys", [
    ("gpipe", ["model_states", "model_states_and_activation", "activation", 
                "model_states_and_peak_activation", "peak_activation"]),
    ("pipedream_flush", ["model_states", "model_states_and_peak_activation", "peak_activation"])
])
def test_post_profile_memory(base_profiler, pipeline_type, expected_keys):
    """Test post memory profiling with different pipeline types"""
    base_profiler.args.pipeline_type = pipeline_type
    base_profiler.mem_dict = {
        'iter_4_before_forward': 300,
        'iter_4_after_forward': 900,
        'iter_4_after_backward': 400,
        'iter_4_after_backward_max': 1100
    }
    
    with patch('time.sleep') as mock_sleep:
        base_profiler.post_profile_memory(iter=5)
        
        # Verify all expected keys exist
        for key in expected_keys:
            assert key in base_profiler.mem_dict
        
        # Verify memory calculations
        assert base_profiler.mem_dict['model_states'] == 400
        assert base_profiler.mem_dict['model_states_and_peak_activation'] == 1100
        assert base_profiler.mem_dict['peak_activation'] == 700
        
        if pipeline_type == "gpipe":
            assert base_profiler.mem_dict['model_states_and_activation'] == 900
            assert base_profiler.mem_dict['activation'] == 600

@pytest.mark.profiler
def test_post_profile_memory_with_save(base_profiler):
    """Test post memory profiling with save"""
    base_profiler.args.save_profiled_memory = True
    base_profiler.args.pipeline_type = "gpipe"
    base_profiler.args.pp_deg = 2
    base_profiler.args.global_tp_deg = 2
    base_profiler.args.global_train_batch_size = 16
    base_profiler.args.global_checkpoint = 0
    base_profiler.args.sequence_parallel = True
    base_profiler.args.vocab_tp = 1
    base_profiler.mem_dict = {
        'iter_4_before_forward': 300,
        'iter_4_after_forward': 900,
        'iter_4_after_backward': 400,
        'iter_4_after_backward_max': 1100
    }
    with patch('time.sleep') as mock_sleep, \
         patch('builtins.exit') as mock_exit:
        base_profiler.post_profile_memory(iter=5)

    with open(base_profiler.memory_profiling_path(), "r") as f:
        data = json.load(f)
        for key,value in data.items():
            for k,v in value.items():
                if k.endswith("ms"):
                    assert v == 400
                elif k.endswith("act"):
                    assert v == 600
                elif k.endswith("peak"):
                    assert v == 700

class MockCUDAEvent:
    """Mock CUDA Event class with customizable time records"""
    _time_sequence = [100.0, 100.2]
    _current_index = 0
    
    def __init__(self):
        self.record_time = None
    
    def record(self):
        self.record_time = self._time_sequence[self._current_index]
        MockCUDAEvent._current_index = (self._current_index + 1) % len(self._time_sequence)
    
    def elapsed_time(self, end):
        return (end.record_time - self.record_time) * 1000
    

def test_profile_time_start_normal(base_profiler):
    """Test normal time profiling start"""
    with patch('torch.cuda.synchronize') as mock_sync, \
         patch('builtins.print') as mock_print, \
         patch('builtins.exit') as mock_exit:
        base_profiler.start = MockCUDAEvent()
        base_profiler.end = MockCUDAEvent()
        base_profiler.start_iter = 0
        base_profiler.end_iter = 3
        # Test iteration within range
        base_profiler.profile_time_start(iter=1)
        mock_sync.assert_called_once()
        
        # Test iteration at end
        
        base_profiler.time_list = [0.1, 0.2, 0.3]
        base_profiler.profile_time_start(iter=3)
        mock_print.assert_called_with("Average iteration time is: 0.2000 s")

def test_profile_time_start_with_save(base_profiler):
    """Test time profiling start with saving"""
    base_profiler.start = MockCUDAEvent()
    base_profiler.end = MockCUDAEvent()
    base_profiler.start_iter = 0
    base_profiler.end_iter = 3
    base_profiler.time_list = [0.1, 0.2, 0.3]
    base_profiler.args.global_train_batch_size = 16
    base_profiler.args.profile_forward = True
    
    with patch('torch.cuda.synchronize') as mock_sync, \
         patch('builtins.exit') as mock_exit:
        
        base_profiler.profile_time_start(iter=3)
        
    with open(base_profiler.time_profiling_path(), "r") as f:
        data = json.load(f)
        for key,value in data.items():
            assert abs(value - 200) < 1e-6

def test_profile_time_end_with_loss(base_profiler):
    """Test time profiling end with loss output"""
    mock_loss = MagicMock()
    mock_loss.item.return_value = 0.5
    base_profiler.rank = 3  # last rank
    base_profiler.world_size = 4
    base_profiler.args.lr = 0.001
    base_profiler.args.global_train_batch_size = 32
    base_profiler.start_iter = 0
    base_profiler.end_iter = 3
    MockCUDAEvent._current_index = 0
    base_profiler.start = MockCUDAEvent()
    base_profiler.end = MockCUDAEvent()
    
    
    with patch('torch.cuda.synchronize'), \
            patch('builtins.print') as mock_print:
        base_profiler.profile_time_start(iter=1)
        base_profiler.profile_time_end(
            iter=1,
            loss=mock_loss,
            learning_rate=0.001,
            grad_norm=1.0
        )
        
        # Verify print format
        expected_output = (
            "| Iteration:      2 | Consumed samples:           64 | "
            "Elapsed time per iteration (ms): 200.0 | "
            "Learning rate: 1.000000e-03 | Loss: 5.000000e-01 | "
            "grad norm: 1.00 |"
        )

        mock_print.assert_called_once_with(expected_output)


def test_profile_time_python(base_profiler):
    """Test Python time profiling"""
    base_profiler.start_iter = 0
    base_profiler.end_iter = 3
    base_profiler.args.profile_forward = True
    base_profiler.args.global_train_batch_size = 32
    with patch('time.time', side_effect=[100.0, 101.0]):
        # Start timing
        base_profiler.profile_time_python(iter=0)
        assert base_profiler.total_start_time == 100.0
        
        # End timing
        with patch('builtins.print') as mock_print, \
            patch('galvatron.core.profiler.save_profiled_time') as mock_save, \
            patch('builtins.exit') as mock_exit:
            
            base_profiler.profile_time_python(iter=3)
            assert base_profiler.total_end_time == 101.0
            
            # Verify average time calculation
            mock_print.assert_called_with("Average iteration time is: 0.3333 s")
            
            # Verify save
            mock_save.assert_called_once()
            args = mock_save.call_args[0]
            assert abs(args[1] - 0.3333) < 1e-3  # avg_time
