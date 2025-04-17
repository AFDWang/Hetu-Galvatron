# tests/conftest.py
import pytest
import torch
import torch.distributed as dist
import os, sys, json, subprocess
from typing import Dict, Callable, Tuple    
import tempfile
from pathlib import Path

@pytest.fixture
def small_model_config():
    """Provide a small model config for testing"""
    return {
        "hidden_size": 128,
        "num_layers": 2,
        "num_attention_heads": 4,
        "seq_length": 32,
        "vocab_size": 1000,
    }

@pytest.fixture
def device():
    """Provide device for testing"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def seed():
    """Return a fixed seed for reproducibility"""
    return 42

@pytest.fixture
def run_distributed():
    """Fixture that provides the distributed test runner"""
    def _run_distributed(func_name: str, world_size: int, args: Dict, script: str):
        if torch.cuda.device_count() < world_size:
            pytest.skip(f"Need at least {world_size} GPUs, but got {torch.cuda.device_count()}")
        
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "34567"
        os.environ["WORLD_SIZE"] = str(world_size)
        
        processes = []
        for rank in range(world_size):
            os.environ["RANK"] = str(rank)
            os.environ["LOCAL_RANK"] = str(rank)
            
            cmd = [
                sys.executable,
                script,
                func_name,
                json.dumps(args)
            ]
            
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            processes.append(p)

        for p in processes:
            stdout, stderr = p.communicate()
            retcode = p.poll()
            
            if retcode != 0:
                print(f"Process failed with return code {retcode}")
                print("stdout:", stdout.decode())
                print("stderr:", stderr.decode())
                pytest.fail(f"Distributed test failed with return code {retcode}")
    
    return _run_distributed

@pytest.fixture
def checkpoint_dir():
    with tempfile.TemporaryDirectory() as baseline_dir, \
         tempfile.TemporaryDirectory() as converted_dir:
        yield {
            "baseline": baseline_dir,
            "converted": converted_dir
        }

@pytest.fixture
def base_config_dirs(tmp_path: Path) -> Tuple[Path, Path, Path]:
    """Create and return config directories"""
    configs_dir = tmp_path / "configs"
    hardware_dir = tmp_path / "hardware_configs"
    output_dir = tmp_path / "output"
    return configs_dir, hardware_dir, output_dir

@pytest.fixture
def profiler_model_configs_dir(tmp_path: Path) -> Path:
    """Create and return profiler config directories"""
    configs_dir = tmp_path / "configs"
    os.makedirs(configs_dir, exist_ok=True)
    return configs_dir

@pytest.fixture
def profiler_hardware_configs_dir(tmp_path: Path) -> Path:
    """Create and return profiler config directories"""
    hardware_configs_dir = tmp_path / "hardware_configs"
    scripts_dir = tmp_path / "scripts"
    os.makedirs(hardware_configs_dir, exist_ok=True)
    os.makedirs(scripts_dir, exist_ok=True)
    return tmp_path

@pytest.fixture
def base_log_dirs(tmp_path: Path) -> str:
    """Create and return log directories"""
    log_dir = tmp_path / "logs"
    os.makedirs(log_dir, exist_ok=True)
    return str(log_dir)

