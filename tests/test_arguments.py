import pytest
from unittest.mock import patch, MagicMock
import argparse
from galvatron.core.arguments import galvatron_training_args, galvatron_profile_args, galvatron_profile_hardware_args, galvatron_search_args

@pytest.mark.utils
def test_galvatron_training_args():
    """Test galvatron_training_args function"""
    parser = argparse.ArgumentParser()
    parser = galvatron_training_args(parser, use_megatron=True)
    
    # Parse args with default values
    args = parser.parse_args([])
    
    # Test default values
    assert args.set_model_config_manually == 0
    assert args.global_train_batch_size == 32
    assert args.dropout_prob == 0.1
    assert args.epochs == 10
    assert args.pp_deg == 2
    assert args.mixed_precision == "bf16"
    assert args.pipeline_type == "gpipe"
    assert args.local_rank == -1  # when use_megatron=True
    
    # Test with use_megatron=False
    parser = argparse.ArgumentParser()
    parser = galvatron_training_args(parser, use_megatron=False)
    args = parser.parse_args([])
    assert args.local_rank == 0  # when use_megatron=False
    assert args.lr == 1e-4
    assert args.gpu_id == 0

@pytest.mark.utils
def test_galvatron_profile_args():
    """Test galvatron_profile_args function"""
    parser = argparse.ArgumentParser()
    parser = galvatron_profile_args(parser)
    
    # Parse args with default values
    args = parser.parse_args([])
    
    # Test default values
    assert args.profile_type == "memory"
    assert args.set_layernum_manually == 1
    assert args.profile_mode == "static"
    assert args.profile_batch_size_step == 1
    assert args.profile_seq_length_step == 128
    assert args.layernum_min == 1
    assert args.layernum_max == 2
    assert args.max_tp_deg == 8
    assert args.profile_dp_type == "zero3"
    assert args.mixed_precision == "bf16"
    assert not args.use_flash_attn
    assert args.shape_order == "SBH"

@pytest.mark.utils
def test_galvatron_profile_hardware_args():
    """Test galvatron_profile_hardware_args function"""
    parser = argparse.ArgumentParser()
    parser = galvatron_profile_hardware_args(parser)
    
    # Parse args with default values
    args = parser.parse_args([])
    
    # Test default values
    assert args.num_nodes == 1
    assert args.num_gpus_per_node == 8
    assert args.master_addr == "$MASTER_ADDR"
    assert args.master_port == "$MASTER_PORT"
    assert args.node_rank == "$RANK"
    assert args.max_tp_size == 8
    assert args.backend == "nccl"
    assert args.start_mb == 16
    assert args.end_mb == 512
    assert args.scale == 2
    assert args.avg_or_min_or_first == "first"

@pytest.mark.utils
def test_galvatron_search_args():
    """Test galvatron_search_args function"""
    parser = argparse.ArgumentParser()
    parser = galvatron_search_args(parser)
    
    # Parse args with default values
    args = parser.parse_args([])
    
    # Test default values
    assert args.num_nodes == 1
    assert args.num_gpus_per_node == 8
    assert args.memory_constraint == 24
    assert args.min_bsz == 8
    assert args.max_bsz == 10240
    assert args.bsz_scale == 8
    assert args.search_space == "full"
    assert args.sp_space == "tp"
    assert args.max_tp_deg == 8
    assert args.max_pp_deg == 8
    assert args.default_dp_type == "ddp"
    assert args.mixed_precision == "bf16"
    assert args.pipeline_type == "gpipe"
    assert args.use_pipeline_costmodel == 1
    assert args.costmodel_coe == 1.0
    assert args.fine_grained_mode == 1

@pytest.mark.utils
@pytest.mark.parametrize("args_func", [
    galvatron_training_args,
    galvatron_profile_args,
    galvatron_profile_hardware_args,
    galvatron_search_args
])
def test_argument_groups(args_func):
    """Test if argument groups are correctly created"""
    parser = argparse.ArgumentParser()
    parser = args_func(parser)
    
    # Check if argument group exists
    group_titles = [group.title for group in parser._action_groups]
    
    expected_titles = {
        galvatron_training_args: "Galvatron Training Arguments",
        galvatron_profile_args: "Galvatron Profiling Arguments",
        galvatron_profile_hardware_args: "Galvatron Profiling Hardware Arguments",
        galvatron_search_args: "Galvatron Searching Arguments"
    }
    
    assert expected_titles[args_func] in group_titles