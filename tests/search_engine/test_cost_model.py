import pytest
import numpy as np
from galvatron.core.search_engine.cost_model import MemoryCostModel
from galvatron.core.search_engine.cost_model import TimeCostModel
from galvatron.core.search_engine.cost_model import OtherTimeCostModel
from tests.utils.cost_args import MemoryModelArgs, TimeModelArgs, create_model_args_from_dict

@pytest.fixture
def memory_model_args():
    """Create memory model args"""
    return MemoryModelArgs.from_mock_config()

@pytest.fixture
def time_model_args():
    """Create time model args"""
    return TimeModelArgs.from_mock_config()

@pytest.mark.search_engine
@pytest.mark.parametrize("strategy,config_updates,expected", [
    # dp
    (
        [1, 1, 8, {'fsdp': 0}],
        {
            'global_batch_size': 32,
            'pipeline_type': 'gpipe',
            'sequence_parallel': True,
            'use_zero2_for_dp': 0,
        },
        {
            'sdp_size': 8,
            'pp_stages': 1,
            'check_activation': True
        }
    ),
    # tp + checkpoint
    (
        [1, 2, 4, {'fsdp': 0, 'cpt': 1}],
        {
            'global_batch_size': 32,
            'tp_activation_per_bsz_dict': {
                1: 85, 2: 47, 4: 28, 8: 18.5,
                'checkpoint': 10.0
            },
            'sequence_parallel': True
        },
        {
            'sdp_size': 4,
            'has_checkpoint': True,
            'check_tp_division': True
        }
    ),
    # sp + checkpoint
    (
        [1, 4, 2, {'sp': 1, 'cpt': 1}],  # PP=1, TP=4, DP=2, with SP and checkpoint
        {
            'global_batch_size': 32,
            'parameter_size': 48,
            'sequence_parallel': True,
            'tp_activation_per_bsz_dict': {
                1: 85, 2: 47, 4: 28, 8: 18.5,
                'checkpoint': 10.0
            },
            'mixed_precision': True,
            'async_grad_reduce': True
        },
        {
            'sdp_size': 8,  # TP * DP = 4 * 2
            'check_sp': True,
            'has_checkpoint': True
        }
    ),
    # pp + FSDP
    (
        [2, 1, 4, {'fsdp': 1}],
        {
            'global_batch_size': 32,
            'pipeline_type': 'pipedream_flush',
            'mixed_precision': True,
            'async_grad_reduce': True
        },
        {
            'pp_stages': 2,
            'has_fsdp': True,
            'check_pipeline': True
        }
    ),
    # hybrid + Zero2
    (
        [2, 2, 2, {'fsdp': 0}],
        {
            'global_batch_size': 32,
            'use_zero2_for_dp': 1,
            'mixed_precision': True,
            'vsp': 1,
            'disable_vtp': 0,
            'async_grad_reduce': True
        },
        {
            'pp_stages': 2,
            'has_zero2': True,
            'has_vsp': True,
            'check_hybrid': True
        }
    ),
    # vsp + fsdp + async_grad_reduce=False
    (
        [1, 4, 2, {'fsdp': 1}],
        {
            'global_batch_size': 16,
            'vsp': 1,
            'async_grad_reduce': False,
            'mixed_precision': True
        },
        {
            'has_vsp': True,
            'has_fsdp': True,
            'check_async_grad': True
        }
    )
])
def test_memory_cost_model(memory_model_args, strategy, config_updates, expected):
    """Test memory cost model with various configurations"""

    config_updates['mbsz'] = 2
    config_updates['min_tp'] = 1
    config_updates['max_tp'] = 8
    args = memory_model_args.with_updates(**config_updates)
    
    # Convert config_updates to model parameter object
    model_args, train_args, parallel_args, profile_model_args, _ = create_model_args_from_dict(args.__dict__)

    print(args, profile_model_args)
    
    model = MemoryCostModel(
        strategy=strategy, 
        global_batch_size=args.__dict__.get('global_batch_size', 8),
        mbsz=args.__dict__.get('mbsz', -1),
        min_tp=args.__dict__.get('min_tp', -1),
        max_tp=args.__dict__.get('max_tp', -1),
        stage_idx=args.__dict__.get('stage_idx', 0),
        vsp=args.__dict__.get('vsp', 0),
        embed_sdp=args.__dict__.get('embed_sdp', False),
        model_args=model_args,
        train_args=train_args,
        parallel_args=parallel_args,
        profile_model_args=profile_model_args
    )
    costs = model.get_memory_cost()
    
    # Basic structure check
    assert isinstance(costs, dict)
    assert all(k in costs for k in ['parameter', 'model_states', 'activation', 'enc_total', 'other'])
    
    # Verify sdp_size
    if 'sdp_size' in expected:
        assert model.sdp_size == expected['sdp_size']
    
    # Verify pipeline stages
    if 'pp_stages' in expected:
        print(costs)
        assert len(costs['other'][1]) == expected['pp_stages']
    
    # Verify checkpoint related calculations
    if expected.get('has_checkpoint'):
        if args.sequence_parallel:
            assert model.activation_size == args.tp_activation_per_bsz_dict['checkpoint'] * model.bsz / model.tp_size
        else:
            assert model.activation_size == args.tp_activation_per_bsz_dict['checkpoint'] * model.bsz

    # Verify FSDP related calculations
    if expected.get('has_fsdp'):
        if model.chunks == 1:
            zero3_ratio = lambda d: (1/d+0.003)
        else:
            if args.async_grad_reduce:
                zero3_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
            else:
                zero3_ratio = lambda d: (1/d+0.003) * 5/4
        assert model.model_states_size == 4 * costs['parameter'] * zero3_ratio(model.sdp_size)

    # Verify Zero2 related calculations
    if expected.get('has_zero2'):
        if model.chunks == 1:
            zero2_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8)) if args.mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
        else:
            if args.async_grad_reduce:
                zero2_ratio = (lambda d: (6/8 * (1/d + 0.003) + 2/8)) if args.mixed_precision else (lambda d: (2/4 * (1/d + 0.003) + 2/4))
            else:
                zero2_ratio = (lambda d: (7/8 * (1/d + 0.003) + 1/8) * 5/4) if args.mixed_precision else (lambda d: (3/4 * (1/d + 0.003) + 1/4))
        assert abs(model.model_states_size - costs['parameter'] * 4 * zero2_ratio(model.sdp_size)) < 1e-6
    
    # Verify VSP
    if expected.get('has_vsp'):
        if 'sp' in strategy[-1].keys() and strategy[-1]['sp'] == 1:
            assert model.parameter_size == args.parameter_size  # vsp doesn't affect parameter_size
        else:
            assert model.parameter_size == args.parameter_size / model.tp_size
    
    # Specific checkpoint checks
    if expected.get('check_activation'):
        assert model.activation_size == args.tp_activation_per_bsz_dict[model.tp_size] * model.bsz
    
    if expected.get('check_tp_division'):
        assert costs['parameter'] == args.parameter_size / model.tp_size
    
    if expected.get('check_pipeline'):
        if args.pipeline_type == 'pipedream_flush':
            assert hasattr(model, 'bsz')
            assert model.bsz != config_updates['global_batch_size'] / model.dp_size
    
    if expected.get('check_hybrid'):
        assert model.tp_size > 1 and model.pp_size > 1
        assert model.parameter_size == args.parameter_size / model.tp_size
    
    if expected.get('check_async_grad'):
        assert hasattr(model, 'model_states_size')
        if not args.async_grad_reduce:
            assert model.model_states_size > costs['parameter'] * 4 / model.tp_size

    if expected.get('check_sp'):
        assert model.sdp_size == model.tp_size * model.dp_size

@pytest.mark.search_engine
@pytest.mark.parametrize("strategy,config_updates,expected", [
    # Pure Data Parallel
    (
        [1, 1, 8, {'fsdp': 0, 'tp': 1}],
        {
            'global_batch_size': 32,
            'microbatch': False,
            'comm_coe_dict': {
                '8': 1.0, '8_1': 0.8,
                '1': 1.0, '1_1': 1.0
            },
            'allreduce_dict': {1: 1.0},
            'all2all_dict': {1: 1.0}
        },
        {
            'check_dp': True,
            'has_overlap': True,
            'pp_size': 1,
            'tp_size': 1,
            'dp_size': 8
        }
    ),
    # Tensor Parallel + Checkpoint
    (
        [1, 4, 2, {'fsdp': 0, 'tp': 1, 'cpt': 1}],
        {
            'global_batch_size': 32,
            'microbatch': False,
            'sequence_length': 1024,
            'hidden_size': 2048,
            'sp_space': 'tp'
        },
        {
            'check_tp': True,
            'has_checkpoint': True,
            'check_message_size': True
        }
    ),
    # Pipeline Parallel + FSDP
    (
        [2, 1, 4, {'fsdp': 1, 'tp': 1}],
        {
            'global_batch_size': 32,
            'microbatch': False,
            'p2p_comm_coe_dict': {2: 1.0, 4: 0.8, 8: 0.6},
            'mixed_precision': True
        },
        {
            'check_pp': True,
            'has_fsdp': True,
            'check_p2p': True
        }
    ),
    # Sequence Parallel Test
    (
        [1, 4, 2, {'sp': 1, 'tp': 1}],
        {
            'global_batch_size': 32,
            'microbatch': False,
            'sp_space': 'tp+sp',
            'sequence_length': 1024,
            'hidden_size': 2048
        },
        {
            'check_sp': True,
            'check_tp_comm': True
        }
    ),
    # Hybrid Parallel + no_comm
    (
        [2, 2, 2, {'fsdp': 0, 'tp': 0}],
        {
            'global_batch_size': 32,
            'microbatch': False,
            'no_comm': True
        },
        {
            'check_hybrid': True,
            'check_no_comm': True
        }
    )
])
def test_time_cost_model(time_model_args, strategy, config_updates, expected):
    """Test time cost model with various configurations
    
    Args:
        base_time_args: Base configuration for time cost model
        strategy: Parallel strategy configuration
        config_updates: Updates to base configuration
        expected: Expected test results and checks to perform
    """
    
    # Update base parameters
    args = time_model_args.with_updates(**config_updates)
    
    # Convert config_updates to model parameter object
    model_args, train_args, parallel_args, profile_model_args, profile_hardware_args = create_model_args_from_dict(args.__dict__)
    
    # Extract global_batch_size and no_comm parameters
    global_batch_size = args.__dict__.get('global_batch_size', 8)
    no_comm = args.__dict__.get('no_comm', False)
    
    # Create model instance
    model = TimeCostModel(
        strategy=strategy,
        global_batch_size=global_batch_size,
        no_comm=no_comm,
        model_args=model_args,
        train_args=train_args,
        parallel_args=parallel_args,
        profile_model_args=profile_model_args,
        profile_hardware_args=profile_hardware_args
    )
    result = model.gen_result()
    
    # Basic checks
    assert isinstance(result, float), "Result should be a float"
    assert result >= 0, "Result should be non-negative"
    
    # Verify parallel configuration
    assert model.pp_size == strategy[0], "Pipeline parallel size mismatch"
    assert model.tp_size == strategy[1], "Tensor parallel size mismatch"
    assert model.dp_size == strategy[2], "Data parallel size mismatch"
    
    # Data parallel related checks
    if expected.get('check_dp'):
        # Verify dp message size calculation
        dp_message_size = (2*(model.dp_size-1)/model.dp_size*model.parameter_size) * model.layer_num
        if args.mixed_precision:
            dp_message_size /= 2
        assert model.dp_message_size == dp_message_size, "DP message size mismatch"
        
        if expected.get('has_overlap'):
            # Check overlap computation
            overlap_part, rest_part, _ = model.bct_dp_overlap(model.dp_message_size, model.bct)
            assert overlap_part > 0, "Should have positive overlap"
    
    # Tensor parallel related checks
    if expected.get('check_tp'):
        if args.sp_space == 'tp':
            # Verify tp message size calculation
            tp_comm_times = 4
            expected_tp_message_size = 2*(model.tp_size-1)/model.tp_size * \
                (model.bsz*model.seq_len*model.hidden_size*tp_comm_times*4/1024/1024) * model.layer_num
            if args.mixed_precision:
                expected_tp_message_size /= 2
            if not model.checkpoint:
                assert abs(model.tp_message_size - expected_tp_message_size) < 1e-6, \
                    "TP message size mismatch"
    
    # Pipeline parallel related checks
    if expected.get('check_pp'):
        if model.p2p_comm_coe is not None:
            # Verify p2p message size calculation
            expected_p2p_size = model.pp_size*2*model.bsz*model.seq_len*model.hidden_size*4/1024/1024
            if args.mixed_precision:
                expected_p2p_size /= 2
            assert model.p2p_message_size == expected_p2p_size, "P2P message size mismatch"
    
    # Sequence parallel related checks
    if expected.get('check_sp'):
        assert model.sdp_size == model.tp_size * model.dp_size, "SDP size mismatch"
        assert model.parameter_size == args.parameter_size, "Parameter size should not be divided in SP"
        
        if expected.get('check_tp_comm'):
            # Verify tp communication in SP
            per_tp_message_size = model.bsz*model.seq_len*model.hidden_size * (2 if args.mixed_precision else 4)
            assert model.per_tp_message_size == per_tp_message_size, "TP message size mismatch in SP"
            assert model.tp_comm_num == 4 * model.layer_num, "TP communication count mismatch"
    
    # Checkpoint related checks
    if expected.get('has_checkpoint'):
        assert model.checkpoint, "Checkpoint should be enabled"
        assert model.bct > model.fct, "Backward time should increase with checkpoint"
        if args.sp_space == 'tp+sp':
            assert model.tp_comm_num == 6 * model.layer_num, "TP comm should increase by 1.5x"
        else:
            assert model.tp_message_size == 1.5 * expected_tp_message_size, \
                "TP message size should increase by 1.5x"
    
    # FSDP related checks
    if expected.get('has_fsdp'):
        assert model.fsdp, "FSDP should be enabled"
        assert hasattr(model, 'fsdp_allgather_message_size'), "Should have allgather message size"
        assert model.fsdp_allgather_message_size == model.dp_message_size * 0.5, \
            "FSDP allgather message size mismatch"
    
    # Hybrid parallel checks
    if expected.get('check_hybrid'):
        assert model.pp_size > 1 and model.tp_size > 1 and model.dp_size > 1, \
            "Should be hybrid parallel"
    
    # No communication checks
    if expected.get('check_no_comm'):
        assert model.dp_message_size == 0, "Should have no communication"

@pytest.fixture
def base_other_time_args():
    """Create base arguments for OtherTimeCostModel"""
    return {
        'mbsz': 4,
        'pp_deg': 1,
        'world_size': 8,
        'sequence_length': [1024],
        'hidden_size': 1024,
        'mixed_precision': False,
        'comm_coe_dict': {
            '1': 1.0, '1_1': 1.0,
            '2': 0.8, '2_1': 0.8, '2_0': 0.9,
            '4': 0.6, '4_1': 0.6, '4_0': 0.7,
            '8': 0.5, '8_1': 0.5, '8_0': 0.6
        },
        'allreduce_dict': {
            2:{
                1024: 0.1,
                2048: 0.2,
                4096: 0.4,
                'popt': [0.0001, 0.1]  # Linear function parameters
            },
            4:{
                1024: 0.1,
                2048: 0.2,
                4096: 0.4,
                'popt': [0.0001, 0.1]  # Linear function parameters
            },
            8:{
                1024: 0.1,
                2048: 0.2,
                4096: 0.4,
                'popt': [0.0001, 0.1]  # Linear function parameters
            }
        },
        'sp_space': 'tp',
        'vsp': 0,
        'min_tp': 1,
        'max_tp': 8,
        'other_memory_pp_on': {
            'first_stage': {
                'model_states': {1: 640, 2: 320, 4: 160, 8: 80}
            },
            'last_stage': {
                'model_states': {1: 640, 2: 320, 4: 160, 8: 80}
            }
        },
        'other_memory_pp_off': {
            'model_states': {1: 640, 2: 320, 4: 160, 8: 80}
        },
        'other_time_profiled_list': 35.0
    }

@pytest.mark.search_engine
@pytest.mark.parametrize("config_updates,expected", [
    # Test case 1: Basic configuration (PP=1)
    (
        {
            'pp_deg': 1,
            'world_size': 8,
            'min_tp': 1,
            'max_tp': 4
        },
        {
            'tp_sizes': [1, 2, 4],
            'has_pp': False
        }
    ),
    # Test case 2: Pipeline parallel
    (
        {
            'pp_deg': 4,
            'world_size': 8,
            'min_tp': 1,
            'max_tp': 4
        },
        {
            'tp_sizes': [1, 2],
            'has_pp': True
        }
    ),
    # Test case 3: With VSP
    (
        {
            'pp_deg': 1,
            'world_size': 8,
            'vsp': 1,
            'min_tp': 1,
            'max_tp': 4
        },
        {
            'tp_sizes': [1, 2, 4],
            'check_vsp': True
        }
    ),
    # Test case 4: Mixed precision
    (
        {
            'pp_deg': 1,
            'world_size': 8,
            'mixed_precision': True,
            'min_tp': 1,
            'max_tp': 4
        },
        {
            'tp_sizes': [1, 2, 4],
            'check_precision': True
        }
    ),
    # Test case 5: SP+TP space
    (
        {
            'pp_deg': 1,
            'world_size': 8,
            'sp_space': 'tp+sp',
            'min_tp': 1,
            'max_tp': 4
        },
        {
            'tp_sizes': [1, 2, 4],
            'check_sp_tp': True
        }
    )
])
def test_other_time_cost_model(base_other_time_args, config_updates, expected):
    """Test OtherTimeCostModel with various configurations
    
    Args:
        base_other_time_args: Base configuration
        config_updates: Updates to base configuration
        expected: Expected test results and checks to perform
    """
    # Update configuration
    args = {**base_other_time_args, **config_updates}
    
    # Convert config_updates to model parameter object
    model_args, train_args, parallel_args, profile_model_args, profile_hardware_args = create_model_args_from_dict(args)
    
    # Fix parameter names
    if 'sequence_length' in args:
        sequence_length_list = args['sequence_length']
    else:
        sequence_length_list = [512]
        
    if 'other_time_profiled_list' in args:
        profile_model_args.other_time_profiled = args['other_time_profiled_list']
    
    # Create model instance
    model = OtherTimeCostModel(
        mbsz=args.get('mbsz', 1),
        pp_deg=args.get('pp_deg', 2),
        world_size=args.get('world_size', 8),
        vsp=args.get('vsp', False),
        embed_sdp=args.get('embed_sdp', False),
        min_tp=args.get('min_tp', 1),
        max_tp=args.get('max_tp', 8),
        sequence_length_list=sequence_length_list,
        model_args=model_args,
        train_args=train_args,
        parallel_args=parallel_args,
        profile_model_args=profile_model_args,
        profile_hardware_args=profile_hardware_args
    )
    
    # OtherTimeCostModel.gen_result() returns two values
    other_time_cost, _ = model.gen_result()
    result = other_time_cost
    
    # Basic checks
    assert isinstance(result, dict)
    assert set(result.keys()) == set(expected['tp_sizes'])
    
    for tp in expected['tp_sizes']:
        # Check list length matches pp_deg
        assert len(result[tp]) == args['pp_deg']
        
        # All values should be non-negative
        assert all(v >= 0 for v in result[tp])
        
        # Calculate expected dp_size
        dp_size = args['world_size'] // args['pp_deg'] // tp
        
        if expected.get('has_pp'):
            # For pipeline parallel, check first and last stage
            assert len(result[tp]) == args['pp_deg']
            # Values should be equal for first and last stage when first stage memory == last stage memory
            assert abs(result[tp][0] - result[tp][-1]) < 1e-6
        else:
            # For non-pipeline parallel, check single stage
            assert len(result[tp]) == 1
        
        if expected.get('check_vsp'):
            # VSP should use model_states[1] instead of model_states[tp]
            if args['pp_deg'] == 1:
                expected_dp_size = args['other_memory_pp_off']['model_states'][1] / 4
            else:
                expected_dp_size = args['other_memory_pp_on']['first_stage']['model_states'][1] / 4
            assert model.dp_size[tp] == expected_dp_size if args['pp_deg'] == 1 else \
                   (expected_dp_size, expected_dp_size)
        
        if expected.get('check_sp_tp'):
            # Check SP+TP specific calculations
            per_tp_message_size = args['mbsz']*args['sequence_length'][0]*args['hidden_size'] * (2 if args['mixed_precision'] else 4)
            if tp > 1:
                assert hasattr(model, 'per_tp_message_size')
                assert model.per_tp_message_size[0] == per_tp_message_size
   
    if expected.get('check_precision'):
        # Message sizes should be halved for mixed precision
        assert model.tp_message_size[0] == (expected['tp_sizes'][-1]-1)/expected['tp_sizes'][-1]*(args['mbsz']*args['sequence_length'][0]*args['hidden_size']/1024/1024) * 2
        