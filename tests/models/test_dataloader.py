import pytest
import torch
import torch.distributed as dist
import sys, json
from galvatron.utils.training_utils import set_seed, distributed_dataloader

from galvatron.models.gpt_fa.dataloader import (
    DataLoaderForGPT as DataLoaderForGPT_FA,
    random_collate_fn as random_collate_fn_gpt_fa
)
from galvatron.models.llama_fa.dataloader import (
    DataLoaderForLlama as DataLoaderForLlama_FA,
    random_collate_fn as random_collate_fn_llama_fa
)
from galvatron.models.gpt_hf.dataloader import (
    DataLoaderForGPT as DataLoaderForGPT_HF,
    random_collate_fn as random_collate_fn_gpt_hf
)
from galvatron.models.llama_hf.dataloader import (
    DataLoaderForLlama as DataLoaderForLlama_HF,
    random_collate_fn as random_collate_fn_llama_hf
)
from tests.utils.init_dist import init_dist_env
from megatron.training.global_vars import set_args

def _run_test(args):
    """Test dataloader with different group sizes"""
    rank, world_size = init_dist_env()
    group_size = args["group_size"]
    seed = args["seed"]
    small_model_config = args["small_model_config"]
    model_type = args["model_type"]
    use_flash_attn = args["use_flash_attn"]
    dataloader_backend = args["dataloader_backend"]

    if world_size < group_size:
        pytest.skip(f"Test requires at least {group_size} processes")

    num_groups = world_size // group_size
    group_id = rank // group_size
    groups = []
    for i in range(num_groups):
        ranks_in_group = list(range(i * group_size, (i + 1) * group_size))
        group = dist.new_group(ranks=ranks_in_group)
        groups.append(group)
    
    current_group = groups[group_id]
    
    dataloaders = {
        ("gpt", "fa"): (DataLoaderForGPT_FA, random_collate_fn_gpt_fa),
        ("llama", "fa"): (DataLoaderForLlama_FA, random_collate_fn_llama_fa),
        ("gpt", "hf"): (DataLoaderForGPT_HF, random_collate_fn_gpt_hf),
        ("llama", "hf"): (DataLoaderForLlama_HF, random_collate_fn_llama_hf),
    }

    set_seed(seed)

    DatasetClass, collate_fn = dataloaders[(model_type, dataloader_backend)]
    class DummyArgs:
        def __init__(self, config):
            self.vocab_size = config["vocab_size"]
            self.seq_length = config["seq_length"]
            self.use_flash_attn = use_flash_attn

    args = DummyArgs(small_model_config)
    set_args(args)
    dataset = DatasetClass(
        args=args,
        device=f'cuda:{rank}',
        dataset_size = 64
    )

    global_bsz = 16
    
    # Set seed before creating dataloader
    
    
    loader = distributed_dataloader(
        dataset=dataset,
        global_bsz=global_bsz,
        shuffle=True,
        group=current_group,
        collate_fn=collate_fn
    )

    assert loader is not None
    assert isinstance(loader.sampler, torch.utils.data.distributed.DistributedSampler)

    expected_local_bsz = global_bsz // group_size
    assert loader.batch_size == expected_local_bsz

    # Get first batch
    first_batch = None
    for batch in loader:
        first_batch = batch
        break

    assert(first_batch[0].shape == (expected_local_bsz, small_model_config["seq_length"]))
    assert(isinstance(first_batch[1], dict) )
    if use_flash_attn:
        assert(first_batch[1]["attention_mask"] is None)
    else:
        assert(first_batch[1]["attention_mask"] is not None)
    assert(first_batch[1]["labels"].shape == (expected_local_bsz, small_model_config["seq_length"]))
    assert(first_batch[2] is None)
    rank_in_group = rank % group_size
    all_position_groups = []
    for pos in range(group_size):
        ranks_with_same_position = [i * group_size + pos for i in range(num_groups)]
        all_position_groups.append(ranks_with_same_position)

    groups = []
    for ranks_in_group in all_position_groups:
        group = dist.new_group(ranks=ranks_in_group)
        groups.append(group)

    my_group = groups[rank_in_group]
    
    assert rank in all_position_groups[rank_in_group]

    same_rank_samples = [torch.zeros_like(first_batch[0]) for _ in range(num_groups)]
    dist.all_gather(same_rank_samples, first_batch[0], group=my_group)
    assert all(torch.equal(same_rank_samples[0], sample) for sample in same_rank_samples), \
        f"Same rank in different DP groups should get same data, but got different samples"
        
    
@pytest.mark.distributed
@pytest.mark.parametrize("model_type", ["gpt", "llama"])
@pytest.mark.parametrize("use_flash_attn", [True, False])
@pytest.mark.parametrize("dataloader_backend", ["fa", "hf"])
@pytest.mark.parametrize("group_size", [2])
def test_distributed_dataloader_with_groups(run_distributed, small_model_config, seed, group_size, model_type, use_flash_attn, dataloader_backend):
    args = {
        "group_size": group_size,
        "seed": seed,
        "small_model_config": small_model_config,
        "model_type": model_type,
        "use_flash_attn": use_flash_attn,
        "dataloader_backend": dataloader_backend
    }

    run_distributed(
        func_name="_run_test",
        world_size=8,
        args=args,
        script=__file__
    )
if __name__ == "__main__":
    """Entry point for distributed processes"""
    if len(sys.argv) != 3:
        print("Usage: python test_file.py <function_name> <json_args>")
        sys.exit(1)
        
    func_name = sys.argv[1]
    args = json.loads(sys.argv[2])
    
    if func_name == "_run_test":
        _run_test(args)
    else:
        print(f"Unknown function: {func_name}")
        sys.exit(1)