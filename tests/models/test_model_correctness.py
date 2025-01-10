import pytest
import torch
import sys
import json
import tempfile
from typing import Dict, Any
from galvatron.utils.training_utils import set_seed, distributed_dataloader
from tests.utils.init_dist import init_dist_env
from tests.utils.runtime_args import RuntimeArgs
from tests.utils.model_utils import ModelFactory
from tests.models.configs.get_config_json import ConfigFactory
from megatron.training.global_vars import set_args
from megatron.core.tensor_parallel import random
from megatron.core.parallel_state import initialize_model_parallel
from torch.optim import Adam

def _run_test(args: Dict[str, Any]):
    """Run data parallel correctness test"""
    rank, world_size = init_dist_env()
    dp_size = args["dp_size"]
    assert dp_size == world_size, "Distributed environment is not correctly initialized"
    model_type = args["model_type"]
    backend = args["backend"]
    batch_size = args["batch_size"]
    chunks = args["chunks"]
    num_steps = args["num_steps"]
    checkpoint_dir = args["checkpoint_dir"]
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    # Initialize
    set_seed(args["seed"])
    initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    random.model_parallel_cuda_manual_seed(args["seed"])
    
    args = RuntimeArgs(model_type=model_type, rank=rank, checkpoint_dir=checkpoint_dir, backend=backend)
    config_json = ConfigFactory.get_config_json(model_type)
    args.model_size = config_json
    components = ModelFactory.get_components(model_type, backend)
    config = ModelFactory.create_config(model_type, backend, args)
    # Set custom args
    args.global_train_batch_size = batch_size
    args.chunks = chunks
    set_args(args)

    if rank == 0:
        baseline_model = components.ModelClass(config)
        baseline_optimizer = Adam(baseline_model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)
        baseline_model.save_pretrained(checkpoint_dir["baseline"])
        components.convert_checkpoints(checkpoint_dir["baseline"], checkpoint_dir["converted"])
        baseline_model = baseline_model.to(device)
    
    torch.distributed.barrier()

    model = ModelFactory.create_model(model_type, backend, config, args)
    trainloader = distributed_dataloader(
        dataset=components.DatasetClass(args, device, 256),
        global_bsz=args.global_train_batch_size,
        shuffle=True,
        args=args,
        group = model.dp_groups_whole[0].group,
        collate_fn = components.collate_fn
    )
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    for i, batch in enumerate(trainloader):
        tokens, kwargs, loss_func = batch
        input_ids = tokens
        batch = [input_ids]
        if rank == 0:
            gathered_input_ids = [torch.zeros_like(input_ids) for _ in range(world_size)]
            gathered_labels = [torch.zeros_like(kwargs["labels"]) for _ in range(world_size)]
        else:
            gathered_input_ids = None
            gathered_labels = None
        torch.distributed.gather(input_ids, gathered_input_ids, dst=0)
        torch.distributed.gather(kwargs["labels"], gathered_labels, dst=0)
        
        loss = model.forward_backward(batch, i, None, 
                                    loss_func=loss_func,
                                    **kwargs)
        loss = torch.tensor(loss, device=device, dtype=torch.float)
        optimizer.step()
        optimizer.zero_grad()

        torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)

        if rank == 0:
            full_batch = torch.cat(gathered_input_ids, dim=0)
            full_labels = torch.cat(gathered_labels, dim=0)
            shift_logits = baseline_model(input_ids=full_batch).logits
            from torch.nn import CrossEntropyLoss
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = full_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            baseline_loss = loss_fct(shift_logits, shift_labels)
            baseline_loss.backward()
            baseline_optimizer.step()
            baseline_optimizer.zero_grad()

            # print(f"loss: {loss}, baseline_loss: {baseline_loss}")

            assert torch.allclose(loss, baseline_loss, atol=1e-4), f"Loss is not correct in iteration {i}: {loss} vs {baseline_loss}"
        
        if i == num_steps - 1:
            break

@pytest.mark.distributed
@pytest.mark.model
@pytest.mark.parametrize("model_type", ["gpt", "llama", "llama2"])
@pytest.mark.parametrize("backend", ["hf"])
@pytest.mark.parametrize("dp_size", [8])
# @pytest.mark.parametrize("model_type", ["gpt", "llama", "llama2"])
# @pytest.mark.parametrize("backend", ["hf", "fa"])
# @pytest.mark.parametrize("dp_size", [8])
def test_dp_correctness(run_distributed, model_type, backend, dp_size, checkpoint_dir):
    """Test data parallel training correctness"""
    config = {
        "model_type": model_type,
        "backend": backend,
        "dp_size": dp_size,
        "batch_size": 16,
        "chunks": 2,
        "num_steps": 3,
        "seed": 42,
        "checkpoint_dir": checkpoint_dir
    }
    
    run_distributed(
        func_name="_run_test",
        world_size=dp_size,
        args=config,
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