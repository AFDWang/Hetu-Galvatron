import pytest
import torch
import sys
import json
import tempfile
from typing import Dict, Any
from galvatron.utils.training_utils import set_seed, distributed_dataloader
from tests.utils.init_dist import init_dist_env
from tests.utils.dummy_args import DummyArgs
from tests.utils.model_utils import ModelFactory
from tests.utils.parallel_config import ParallelConfig
from tests.models.configs.get_config_json import ConfigFactory
from megatron.training.global_vars import set_args
from megatron.core.tensor_parallel import random
from megatron.core.parallel_state import initialize_model_parallel
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler

def _run_test(args: Dict[str, Any]):
    """Run data parallel correctness test"""
    rank, world_size = init_dist_env()
    tp_list = args["tp_size"]
    dp_size = world_size // tp_list["vocab_tp"]
    model_type = args["model_type"]
    backend = args["backend"]
    batch_size = args["batch_size"]
    chunks = args["chunks"]
    num_steps = args["num_steps"]
    sequence_parallel = args["sequence_parallel"]
    checkpoint_dir = args["checkpoint_dir"]
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    # Initialize
    set_seed(args["seed"])
    initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    random.model_parallel_cuda_manual_seed(args["seed"])
    
    args = DummyArgs(model_type=model_type, rank=rank, checkpoint_dir=checkpoint_dir, backend=backend)
    config_json = ConfigFactory.get_config_json(model_type)
    args.model_size = config_json
    components = ModelFactory.get_components(model_type, backend)
    config = ModelFactory.create_config(model_type, backend, args)
    # Set custom args
    args.global_train_batch_size = batch_size
    args.chunks = chunks
    args.mixed_precision = "bf16"
    args.use_flash_attn = True
    args.default_dp_type = "zero2"
    args.sequence_parallel = sequence_parallel
    parallel_config = ParallelConfig(
        pp_deg=1,
        tp_sizes_enc=tp_list["tp"],
        tp_consecutive_flags=[1] * len(tp_list["tp"]),
        dp_types_enc=["0"] * len(tp_list["tp"]),
        use_sp=[0] * len(tp_list["tp"]),
        checkpoint=[0] * len(tp_list["tp"]),
        global_bsz=batch_size,
        chunks=chunks,
        pp_division=[4],
        pipeline_type="pipedream_flush",
        default_dp_type="zero2",
        vtp=tp_list["vocab_tp"],
        vsp=0
    )
    args.galvatron_config_path = parallel_config.to_dict()
    set_args(args)

    if rank == world_size - 1:
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
        if input_ids is not None:
            gathered_input_ids = [torch.zeros_like(input_ids) for _ in range(dp_size)]
            gathered_labels = [torch.zeros_like(kwargs["labels"]) for _ in range(dp_size)]
            torch.distributed.all_gather(gathered_input_ids, input_ids, group=model.dp_groups_whole[0].group)
            torch.distributed.all_gather(gathered_labels, kwargs["labels"], group=model.dp_groups_whole[0].group)
        loss = model.forward_backward(batch, i, None, 
                                    loss_func=loss_func,
                                    **kwargs)
        optimizer.step()
        optimizer.zero_grad()

        if loss is not None:
            loss = torch.tensor(loss, device=device, dtype=torch.float)
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG, group=model.dp_groups_whole[0].group)

        if rank == world_size - 1:
            full_batch = torch.cat(gathered_input_ids, dim=0)
            full_labels = torch.cat(gathered_labels, dim=0)
            with autocast(dtype = torch.bfloat16):
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
        else:
            baseline_loss = torch.tensor(0.0, device=device, dtype=torch.float)
            loss = torch.tensor(0.0, device=device, dtype=torch.float)

        torch.distributed.broadcast(baseline_loss, src=world_size-1)
        torch.distributed.broadcast(loss, src=world_size-1)

        assert torch.allclose(loss, baseline_loss, rtol=5e-3), f"Loss is not correct in iteration {i}: {loss} vs {baseline_loss}"

        torch.distributed.barrier()
        if i == num_steps - 1:
            break

@pytest.mark.distributed
@pytest.mark.parallel
@pytest.mark.parametrize("model_type", ["gpt256"])
@pytest.mark.parametrize("backend", ["hf"])
@pytest.mark.parametrize("world_size", [8])
@pytest.mark.parametrize("tp_size", (
    {"tp":[1,2,4,8], "vocab_tp":8},
    {"tp":[2,8,2,1], "vocab_tp":4},
    {"tp":[8,4,1,2], "vocab_tp":2}
))
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_redistributed(run_distributed, model_type, backend, world_size, tp_size, sequence_parallel, checkpoint_dir):
    """Test redistributed correctness"""
    config = {
        "model_type": model_type,
        "backend": backend,
        "tp_size": tp_size,
        "batch_size": 32,
        "chunks": 2,
        "num_steps": 3,
        "seed": 42,
        "sequence_parallel": sequence_parallel,
        "checkpoint_dir": checkpoint_dir,
    }
    
    run_distributed(
        func_name="_run_test",
        world_size=world_size,
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