# Galvatron Model Usage

Galvatron provides sample code for a bunch of mainstream models to demonstrate how a Transformer model should be rewritten to accommodate Galvatron's automatic optimization API. In addition, users can quickly start from these models, optimizing parallelism strategies in their own hardware environment. Enter model directory by ```cd model_name``` to start.


## Profiling with Galvatron
The first step to use Galvatron is to profile the hardware environment and the model forward computation time.

(1) Firstly, profile the hardward environment. Please refer to the [Galvatron Document](../../README.md#profiling-with-galvatron) for details. Make sure that the hardward environment is already profiled before running any script in model directory!

(2) Secondly, profile the model computation time:
``` shell
sh scripts/profile_computation.sh
```

For models and configurations in the [Galvatron Model Zoo](.), the profiling step is already done. For user-customized models, an extra step is required to profile the model memory cost: 
``` shell
sh scripts/profile_memory.sh
```


## Parallelism Optimizing with Galvatron

Given the cluster and the memory budget, Galvatron Search Engine will generate the optimal parallelism strategy automatically. The optimized parallelism strategy will be saved in `configs` as JSON file for the training. To conduct parallelim optimization with Galvatron Search Engine, run:
``` shell
sh scripts/search_dist.sh
```

Users can customize multiple parallelism optimization options:

### Model Configuration
Users can set `model_size` and easily get a pre-defined model configuration. Users can also customize model configuration: specify `set_model_config_manually` to `1` and specify model configs manually, or specify `set_layernum_manually` to `1` and specify layer numbers manually only.

### Cluster Size & Memory Constraint
Galvatron can perform searching over multiple nodes with same number of GPUs. Users should set `num_nodes`, `num_gpus_per_node` and `memory_constraint` (memory budget for each GPU).

### Batch Size
For batch size controlling, the searching process starts from `min_bsz` and ends at `max_bsz`, with a scale of `bsz_scale`. Users can also set `settle_bsz` to find the optimal strategy when batch size is `settle_bsz`.

### Parallelism Search Space
Galvatron incorporates five parallelism dimensions in search space (`dp` for data parallel, `sdp` for sharded data parallel, `tp` for tensor parallel, `pp` for pipeline parallel, and `ckpt` for activation checkpointing). Users can use pre-defined search space (`full` for layerwise optimization over all parallelism dimensions introduced in Galvatron, `3d` for model-wise optimization over `(dp,tp,pp)`, and other options for layerwise optimization over the corresponding combination of dimensions). Users can disable any parallelism dimension by set `disable_*` to `1`. 

Please refer to ```galvatron_search_args``` in [arguments.py](../core/arguments.py) for the full list of searching arguments.

## Training with Galvatron

To train the model with Galvatron, run:
``` shell
sh scripts/train_dist.sh
```

Users can customize multiple training options:

### Model Configuration
Users can set `model_size` and easily get a pre-defined model configuration. Users can also customize model configuration: specify `set_model_config_manually` to `1` and specify model configs manually, or specify `set_layernum_manually` to `1` and specify layer numbers manually only.

### Cluster Environment
Galvatron can perform training over multiple nodes with same number of GPUs. Users should set ```NUM_NODES, NUM_GPUS_PER_NODE, MASTER_ADDR, MASTER_PORT, NODE_RANK``` according to the environment.

### Parallelism Strategy

In distributed training with Galvatron, users can either train models with the optimal parallelism strategy searched by the parallelism optimization to obtain the optimal throughput, or specify the hybrid parallelism strategies as they like.

#### JSON Config Mode [Recommended]
JSON config mode is a **recommended** layerwise hybrid parallel training mode, activated by assigning argument `galvatron_config_path` with the config path in `configs` directory. In JSON config mode, users don't need be aware of the details of searched parallelism strategies, and don't need to tune any parallelism strategies or hyper-parameters. Users can simply use the searched optimal parallelism strategy saved in `configs` directory by setting `galvatron_config_path` as `./configs/galvatron_config_xxx.json`. For advanced users, JSON config mode also provides a more fine-grained approach to parallelism tuning.

#### GLOBAL Config Mode
GLOBAL config mode is a global hybrid parallel training mode, activated by assigning argument `galvatron_config_path` as `None`. In this mode, users can specify `pp_deg`, `global_tp_deg`, `global_tp_consec`, `sdp`, `global_train_batch_size`, `chunks`, `global_checkpoint`, `pipeline_type` to determine the global parallelism strategy, and all the layers of the Transformer model uses the same hybrid parallelism strategy assigned by the users (just as in Megatron-LM).

### Arguments
1. JSON Config Mode
- `galvatron_config_path`: str, json config path, whether to activate JSON config mode. If activated, arguments in GLOBAL config mode will be ignored and overwritten by the JSON config.
2. GLOBAL Config Mode
- `global_train_batch_size`: Integer, global batch size of distributed training.
- `pp_deg`: Integer, pipeline (PP) degree,.
- `global_tp_deg`: Integer, tensor parallel (TP) degree.
- `global_tp_consec`: `0`/`1`, whether the communication group of TP is consecutive, (eg., [0,1,2,3] is consecutive while [0,2,4,6] is not).
- `sdp`: `0`/`1`, whether to use SDP instead of DP.
- `chunks`: Integer, number of microbatches of PP.
- `global_checkpoint`: `0`/`1`, whether to turn on activation checkpointing to the whole model.
- `pipeline_type`: `gpipe` or `pipedream_flush`, choose the pipeline type to use.

### Other Training Optimizations
Set `mixed_precision` to allow mixed precision training, e.g., `bf16`. Set `use-flash-attn` to allow [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) features.

Please refer to function ```galvatron_training_args``` in [arguments.py](../core/arguments.py) for the full list of training arguments.