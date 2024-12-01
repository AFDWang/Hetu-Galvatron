# Galvatron Model Usage

Galvatron provides sample code for a bunch of mainstream models to demonstrate how a Transformer model should be rewritten to accommodate Galvatron's automatic optimization API. In addition, you can quickly start from these models, optimizing parallelism strategies in their own hardware environment. Enter model directory by ```cd model_name``` to start.


## Profiling with Galvatron
The first step to use Galvatron is to profile the hardware environment and the model forward computation time.

(1) Firstly, profile the hardward environment. Please refer to the [Quick Start](../3_quick_start/quick_start.html#profiling-with-galvatron) for details. Make sure that the hardward environment is already profiled before running any script in model directory!

(2) Secondly, profile the model computation time:
``` shell
sh scripts/profile_computation.sh
```

For models and configurations in the [Galvatron Model Zoo](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models), the profiling step is already done. For user-customized models, an extra step is required to profile the model memory cost: 
``` shell
sh scripts/profile_memory.sh
```

### Other Profile Arguments

By setting `profile_min_batch_size`, `profile_max_batch_size`, and `profile_batch_size_step`, you can control the batch sizes used during time profiling. Specifically, the time profiling will be performed using batch sizes in `range(profile_min_batch_size, profile_max_batch_size + 1, profile_batch_size_step)`. Similarly, by setting `profile_min_seq_length`, `profile_max_seq_length`, `profile_seq_length_step`, you can control the sequence lengths used during time and memory profiling. The former should be used with `profile_mode == 'batch'`, and the latter with `profile_mode == 'sequence'`. Further details about `profile_mode` will be discussed later. 

## Parallelism Optimizing with Galvatron

Given the cluster and the memory budget, Galvatron Search Engine will generate the optimal parallelism strategy automatically. The optimized parallelism strategy will be saved in `configs` as JSON file for the training. To conduct parallelim optimization with Galvatron Search Engine, run:
``` shell
sh scripts/search_dist.sh
```

You can customize multiple parallelism optimization options:

### Model Configuration
You can set `model_size` and easily get a pre-defined model configuration. You can also customize model configuration: specify `set_model_config_manually` to `1` and specify model configs manually, or specify `set_layernum_manually` to `1` and specify layer numbers manually only.

### Cluster Size & Memory Constraint
Galvatron can perform searching over multiple nodes with same number of GPUs. You should set `num_nodes`, `num_gpus_per_node` and `memory_constraint` (memory budget for each GPU).

### Batch Size & Chunk
For batch size controlling, the searching process starts from `min_bsz` and ends at `max_bsz`, with a scale of `bsz_scale`. You can also set `settle_bsz` to find the optimal strategy when batch size is `settle_bsz`. Additionally, you can configure `settle_chunk` to determine the optimal strategy for a chunk size of `settle_chunk`.

### Parallelism Search Space
Galvatron incorporates five parallelism dimensions in search space (`dp` for data parallel, `sdp` for sharded data parallel, `tp&vtp` for tensor parallel, `pp` for pipeline parallel, and `ckpt` for activation checkpointing). You can use pre-defined search space (`full` for layerwise optimization over all parallelism dimensions introduced in Galvatron, `3d` for model-wise optimization over `(dp,tp,pp)`, and other options for layerwise optimization over the corresponding combination of dimensions). You can disable any parallelism dimension by set `disable_*` to `1`. 

Please refer to ```galvatron_search_args``` in [arguments.py](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/core/arguments.py) for the full list of searching arguments.

### Other Searching Arguments

Set `sequence-parallel` to account for the `Megatron-TP-SP` method when building the cost model.

Set `fine_grained_mode` to `0` / `1`(default:`1`) to disable/enable fine-grained parallel strategy and search. For the former, the search engine will find a global parallel strategy, meaning the same parallel strategy is applied to all layers. For the latter, it refers to the standard fine-grained parallel strategy search.

Set `profile_mode` to `static` / `batch` / `sequence` (default:`static`) to determine the estimation method for computation time and memory when building a cost model, `static` indicates that computation time increases proportionally with batch size. In contrast, `batch` suggests that computation time grows linearly with batch size. Specifically, we will use an $\alpha-\beta$ model to fit a linear function based on the profiled data. To ensure accuracy, when using `batch`, we require profile results for 8 different batch sizes for the same layer type. Additionally, `sequence` uses profiled data to model memory and time performance for other sequence lengths. In practice, `profile_mode` in the searching argument should typically match the profile argument. When using `static` or `batch` modes, user also need to ensure the sequence length is consistent. However, this is not necessary when using the `sequence` mode.

Set `no_global_memory_buffer` to disable the estimation of global memory for all-gather buffer when using Megatron-SP. In Megatron-SP, a buffer is allocated to store the results of all-gather communication operations. This memory is not released, and as the sequence length increases, the memory usage of this buffer can become significant.

## Training with Galvatron

To train the model with Galvatron, run:
``` shell
sh scripts/train_dist.sh
```

You can customize multiple training options:

### Checkpoint loading
Galvatron supports loading Huggingface models and adapts to fine-grained parallelism strategies. With a simple weight conversion process, this can be achieved by executing the following command:
```shell
cd tools
bash convert_{MODEL_TYPE}.sh
```
You need to modify the script by setting INPUT_PATH and OUTPUT_PATH to the directories where the checkpoint files are stored before and after conversion, respectively.
Please note that the weight conversion is independent of the parallelism strategy.

Next, you can use the following arguments in their training script to load the checkpoint:
```shell
--initialize_on_meta 1 \
--load ${OUTPUT_PATH}
```

### Training with datasets
Galvatron supports the use of the Megatron dataset, with preprocessing and usage methods compatible with [Megatron](https://github.com/NVIDIA/Megatron-LM).


### Model Configuration
you can set `model_size` and easily get a pre-defined model configuration. You can also customize model configuration: specify `set_model_config_manually` to `1` and specify model configs manually, specify `set_layernum_manually` to `1` and specify layer numbers manually, specify `set_seqlen_manually` to `1` and specify sequence length manually.

### Cluster Environment
Galvatron can perform training over multiple nodes with same number of GPUs. You should set ```NUM_NODES, NUM_GPUS_PER_NODE, MASTER_ADDR, MASTER_PORT, NODE_RANK``` according to the environment.

### Parallelism Strategy

In distributed training with Galvatron, you can either train models with the optimal parallelism strategy searched by the parallelism optimization to obtain the optimal throughput, or specify the hybrid parallelism strategies as they like.

#### JSON Config Mode [Recommended]
JSON config mode is a **recommended** layerwise hybrid parallel training mode, activated by assigning argument `galvatron_config_path` with the config path in `configs` directory. In JSON config mode, you don't need be aware of the details of searched parallelism strategies, and don't need to tune any parallelism strategies or hyper-parameters. You can simply use the searched optimal parallelism strategy saved in `configs` directory by setting `galvatron_config_path` as `./configs/galvatron_config_xxx.json`. For advanced you, JSON config mode also provides a more fine-grained approach to parallelism tuning.

#### GLOBAL Config Mode
GLOBAL config mode is a global hybrid parallel training mode, activated by assigning argument `galvatron_config_path` as `None`. In this mode, you can specify `pp_deg`, `global_tp_deg`, `global_tp_consec`, `sdp`, `global_train_batch_size`, `chunks`, `global_checkpoint`, `pipeline_type` to determine the global parallelism strategy, and all the layers of the Transformer model uses the same hybrid parallelism strategy assigned by the you (just as in Megatron-LM).

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
- `vocab_tp`: Interger, vocab embedding parallel degree.


### Other Training Optimizations
Set `mixed_precision` to allow mixed precision training, e.g., `bf16`. Set `use-flash-attn` to allow [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) features.

Set `sequence-parallel` to enable `Megatron-TP-SP` method, which can further reduce memory usage.

Set `use_ulysses` to enable [Ulysses-SP](https://github.com/microsoft/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md) method, which will replace `Megatron-TP-SP`. Once activated, the TP (tensor parallel) dimension will automatically be converted to the SP (sequence parallel) dimension.


Set `no_async_grad_reduce` to disable the asynchronous gradient synchronization method, which is enabled by default. In Galvatron, during each iteration of training, when gradient accumulation is required, the default behavior is to perform the gradient reduce scatter operation only after all  backward passes are completed. This approach reduces communication overhead but incurs additional memory usage: each device holds a full copy of the gradients until gradient synchronization, causing Zero-2 to degrade to Zero-1.When `no_async_grad_reduce` is set, Galvatron synchronizes gradients after every backward step, maintaining low memory usage. However, this introduces additional communication, though much of it can overlap with computation. The trade-off is increased complexity in the cost model, potentially reducing the accuracy of cost model. We plan to offer a more fine-grained and accurate cost model in the future.

Please refer to function ```galvatron_training_args``` in [arguments.py](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/core/arguments.py) for the full list of training arguments.

**New features are only supported on llama_hf, gpt_hf.**
