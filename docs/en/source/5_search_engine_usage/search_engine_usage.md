# Search Engine Usage

## Integration with Galvatron Runtime

The Search Engine can be used in conjunction with the Galvatron runtime as described in the [Quick Start](../3_quick_start/quick_start.html#profiling-with-galvatron).

## Standalone Usage

Beyond its integration with the Galvatron runtime, the Galvatron Search Engine can also be used independently, offering more flexible modeling and search capabilities.

Specifically, to use the Search Engine independently, you need to modify configurations related to both the environment and the model.

### Environment Configuration

Environment configurations are located in the `profile_hardware/hardware_configs` directory and include files such as `allreduce_bandwidth_{num_nodes}nodes_{num_gpus}gpus_per_node.json`, `p2p_bandwidth_{num_nodes}nodes_{num_gpus}gpus_per_node.json`, and `overlap_coefficient.json`. The first two files represent the measured total bandwidth for allreduce or p2p operations at different scales (with `num_nodes` nodes and `num_gpus` GPUs per node).

The format of these files is as follows:

`allreduce_bandwidth_{num_nodes}nodes_{num_gpus}gpus_per_node.json`:

```
{
    "allreduce_size_{group_size}_consec_[0/1]": {bandwidth}
    ...
}
```
Here, `group_size` denotes the size of the communication group, `0/1` indicates whether the group is contiguous, and `bandwidth` represents the measured bus bandwidth.

`p2p_bandwidth_{num_nodes}nodes_{num_gpus}gpus_per_node.json`:

```
{
    "pp_size_{stage_num}": {bandwidth}
    ...
}
```
`stage_num` signifies the size of the pp stage, and `bandwidth` indicates the bus bandwidth for p2p communication at this stage size.

`overlap_coefficient.json`:
```
{
    "overlap_coe": {coe}
}
```
When computation and communication overlap, the CUDA kernel is simultaneously preempted by both, causing a slowdown. `coe` represents the slowdown ratio of the kernel when overlap occurs, typically ranging between 1.1 and 1.3.

Additionally, if you want to perform a search with `sp_space` set to `tp+sp`, you will need a new file named `sp_time_{num_nodes}nodes_{num_gpus}gpus_per_node.json`. The format of this file is as follows:

```
{
    "allreduce_size_{group_size}_{message_size}MB_time": {time},
    "all2all_size_{group_size}_{message_size}MB_time": {time},
    ...
}
```

Here, `group_size` denotes the size of the communication group for the corresponding operation (allreduce/all2all), `message_size` is the amount of data being communicated (in MB), and `time` is the duration of this communication operation.

### Model Configuration

Model configurations are found in the `models/{model_name}/configs` directory.

It is essential to modify or create files prefixed with `computation_profiling` and `memory_profiling` within `models/{model_name}/configs`. The file names follow the format `[computation/memory]_profiling_[bf16/fp16/fp32]_hidden_{hidden_size}_head_{head_num}.json`, where `bf16/fp16/fp32` indicates the data type used during training, and `hidden_size` and `head_num` correspond to the model's configuration.

The format of these files is as follows:

`computation_profiling_[bf16/fp16/fp32]_hidden_{hidden_size}_head_{head_num}.json`:
```
{
    "layertype_{layer_type}_bsz{batch_size}_seq{sequence_length}": {time},
}
```

`layer_type` denotes the type of layer. For GPT models, it is 0 for decoder layers, while for T5 models, it can be 0 or 1, representing encoder and decoder layers, respectively. `time` is the forward computation time per layer for inputs with the specified `batch_size` and `sequence_length`.

`memory_profiling_[bf16/fp16/fp32]_hidden_{hidden_size}_head_{head_num}.json`:
```
{
    "layertype_{layer_type}[/_sp]": {
        "{sequence_length}": {
            "parameter_size": {layer_parameter},
            "tp_activation_per_bsz_dict": {
                "checkpoint": {layer_ckpt_act},
                "1": {layer_tp1_act},
                "2": {layer_tp2_act},
                ...
            }
        }
        ...
    }
    "other_memory_pp_off[/_sp]": {
        "{sequence_length}": {
            "model_states": {
                "1": {othe_pp_off_tp1_ms},
                "2": {othe_pp_off_tp2_ms},
                ...
            },
            "activation": {
                "1": {othe_pp_off_tp1_act},
                "2": {othe_pp_off_tp2_act},
                ...
            }
        }
    }
    "other_memory_pp_on_first[/_sp]": {
        "{sequence_length}": {
            "model_states": {
                "1": {othe_pp_on_first_tp1_ms},
                "2": {othe_pp_on_first_tp1_ms},
                ...
            },
            "activation": {
                "1": {othe_pp_on_first_tp1_act},
                "2": {othe_pp_on_first_tp1_act},
                ...
            }
        }
    }
    "other_memory_pp_on_last[/_sp]": {
        "{sequence_length}": {
            "model_states": {
                "1": {othe_pp_on_last_tp1_ms},
                "2": {othe_pp_on_last_tp1_ms},
                ...
            },
            "activation": {
                "1": {othe_pp_on_last_tp1_act},
                "2": {othe_pp_on_last_tp1_act},
                ...
            }
        }
    }
}
```
The meaning of `layer_type` is consistent with the computation profiling file; `/_sp` indicates whether sequence parallel was enabled during measurement; `sequence_length` is the sequence length during measurement; `layer_parameter` is the memory occupied by parameters per layer; `layer_ckpt_act` is the activation memory per layer when using checkpointing; `layer_tpx_act` is the activation memory per layer when using a tp dimension of x. For sequence parallel, `layer_tpx_act` is inversely proportional to x, so not every strategy needs manual measurement. Without sequence parallel, each strategy requires separate measurement. `othe_pp_[off/on_first/on_last]_tpx_[ms/act]` represents the model states or activation memory size for embedding layers when tp is x, in stages where pp is 1, greater than 1 (first stage), and less than 1 (last stage), excluding conventional layers.

### Usage

You can modify the contents of `models/{model_name}/scripts/search_dist.sh` to use Galvatron or third-party profiling data for modeling and search. For third-party data, refer to the previous sections to modify the relevant configuration documents.