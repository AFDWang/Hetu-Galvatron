# Search Engine Usage
## 与Galvatron runtime 一起使用

Search Engine可以像[Quick Start](../3_quick_start/quick_start.html#profiling-with-galvatron)中描述的那样与Galvatron runtime配合使用。

## 独立使用
除了与Galvatron runtime配合使用之外，Galvatron Search Engine还可以独立使用，提供更加灵活的建模与搜索方式。

具体来说，为了独立使用Search Engine，用户需要修改环境和模型两个方面的配置。

### 环境配置
环境配置为`profile_hardware/hardware_configs`中的相关文件，包括`allreduce_bandwidth_{num_nodes}nodes_{num_gpus}gpus_per_node.json`，`p2p_bandwidth_{num_nodes}nodes_{num_gpus}gpus_per_node.json`，`overlap_coeffcient.json`这三个文件，其中前两个文件代表进行不同规模（num_nodes个节点，每个节点num_gpus个GPU）allreduce操作或者p2p操作时，测量出的环境总线带宽。

三个文件的具体格式如下：

`allreduce_bandwidth_{num_nodes}nodes_{num_gpus}gpus_per_node.json`:

```

{
    "allreduce_size_{group_size}_consec_[0/1]":{bandwidth}
    ...
}
```
其中group_size为进行通信操作的通信组大小，0/1代表通信组是否连续，bandwidth代表测量出的总线带宽。

`p2p_bandwidth_{num_nodes}nodes_{num_gpus}gpus_per_node.json`:

```

{
    "pp_size_{stage_num}":{bandwidth}
    ...
}
```
其中stage_num为pp stage大小，bandwidth代表当pp stage为stage_num时，进行p2p通信操作时的总线带宽。

`overlap_coeffcient.json`:
```
{
    "overlap_coe":{coe}
}
```
当计算与通信发生 overlap 时，CUDA 内核 (Kernel) 会同时被计算和通信抢占导致降速，coe代表当通信计算重叠时导致的内核降速比例，通常这个值介于1.1-1.3之间。

### 模型配置
模型配置为`models/{model_name}/configs`中的部分文件

主要需要修改或创建`models/{model_name}/configs`中前缀为`computation_profiling`和`memory_profiling`中的文件，具体来说，文件名格式类似`[computation/memory]_profiling_[bf16/fp16/fp32]_hidden_{hidden_size}_head_{head_num}.json`，其中`bf16/fp16/fp32`代表训练时要是用的数据类型，`hidden_size`，`head_num`分别为模型对应config。

这两个文件的具体格式如下：

`computation_profiling_[bf16/fp16/fp32]_hidden_{hidden_size}_head_{head_num}.json`:
```
{
    "layertype_{layer_type}_bsz{batch_size}_seq{sequence_length}": {time},
}
```

layer_type代表layer类型，对于GPT系列模型，layer_type只能为0，代表decoder层，对于T5模型，则layer_type可以为0或1，分别代表encoder层和decoder层；
time代表采用batch size为batch_size，序列长度为sequence_length的输入数据时候，单层的**仅前向计算**时间。

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
layer_type的意义与computation_profiling文件相同；`/_sp`代表该组数据测量时是否开启sequence parallel；`sequence_length`代表测量时的序列长度；layer_parameter代表单层的参数量所占内存；`layer_ckpt_act`代表使用checkpoint策略时，单层的激活值占用是多少，`layer_tpx_act`代表使用tp维度为x的策略时，单层的激活值是多少，对于开启sequence parallel的情况，`layer_tpx_act`关于x成反比例关系，可以不需要每种策略都手动测量，而不开启sequence parallel时，则需要每组策略单独测量；`othe_pp_[off/on_first/on_last]_tpx_[ms/act]`分别代表pp为1，pp大于1的第一个stage和pp小于1的最后一个stage中，对embedding层进行tp维度为x的切分时，除常规的layer以外的其他模块（主要是embedding模块）占用的model states或激活值内存大小，这里的model states包括optimzer states，parameter和gradient。

此外，如果你想使用`sp_space`为`tp+sp`的方式进行搜索，那么你还需要一个新文件`sp_time_{num_nodes}nodes_{num_gpus}gpus_per_node.json`，该文件的格式为：

```
{
    "allreduce_size_{group_size}_{message_size}MB_time": {time},
    "all2all_size_{group_size}_{message_size}MB_time": {time},
    ...
}
```
其中group_size为进行对应通信操作（allreduce/all2all）的通信组大小，message_size为进行通信操作的通信量（单位：MB），time为进行这种通信操作的时间。

### 使用

用户可以通过修改`models/{model_name}/scripts/search_dist.sh`中的内容，即可使用Galvatron/第三方的profile数据进行建模和搜索，如果想使用第三方数据，请参考前两小节修改相关配置文档，如果想使用Galvatron profile出的配置信息，请参考[使用文档](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/dev/galvatron/models/README.md)。