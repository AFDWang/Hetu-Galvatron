# Galvatron 模型使用

Galvatron 为多个主流模型提供了示例代码，展示了如何重写 Transformer 模型以适应 Galvatron 的自动优化 API。此外，你可以从这些模型快速开始，在自己的硬件环境中优化并行策略。通过 ```cd model_name``` 进入模型目录开始。

## 使用 Galvatron 进行性能分析
使用 Galvatron 的第一步是对硬件环境和模型前向计算时间进行性能分析。

(1) 首先，对硬件环境进行性能分析。详细信息请参考 [快速入门](../3_quick_start/quick_start_zh.html#galvatron)。在运行模型目录中的任何脚本之前，请确保已完成硬件环境的性能分析！

(2) 其次，对模型计算时间进行性能分析：
````shell
sh scripts/profile_computation.sh
````

对于 [Galvatron Model Zoo](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models) 中的模型和配置，性能分析步骤已经完成。对于用户自定义模型，需要额外进行模型内存消耗的性能分析：
````shell
sh scripts/profile_memory.sh
````

### 其他性能分析参数

通过设置 `profile_min_batch_size`、`profile_max_batch_size` 和 `profile_batch_size_step`，你可以控制时间性能分析期间使用的批量大小。具体来说，时间性能分析将使用 `range(profile_min_batch_size, profile_max_batch_size + 1, profile_batch_size_step)` 范围内的批量大小。类似地，通过设置 `profile_min_seq_length`、`profile_max_seq_length`、`profile_seq_length_step`，你可以控制时间和内存性能分析期间使用的序列长度。前者应与 `profile_mode == 'batch'` 一起使用，后者与 `profile_mode == 'sequence'` 一起使用。而对于`static`模式，则需要通过设置`profile_batch_size`来控制批量大小，设置`profile_seq_length_list`来控制序列长度。关于 `profile_mode` 的更多细节将在后面讨论。

## 使用 Galvatron 进行并行优化

给定集群和内存预算，Galvatron 搜索引擎将自动生成最优并行策略。优化后的并行策略将以 JSON 文件形式保存在 `configs` 中用于训练。要使用 Galvatron 搜索引擎进行并行优化，运行：
````shell
sh scripts/search_dist.sh
````

你可以自定义多个并行优化选项：

### 模型配置
你可以设置 `model_size` 来轻松获取预定义的模型配置。你也可以自定义模型配置：将 `set_model_config_manually` 设为 `1` 并手动指定模型配置，或将 `set_layernum_manually` 设为 `1` 仅手动指定层数。

### 集群大小和内存约束
Galvatron 可以在具有相同 GPU 数量的多个节点上进行搜索。你需要设置 `num_nodes`、`num_gpus_per_node` 和 `memory_constraint`（每个 GPU 的内存预算）。

### 批量大小和分块
对于批量大小控制，搜索过程从 `min_bsz` 开始，以 `bsz_scale` 的比例增长，到 `max_bsz` 结束。你也可以设置 `settle_bsz` 来找到批量大小为 `settle_bsz` 时的最优策略。此外，你可以配置 `settle_chunk` 来确定分块大小为 `settle_chunk` 时的最优策略。

### 并行搜索空间
Galvatron 在搜索空间中包含五个并行维度（`dp` 用于数据并行，`sdp` 用于分片数据并行，`tp&vtp` 用于张量并行，`pp` 用于流水线并行，以及 `ckpt` 用于激活检查点）。你可以使用预定义的搜索空间（`full` 用于在 Galvatron 引入的所有并行维度上进行逐层优化，`3d` 用于在 `(dp,tp,pp)` 上进行模型级优化，以及其他用于在相应维度组合上进行逐层优化的选项）。你可以通过将 `disable_*` 设为 `1` 来禁用任何并行维度。

有关搜索参数的完整列表，请参考 [arguments.py](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/core/arguments.py) 中的 ```galvatron_search_args```。

### 其他搜索参数

设置 `sequence-parallel` 以在构建成本模型时考虑 `Megatron-TP-SP` 方法。

设置 `fine_grained_mode` 为 `0` / `1`（默认：`1`）以禁用/启用细粒度并行策略和搜索。对于前者，搜索引擎将找到一个全局并行策略，即对所有层应用相同的并行策略。对于后者，它指的是标准的细粒度并行策略搜索。

设置 `profile_mode` 为 `static` / `batch` / `sequence`（默认：`static`）以确定构建成本模型时的计算时间和内存估算方法。`static` 表示计算时间与批量大小成比例增长。相比之下，`batch` 表示计算时间与批量大小线性增长。具体来说，我们将使用 $\alpha-\beta$ 模型基于分析数据拟合线性函数。为确保准确性，使用 `batch` 时，我们需要对同一层类型的 8 个不同批量大小进行性能分析。此外，`sequence` 使用分析数据来模拟其他序列长度的内存和时间性能。在实践中，搜索参数中的 `profile_mode` 通常应与性能分析参数匹配。使用 `static` 或 `batch` 模式时，用户还需要确保序列长度一致。但使用 `sequence` 模式时则不需要。

设置 `sp_space` 为 `tp+sp` / `tp`（默认：`tp`）以确定序列并行的搜索空间。`tp+sp` 表示同时考虑 Megatron-SP 和 Ulysses，而 `tp` 表示仅考虑 Megatron-SP。

设置 `no_global_memory_buffer` 以禁用使用 Megatron-SP 时全局内存的 all-gather 缓冲区估算。在 Megatron-SP 中，会分配一个缓冲区来存储 all-gather 通信操作的结果。这个内存不会被释放，随着序列长度的增加，这个缓冲区的内存使用量可能会变得很大。

此外，为了加速搜索，我们还提供了并行搜索选项，可以通过开启`parallel_search`启用并行搜索，并使用`worker`参数设置并行搜索的线程数，默认是2xCPU核心数，此外，我们还提供了`log_dir`参数设置搜索日志保存路径。

**`sp_space` 设为 `tp+sp` 与 `tp_consec` 设为 0 不兼容。`tp_consec` 的搜索很少见，我们计划在未来版本中移除它。**

## 使用 Galvatron 进行训练

要使用 Galvatron 训练模型，运行：
````shell
sh scripts/train_dist.sh
````

你可以自定义多个训练选项：

### 检查点加载和保存

#### 检查点加载
Galvatron 支持加载 Huggingface 模型并适应细粒度并行策略。通过简单的权重转换过程，可以执行以下命令来实现：
````shell
cd tools
bash convert_{MODEL_TYPE}_h2g.sh
````

你需要修改脚本，设置 INPUT_PATH 和 OUTPUT_PATH 分别为转换前后存储检查点文件的目录。
请注意，权重转换与并行策略无关。

接下来，你可以在训练脚本中使用以下参数来加载检查点：
````shell
--initialize_on_meta 1 \
--load ${OUTPUT_PATH}
````

对于之前由 Galvatron 保存的检查点，你可以通过添加 ```--load_distributed``` 来加载。注意，这种方法要求当前的并行策略与保存检查点时使用的并行策略一致。

#### 检查点保存
Galvatron 支持在训练期间保存检查点。你可以在训练脚本中使用以下参数来保存检查点：
````shell
--save ${OUTPUT_PATH}
--save-interval ${SAVE_INTERVAL}
````

Galvatron 将在目标目录中存储指定并行策略的分布式检查点，包括参数和优化器状态。

要将已保存的分布式 Galvatron 检查点转换为 Hugging Face 格式，你可以使用以下命令：
````shell
cd tools
bash convert_{MODEL_TYPE}_g2h.sh
````

### 使用数据集训练
Galvatron 支持使用 Megatron 数据集，其预处理和使用方法与 [Megatron](https://github.com/NVIDIA/Megatron-LM) 兼容。

### 模型配置
你可以设置 `model_size` 来轻松获取预定义的模型配置。你也可以自定义模型配置：将 `set_model_config_manually` 设为 `1` 并手动指定模型配置，将 `set_layernum_manually` 设为 `1` 并手动指定层数，将 `set_seqlen_manually` 设为 `1` 并手动指定序列长度。

### 集群环境
Galvatron 可以在具有相同 GPU 数量的多个节点上进行训练。你应该根据环境设置 ```NUM_NODES, NUM_GPUS_PER_NODE, MASTER_ADDR, MASTER_PORT, NODE_RANK```。

### 并行策略

在使用 Galvatron 进行分布式训练时，你可以选择使用并行优化搜索到的最优并行策略来获得最佳吞吐量，或者按照自己的喜好指定混合并行策略。

#### JSON 配置模式 [推荐]
JSON 配置模式是一种**推荐的**逐层混合并行训练模式，通过将参数 `galvatron_config_path` 指定为 `configs` 目录中的配置路径来激活。在 JSON 配置模式下，你不需要了解搜索到的并行策略的细节，也不需要调整任何并行策略或超参数。你可以通过将 `galvatron_config_path` 设置为 `./configs/galvatron_config_xxx.json` 来简单地使用保存在 `configs` 目录中的搜索到的最优并行策略。对于高级用户，JSON 配置模式还提供了更细粒度的并行调优方法。

混合并行策略在 JSON 格式中表示如下：
````json
{
    // 流水线并行配置
    "pp_deg": <num_pipeline_stages>,
    "pp_division": "<layers_per_stage_1>,<layers_per_stage_2>,...",
    "pipeline_type": "pipedream_flush",  // or "gpipe"
    "chunks": <num_micro_batches>,

    // 张量并行配置（每层）
    "tp_sizes_enc": "<tp_size_1>,<tp_size_2>,...,<tp_size_n>",
    "tp_consecutive_flags": "<consec_1>,<consec_2>,...,<consec_n>",
    
    // 数据并行配置（每层）
    "dp_types_enc": "<dp_type_1>,<dp_type_2>,...,<dp_type_n>",
    "default_dp_type": "zero2",    // or "ddp", "zero3"
    
    // 序列并行配置（每层）
    "use_sp": "<sp_flag_1>,<sp_flag_2>,...,<sp_flag_n>",

    // 内存优化配置（每层）
    "checkpoint": "<ckpt_flag_1>,<ckpt_flag_2>,...,<ckpt_flag_n>",
    
    // 全局训练配置
    "global_bsz": <global_batch_size>,
    
    // 词汇并行配置
    "vtp": <vocab_tp_size>,
    "vsp": <vocab_sp_flag>,
    "embed_sdp": <embed_sdp_flag>
}
````

JSON 配置字段按类别组织：

### 流水线并行配置
- `pp_deg`：模型分段的流水线阶段数
- `pp_division`：每个流水线阶段中的层数，以逗号分隔
- `pipeline_type`：调度策略（"pipedream_flush" 或 "gpipe"）
- `chunks`：流水线并行的微批次数

### 张量并行配置
- `tp_sizes_enc`：每层的张量并行度
- `tp_consecutive_flags`：GPU 分配方法（1=连续，0=非连续）

### 数据并行配置
- `dp_types_enc`：每层的数据并行类型（0=default_dp_type，1=zero3）
- `default_dp_type`：默认数据并行策略（"ddp"、"zero2" 或 "zero3"）

### 序列并行配置
- `use_sp`：每层的 Ulysses 序列并行标志（0=禁用，1=启用）

### 内存优化
- `checkpoint`：每层的激活检查点标志（0=禁用，1=启用）

### 全局配置
- `global_bsz`：所有设备的总训练批量大小

### 词表并行
- `vtp`：词表的张量并行度
- `vsp`：词表的序列并行标志（0=禁用，1=启用）
- `embed_sdp`：词表的数据并行策略（0=使用默认并行策略，1=使用zero3）

#### 全局配置模式
全局配置模式是一种全局混合并行训练模式，通过将参数 `galvatron_config_path` 设为 `None` 来激活。在此模式下，你可以指定 `pp_deg`、`global_tp_deg`、`global_tp_consec`、`sdp`、`global_train_batch_size`、`chunks`、`global_checkpoint`、`pipeline_type` 来确定全局并行策略，Transformer 模型的所有层都使用你指定的相同混合并行策略（就像在 Megatron-LM 中一样）。

### 参数
1. JSON 配置模式
- `galvatron_config_path`：字符串，json 配置路径，是否激活 JSON 配置模式。如果激活，全局配置模式中的参数将被忽略并被 JSON 配置覆盖。
2. 全局配置模式
- `global_train_batch_size`：整数，分布式训练的全局批量大小。
- `pp_deg`：整数，流水线（PP）度。
- `global_tp_deg`：整数，张量并行（TP）度。
- `global_tp_consec`：`0`/`1`，TP 的通信组是否连续（例如，[0,1,2,3] 是连续的，而 [0,2,4,6] 不是）。
- `sdp`：`0`/`1`，是否使用 SDP 代替 DP。
- `chunks`：整数，PP 的微批次数。
- `global_checkpoint`：`0`/`1`，是否对整个模型启用激活检查点。
- `pipeline_type`：`gpipe` 或 `pipedream_flush`，选择要使用的流水线类型。
- `vocab_tp`：整数，词表张量并行度。

### 其他训练优化
设置 `mixed_precision` 以允许混合精度训练，例如 `bf16`。设置 `use-flash-attn` 以允许使用 [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) 功能。

设置 `sequence-parallel` 以启用 `Megatron-TP-SP` 方法，这可以进一步减少内存使用。

设置 `use_ulysses` 以启用 [Ulysses-SP](https://github.com/microsoft/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md) 方法，这将替代 `Megatron-TP-SP`。一旦激活，TP（张量并行）维度将自动转换为 SP（序列并行）维度。

设置 `no_async_grad_reduce` 以禁用默认启用的异步梯度同步方法。在 Galvatron 中，在训练的每次迭代期间，当需要梯度累积时，默认行为是仅在所有反向传播完成后执行梯度 reduce scatter 操作。这种方法减少了通信开销但增加了额外的内存使用：每个设备在梯度同步之前都保持梯度的完整副本，导致 Zero-2 降级为 Zero-1。当设置 `no_async_grad_reduce` 时，Galvatron 在每个反向步骤后同步梯度，保持低内存使用。然而，这引入了额外的通信，尽管其中大部分可以与计算重叠。权衡是成本模型的复杂性增加，可能降低成本模型的准确性。我们计划在未来提供更细粒度和准确的成本模型。

有关训练参数的完整列表，请参考 [arguments.py](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/core/arguments.py) 中的 ```galvatron_training_args```。

**Ulysses 仅在 llama_hf、gpt_hf 上支持。**
