# 快速入门

## 使用 Galvatron 进行性能分析
使用 Galvatron 的第一步是对硬件环境和模型计算时间进行性能分析。Galvatron 会自动将分析结果保存到配置文件中。

(1) 首先，要对硬件环境进行性能分析，```cd galvatron/profile_hardware```，将主机地址写入 ```hostfile```，在 ```scripts/profile_hardware.sh``` 中设置 ```NUM_NODES, NUM_GPUS_PER_NODE, MPI_PATH```，然后运行：
````shell
sh scripts/profile_hardware.sh
````

Galvatron 将调用 [nccl-tests](https://github.com/NVIDIA/nccl-tests) 或 [pytorch profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) 来分析通信带宽。你可以通过在 ```scripts/profile_hardware.sh``` 中将 ```--backend``` 设置为 ```nccl``` 或 ```torch``` 来选择其中之一。

(2) 其次，要分析模型计算时间和内存使用情况，```cd galvatron/models/model_name``` 并运行：
````shell
sh scripts/profile_computation.sh
sh scripts/profile_memory.sh
````

## 使用 Galvatron 进行并行优化
在对环境进行性能分析后，Galvatron 能够自动为给定的 Transformer 模型优化并行策略。给定内存预算，Galvatron 提供具有最大吞吐量的细粒度混合并行策略。优化后的并行策略将保存在 `galvatron/models/model_name/configs` 中用于训练。你可以使用提供的最优策略训练模型以获得最佳吞吐量。

要进行并行优化，```cd galvatron/models/model_name```，在 ```scripts/search_dist.sh``` 中自定义 ```NUM_NODES, NUM_GPUS_PER_NODE, MEMORY```，运行：

````shell
sh scripts/search_dist.sh
````

该脚本将在后台自动运行搜索代码，并在以 `Search` 开头的文件中生成搜索日志结果。当你在文件中看到以下标记时，表示搜索已结束，在此之前无需执行其他命令：

````
========================= Galvatron Search Engine End Searching =========================
````

搜索结束后，获得的并行策略将生成在 `configs` 文件夹中。策略以 JSON 格式存储，文件名以 `galvatron_config_{model_size}_` 开头。

有关自定义并行优化的更多使用详情，请参见 [Galvatron 模型使用](../4_galvatron_model_usage/galvatron_model_usage.html#parallelism-optimizing-with-galvatron)。

## 使用 Galvatron 进行训练
Galvatron 提供了一种简单的方法来以细粒度混合并行方式训练 Transformer 模型。你可以通过指定参数 ```galvatron_config_path``` 使用搜索到的最优并行策略来训练 Transformer 模型以获得最佳吞吐量，或者按照自己的喜好使用任何并行策略。Galvatron 支持两种混合并行配置模式，包括 JSON 配置模式和全局配置模式。你可以通过修改少量参数来指定并行策略。

要使用 Galvatron 训练模型，```cd galvatron/models/model_name```，设置 ```NUM_NODES, NUM_GPUS_PER_NODE, MASTER_ADDR, MASTER_PORT, NODE_RANK```，然后运行：
````shell
sh scripts/train_dist_random.sh
````

使用 `--galvatron_config_path` 参数来应用从搜索引擎获得的并行策略。如果你已经准备好相关的数据集和检查点，可以通过修改和运行 `scripts/train_dist.sh` 来完成实际训练。

提示：在继续之前，请确认是否需要使用 `--set_seqlen_manually` 参数来手动指定训练模型的序列长度。

详细指南和更多自定义训练选项请参见 [Galvatron 模型使用](../4_galvatron_model_usage/galvatron_model_usage.html#training-with-galvatron)。
