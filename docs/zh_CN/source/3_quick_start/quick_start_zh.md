# 快速入门

## 使用 Galvatron 进行性能分析
使用 Galvatron 的第一步是对硬件环境和模型计算时间进行性能分析。Galvatron 会自动将分析结果保存到配置文件中。

(1) 首先，要对硬件环境进行性能分析，```cd galvatron/profile_hardware```，将主机地址写入 ```hostfile```，在 ```scripts/profile_hardware.sh``` 中设置 ```NUM_NODES, NUM_GPUS_PER_NODE, MPI_PATH```，然后运行：
````shell
sh scripts/profile_hardware.sh
````

Galvatron 将调用 [nccl-tests](https://github.com/NVIDIA/nccl-tests) 或 [pytorch profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) 来分析通信带宽。你可以通过在 ```scripts/profile_hardware.sh``` 中将 ```--backend``` 设置为 ```nccl``` 或 ```torch``` 来选择其中之一。

对于```nccl```格式，用户需要设置以下变量：
- ```nccl_test_dir```: 用于指定nccl-tests的目录
- ```mpi_path```: 用于指定mpi的安装路径
- ```start_mb```: 用于指定开始分析的通信带宽大小
- ```end_mb```: 用于指定结束分析的通信带宽大小
- ```scale```: 用于指定通信带宽的缩放因子
- ```hostfile```: 用于指定主机文件，该文件中需要包含所有节点的IP地址或主机名

此外用户还需要设置环境变量```NCCLTEST_OTHER_ARGS```，该变量用于指定nccl-tests需要的额外环境变量，例如可以用于指定nccl-tests的IB设备。

对于```torch```格式，用户需要设置以下变量：
- ```master_addr```: 用于指定主节点的IP地址或主机名
- ```master_port```: 用于指定主节点的端口号
- ```node_rank```: 用于指定当前节点的rank
- ```envs```: 用于指定环境变量

在```torch```格式下，运行脚本并不会直接profile带宽，而是会在```scripts```目录下生成四个脚本，分别是```profile_allreduce```, ```profile_p2p```, ```profile_allreduce_sp```, ```profile_all2all_sp```。用户需要在所有节点依次运行这四个脚本，来获取不同通信模式下的带宽。
注意这里```master_addr```、```master_port```、```node_rank```可以设置成```'$xxx'```的形式，这样在生成脚本的时候保留变量名，运行脚本的时候再从环境变量中获取。

Gavlatron在默认脚本中提供了不同```backend```的配置文件，用户可以在此基础上进行修改。

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

有关自定义并行优化的更多使用详情，请参见 [Galvatron 模型使用](../4_galvatron_model_usage/galvatron_model_usage_zh.html#id3)。

## 使用 Galvatron 进行训练
Galvatron 提供了一种简单的方法来以细粒度混合并行方式训练 Transformer 模型。你可以通过指定参数 ```galvatron_config_path``` 使用搜索到的最优并行策略来训练 Transformer 模型以获得最佳吞吐量，或者按照自己的喜好使用任何并行策略。Galvatron 支持两种混合并行配置模式，包括 JSON 配置模式和全局配置模式。你可以通过修改少量参数来指定并行策略。

要使用 Galvatron 训练模型，```cd galvatron/models/model_name```，设置 ```NUM_NODES, NUM_GPUS_PER_NODE, MASTER_ADDR, MASTER_PORT, NODE_RANK```，然后运行：
````shell
sh scripts/train_dist_random.sh
````

使用 `--galvatron_config_path` 参数来应用从搜索引擎获得的并行策略。如果你已经准备好相关的数据集和检查点，可以通过修改和运行 `scripts/train_dist.sh` 来完成实际训练。

提示：在继续之前，请确认是否需要使用 `--set_seqlen_manually` 参数来手动指定训练模型的序列长度。

详细指南和更多自定义训练选项请参见 [Galvatron 模型使用](../4_galvatron_model_usage/galvatron_model_usage_zh.html#id9)。
