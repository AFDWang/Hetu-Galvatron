# Quick Start

## Profiling with Galvatron
The first step to use Galvatron is to profile the hardware environment and the model computation time. Galvatron will automatically save the profiled results into config files.

(1) Firstly, to profile the hardward environment, ```cd galvatron/profile_hardware```,  write the host address into ```hostfile```, set ```NUM_NODES, NUM_GPUS_PER_NODE, MPI_PATH``` in ```scripts/profile_hardware.sh``` and run:
``` shell
sh scripts/profile_hardware.sh
```

Galvatron will call [nccl-tests](https://github.com/NVIDIA/nccl-tests) or [pytorch profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) to profile the communication bandwidth. You can choose one of them by setting ```--backend``` to ```nccl``` or ```torch``` in ```scripts/profile_hardware.sh```.

For ```nccl``` format, users need to set the following variables:
- ```nccl_test_dir```: the directory of nccl-tests
- ```mpi_path```: the path of mpi
- ```start_mb```: the start communication bandwidth
- ```end_mb```: the end communication bandwidth
- ```scale```: the scale of communication bandwidth
- ```hostfile```: the host file, which needs to contain the IP addresses or hostnames of all nodes

Additionally, users need to set the environment variable ```NCCLTEST_OTHER_ARGS```, which is used to specify the additional environment variables for nccl-tests. For example, it can be used to specify the IB device for nccl-tests.

For ```torch``` format, users need to set the following variables:
- ```master_addr```: the address of master node
- ```master_port```: the port of master node
- ```node_rank```: the rank of current node 
- ```envs```: the environment variables for torch

Additionally, users need to set the environment variable ```ENVS```, which is used to specify the environment variables for torch. 

In ```torch``` format, the script will not directly profile the bandwidth, but will generate four scripts, ```profile_allreduce```, ```profile_p2p```, ```profile_allreduce_sp```, ```profile_all2all_sp```. Users need to run these scripts on all nodes one by one to get the bandwidth of different communication modes.

Note that ```master_addr```, ```master_port```, ```node_rank``` can be set in the form of ```'$xxx'``` in ```scripts/profile_hardware.sh```, so that the variable names can be reserved in the generated scripts, and then retrieves them from environment variables when running the scripts.

Galvatron provides different configuration files for different ```backend``` in the default script. Users can modify them based on the default configurations.

(2) Secondly, to profile the model computation time and memory usage, ```cd galvatron/models/model_name``` and run:
``` shell
sh scripts/profile_computation.sh
sh scripts/profile_memory.sh
```

## Parallelism Optimizing with Galvatron
After profiling the environments, Galvatron is able to automatically optimize the parallelism strategy for the given Transformer model. Given the memory budget, Galvatron provides the fine-grained hybrid parallel strategy with maximum throughput. The optimized parallelism strategy will be saved in `galvatron/models/model_name/configs` for the training. You can train the model with the provided optimal strategy to obtain the optimal throughput. 

To conduct parallelim optimization, ```cd galvatron/models/model_name```, customize ```NUM_NODES, NUM_GPUS_PER_NODE, MEMORY``` in ```scripts/search_dist.sh```, run:

``` shell
sh scripts/search_dist.sh
```

The script will automatically run the search code in the background and generate the search log results in files beginning with `Search`. When you see the following marker in the file, it indicates that the search has concluded, and no other commands need to be executed before this point:

```
========================= Galvatron Search Engine End Searching =========================
```

After the search concludes, the parallel strategy obtained will be generated in the `configs` folder. The strategy is stored in JSON format, with file names starting with `galvatron_config_{model_size}_`.

See more usage details of the customized parallelism optimization in [Galvatron Model Usage](../4_galvatron_model_usage/galvatron_model_usage.html#parallelism-optimizing-with-galvatron).

## Training with Galvatron
Galvatron provides a simple way to train Transformer models in fined-grained hybrid parallelism fashion. You can either train Transformer models with the searched optimal parallel strategy by specifying argument ```galvatron_config_path``` to obtain the optimal throughput, or use any parallel strategies as they like. Galvatron support two hybrid parallel config modes, including JSON config mode and GLOBAL config mode. Ypi can specify parallel strategies by modifying only a few arguments. 

To train the model with Galvatron, ```cd galvatron/models/model_name```, set ```NUM_NODES, NUM_GPUS_PER_NODE, MASTER_ADDR, MASTER_PORT, NODE_RANK```,  and run:
``` shell
sh scripts/train_dist_random.sh
```

Use the `--galvatron_config_path` parameter to apply the parallel strategy obtained from the search engine. If you have the relevant datasets and checkpoints ready, you can complete the actual training by modifying and running `scripts/train_dist.sh`.

Tips: Before proceeding, ensure whether you need to use the `--set_seqlen_manually` parameter to manually specify the sequence length for the training model.

See detailed guidance and more customized training options in [Galvatron Model Usage](../4_galvatron_model_usage/galvatron_model_usage.html#training-with-galvatron).