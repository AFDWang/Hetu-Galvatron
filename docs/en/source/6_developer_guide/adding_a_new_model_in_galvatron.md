## Adding a New Model in Galvatron

This guide will teach you how to add a new model in Galvatron.

### Directory Structure

The directory structure of a model in Galvatron is as follows:

```
MyModel/
├── meta_configs/                              # Directory for model configuration files
│   ├── __init__.py                            
│   ├── config_utils.py                        # Configuration utility functions
│   ├── MyModel-{MODEL_SIZE}b.json        # Model configuration
│   └── ...                                    # Other model size configurations
│
├── scripts/                                   # Directory for running scripts
│   ├── profile.sh                             # Profiling script
│   ├── train.sh                               # Training script
│   └── search.sh                              # Parallel strategy search script
│
├── __init__.py                                
├── arguments.py                               # Argument definitions
├── dataloader.py                              # Data loading implementation
├── profiler.py                                # Profiling entry point
├── search_dist.py                             # Parallel strategy search entry point
├── train.py                                   # Single-machine training entry point
├── train_dist.py                              # Distributed training entry point
├── train_dist_random.py                       # Random data training entry point
│
├── MyModelModel_checkpoint.py            # Checkpoint save/load
├── MyModelModel_hybrid_parallel.py       # Hybrid parallel implementation
├── MyModelModel_sequential.py            # Sequential model implementation
└── MyModelModel_tensor_parallel.py       # Tensor parallel implementation

```

### Galvatron's Hybrid Parallel Model Construction Process

Before adding a new model, let's understand the general process Galvatron uses for constructing hybrid parallel models.

Galvatron builds models without manually defining the entire model structure. Instead, it uses corresponding model structures from [transformers](https://github.com/huggingface/transformers) or [flash attention](https://github.com/Dao-AILab/flash-attention). You can add the suffix `hf` or `fa` to `MyModel` to distinguish the backend you choose for the model structure. If you're unsure which backend to choose, we recommend `hf` as Galvatron provides more comprehensive support for it (the `fa` model does not support the Ulysses-SP parallel method). The process of constructing a hybrid parallel model is detailed in [`construct_hybrid_parallel_model_api`](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/core/hybrid_parallel/model.py). The specific process is as follows:

1. **Preprocessing Configuration**: Obtain information such as hybrid parallel strategy and model configuration.

2. **Communication Group Generation** (Step 0): Generate communication groups required for various parallel strategies.

3. **Build Tensor Parallel Model** (Step 1): Use model-specific TP functions (defined in `MyModelModel_tensor_parallel.py`) to build a tensor parallel model.

4. **Build Sequential Model** (Step 2): Reconstruct the model using model-specific sequential functions (defined in `MyModelModel_sequential.py`).

5. **Wrap Redistribution Modules** (Step 3): Add data redistribution functionality to the model to ensure data distribution corresponds to the parallel strategy.

6. **Build Pipeline Parallelism** (Step 4): Construct a pipeline parallel model, placing different stages on corresponding devices.

7. **Wrap Data Parallel Modules** (Step 5): Wrap data parallel modules based on the FSDP library.

8. **Add Checkpoint Wrapping** (Step 6): Add checkpoint functionality to modules based on checkpoint configuration.

Only the API call and the implementations of Step 1 and Step 2 need to be completed using model-specific functions. The other steps are generally implemented by Galvatron.

### Core File Descriptions

The core of adding a new model is the model implementation files. These are the main parts that developers need to implement, defining the structure and implementation of the model.

#### 1. Tensor Parallel Implementation

The tensor parallel implementation is realized through the `MyModelModel_tensor_parallel.py` file, which defines the tensor parallel implementation of the model. Modules in the Sequential model need to be replaced with modules that support tensor parallelism. Galvatron provides different tensor parallel implementations based on different model backends. Specifically, `hf` uses Megatron-TP, and `fa` uses the TP provided by flash-attn.

For `hf`, you need to implement the `MyModelLayer_tp` class and the `MyModelAttention_tp` and `MyModelMLP_tp` classes. For `fa`, you can directly call the `create_mixer_cls` and `create_mlp_cls` methods from flash_attn. You also need to define the `construct_tensor_parallel_model` function to replace the TP model in the full model. Detailed examples can be found in [gpt_hf](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_hf/GPTModel_tensor_parallel.py) and [gpt_fa](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_fa/GPTModel_tensor_parallel.py).

##### 1.1 Transformer Layer (`hf` Model Format)

The Transformer layer is implemented through the `MyModelLayer_tp` class:

```python
class MyModelLayer_tp(nn.Module):
    def __init__(self, config, layer_number, tp_group=None, sp_group=None):
        """
        Parameters:
            config: Model configuration object, TransformerConfig
            layer_number: Index number of the current layer
            tp_group: Tensor parallel communication group, CommGroup
            sp_group: Sequence parallel communication group, CommGroup
        """
        super().__init__()
        self.attention = MyModelAttention_tp(config, layer_number, tp_group, sp_group)
        self.mlp = MyModelMLP_tp(config, tp_group)
        self.idx = layer_number
        
    def forward(self, hidden_states, attention_mask=None):
        # ...
        pass
```

This class is mainly responsible for defining the implementation of a Transformer layer, including the attention mechanism and feedforward neural network. Note that defining `self.idx` is necessary for distinguishing layers later, and `config` directly uses the `TransformerConfig` class used when creating the model in the Transformer library.

##### 1.2 Attention Layer (`hf` Model Format)

The attention layer is implemented through the `MyModelAttention_tp` class:

```python
class MyModelAttention_tp(nn.Module):
    def __init__(self, config, layer_number, tp_group=None, sp_group=None):
        """
        Parameters:
            config: Model configuration object, TransformerConfig
            layer_number: Index number of the current layer
            tp_group: Tensor parallel communication group, CommGroup
            sp_group: Sequence parallel communication group, CommGroup
        """
        super().__init__()
        # ...
        megatron_config = core_transformer_config_from_args(args)
        self.attention = ParallelAttention(megatron_config, ...)
        # ...
    def forward(self, hidden_states, attention_mask):
        # ...
        pass
```

`ParallelAttention` is the attention layer implementation in Megatron-TP modified by Galvatron. In the original Megatron-TP attention layer implementation, three parameters are added: `tp_group`, `sp_group`, and `use_ulysses`, representing the tensor parallel communication group, sequence parallel communication group, and whether to use Ulysses sequence parallelism, respectively. Generally, you can directly refer to the example of [gpt_hf](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_hf/GPTModel_tensor_parallel.py) for implementation.

##### 1.3 Feedforward Neural Network Layer (`hf` Model Format)

The feedforward neural network layer is implemented through the `MyModelMLP_tp` class:

```python
class MyModelMLP_tp(nn.Module):
    def __init__(self, config, tp_group=None):
        """
        Parameters:
            config: Model configuration object, TransformerConfig
            tp_group: Tensor parallel communication group, CommGroup
        """
        super().__init__()
        # ...
        megatron_config = core_transformer_config_from_args(get_args())
        self.mlp = ParallelMLP(megatron_config, tp_group = self.tp_group)
        # ...
    def forward(self, hidden_states):
        # ...
        pass
```

`ParallelMLP` is the feedforward neural network layer implementation in Megatron-TP modified by Galvatron. In the original Megatron-TP attention layer implementation, the `tp_group` parameter is added to represent the tensor parallel communication group. Generally, you can directly refer to the example of [gpt_hf](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_hf/GPTModel_tensor_parallel.py) for implementation.

##### 1.4 Constructing Tensor Parallel Model (`hf` Model Format)

The tensor parallel model is constructed through the `construct_tensor_parallel_model` function:

```python
def construct_tensor_parallel_model(model, config, tp_groups_enc, sp_groups_enc):
    """
    Convert the model to a tensor parallel version
    
    Parameters:
        model: Original model instance
        config: Model configuration object, TransformerConfig
        tp_groups_enc: List of tensor parallel communication groups for each layer, List[CommGroup]
        sp_groups_enc: List of sequence parallel communication groups for each layer, List[CommGroup]
        
    Returns:
        Converted tensor parallel model
    """
    # ...
    pass
```

This function mainly performs three tasks: replacing the Transformer Layer in the model with `MyModelLayer_tp`, replacing the embedding layer in the model with `VocabParallelEmbedding`, and replacing the lm_head in the model with `ColumnParallelLinear`. `VocabParallelEmbedding` and `ColumnParallelLinear` are the embedding layer and linear layer implementations in Megatron-TP modified by Galvatron, with the `tp_group` and `sp_group` parameters added to represent the tensor parallel communication group and sequence parallel communication group. You can also directly refer to the example of [gpt_hf](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_hf/GPTModel_tensor_parallel.py) for implementation.

Note: The communication groups used in these classes and functions are the CommGroup class customized by Galvatron. If you want to access communication groups generated by torch, please use `tp_group.group` and `sp_group.group`.

##### 1.5 Constructing Tensor Parallel Model (`fa` Model Format)

For `fa`, you only need to implement the `construct_tensor_parallel_model` function. In this function, you need to replace the attention and mlp modules in the Transformer Layer with the `create_mixer_cls` and `create_mlp_cls` methods from flash_attn, replace the embedding layer with the `ParallelGPT2Embeddings` method from flash_attn, and replace the lm_head with the `ColumnParallelLinear` method from flash_attn. A detailed example can be found in [gpt_fa](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_fa/GPTModel_tensor_parallel.py).

#### 2 Sequential Model Implementation

`MyModelModel_sequential.py` defines the sequential implementation of the model, including the implementation of the forward and backward propagation of the model.

For traditional Transformer models, you need to implement classes such as `MyModelEmbeddings_`, `MyModelLayers_`, `MyModelPreNorm_`, and `MyModelCls_`.

In addition, you need to implement the `construct_sequential_model` function to convert the model to a sequential model and the `MyModelModelInfo` class to define model-related information.

Specifically, the definition and format of each class are as follows:

##### 2.1 Embedding Layer

The embedding layer is implemented through the `MyModelEmbeddings_` class:

```python
class MyModelEmbeddings_(nn.Module):
    def __init__(self, model):
            """
            Parameters:
                model: Model instance
            """
            super().__init__()
            # ...
        def forward(self, tokens, **kwargs):
            # ...
            pass
```

This class is mainly used to define the embedding layer in the model, including word embedding, position embedding, etc.

Here, the `model` passed into the `__init__` function is the model obtained directly by calling transformers or flash-attn (the `model` in all APIs needs to be the model obtained by calling transformers or flash-attn).

To enhance the robustness of the code, this function also needs to support some additional features: Megatron sequence parallelism and Ulysses sequence parallelism (not supported by `fa`). Detailed examples can be found in [gpt_hf](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_hf/GPTModel_sequential.py) and [gpt_fa](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_fa/GPTModel_sequential.py).

Note: When using the `hf` backend, for files with multiple types of Embeddings (e.g., GPT has both Vocab and Position Embeddings), you need to define different Embedding classes to distinguish between these different Embedding parameters. An example of this is shown in [gpt_hf](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_hf/GPTModel_sequential.py).

##### 2.2 Transformer Layer

The Transformer layer is implemented through the `MyModelLayers_` class:

```python
class MyModelLayers_(nn.Module):
    def __init__(self, model, layer_idx):
        """
        Parameters:
            model: Model instance
            layer_idx: Index number of the current layer
        """
        super().__init__()
        # ...
    def forward(self, hidden_states, **kwargs):
        # ...
        pass
```

This class is mainly used to define the Transformer layer in the model, including the self-attention layer, feedforward neural network layer, etc.

For the `fa` backend, you need to decide whether to add residuals and dropout based on the actual model structure in the code.

##### 2.3 Normalization Layer

The normalization layer is implemented through the `MyModelPreNorm_` class:

```python
class MyModelPreNorm_(nn.Module):
    def __init__(self, model):
        """
        Parameters:
            model: Model instance
        """
        super().__init__()
        # ...
    def forward(self, hidden_states, **kwargs):
        # ...
        pass
```

This class is mainly used to define the normalization layer before the output layer of the model.

##### 2.4 Output Layer

The output layer is implemented through the `MyModelCls_` class:

```python
class MyModelCls_(nn.Module):
    def __init__(self, model):
        """
        Parameters:
            model: Model instance
        """
        super().__init__()
        # ...
    def forward(self, hidden_states, **kwargs):
        # ...
        pass
```

This class is mainly used to define the output layer of the model.

To enhance the robustness of the code, this function also needs to support some additional features: Megatron sequence parallelism, Ulysses sequence parallelism (not supported by `fa`), and parallel loss computation (not supported by `fa`). Detailed examples can be found in [gpt_hf](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_hf/GPTModel_sequential.py) and [gpt_fa](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_fa/GPTModel_sequential.py).

Note: When using the `hf` backend, to obtain `logits_parallel`, you need to directly reference the `.weight` variable of the original model. This is not allowed in FSDP, so you can place the code for obtaining `logits_parallel` in a separate function, represented by `MyModelLoss_`. An example of this is shown in [gpt_hf](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_hf/GPTModel_sequential.py).

When implementing these layers, special attention should be paid to ensuring that the input and output tensors (excluding `kwargs`) of the forward function of the same type of layer in the Transformer layer have the same format and size. This is to facilitate updating model information to ensure the correctness of pipeline parallelism. For example, in [gpt_hf](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_hf/GPTModel_sequential.py), the input and output tensors of the forward function of the Transformer layer have the same format and size, both being `hidden_states`.

##### 2.5 Constructing Sequential Model

The sequential model is constructed through the `construct_sequential_model` function:

```python
def construct_sequential_model(model, config):
    """
    Convert the model to a sequential version
    
    Parameters:
        model: Original model instance
        config: Model configuration object, TransformerConfig
        
    Returns:
        Converted sequential model
    """
    model_ = PipeSequential()
    # ...
```

This function converts the model into a `PipeSequential` format, a special sequential container specifically for pipeline parallelism. Developers only need to add the model sequentially to `PipeSequential` using the `add_module` method.

Note: If `MyModelLoss_` is used, you also need to add a `reset_parameters` method to ensure the model can be initialized correctly.

##### 2.6 Model Information

Model information is implemented through the `MyModelModelInfo` class:

```python
class MyModelModelInfo(ModelInfo):
    def __init__(self, config, args):
        super(MyModelModelInfo, self).__init__()
        # ...
        self.set_layernums(layernum_list)
        self.set_shapes(layer_shapes_list)
        self.set_dtypes(layer_dtypes_list)
        self.set_module_types(module_types)
```

In this class, you need to assign four variables: `layernums`, `shapes`, `dtypes`, and `module_types`, representing the number of each type of Transformer layer, the shape of input and output tensors for each type of layer, the data type of input and output tensors for each type of layer, and the name of each layer in the model, respectively.

For `layernums`, you need to assign a list, where each element represents the number of each type of Transformer layer. For example, for GPT, the length of the list is 1 because GPT only has one type of Decoder layer. But for T5, the length of the list is 2 because T5 contains both Encoder and Decoder layers, and these two types of layers have different structures.

For `shapes`, you need to assign a list, where each element represents the shape of input and output tensors for each type of Transformer layer. Typically, this is a list of size `[x, y]`, where `x` represents the number of Transformer layer types, and `y` represents the number of input and output tensors per layer. Each value in the list stores the shape of the input and output tensors.

For `dtypes`, you need to assign a list, where each element represents the data type of input and output tensors for each type of Transformer layer. Typically, this is a list of size `[x, y]`, where `x` represents the number of Transformer layer types, and `y` represents the number of input and output tensors per layer. Each value in the list stores the data type of the input and output tensors.

For `module_types`, you need to assign a list where each element sequentially represents the name of each layer in the model.

#### 3 Hybrid Parallel Implementation

The hybrid parallel implementation is realized through the `MyModelModel_hybrid_parallel.py` file. This file acts as a bridge connecting the model with the Galvatron parallel system, mainly responsible for constructing model instances that support hybrid parallelism.

This file primarily implements four functions: `get_hybrid_parallel_configs`, `construct_hybrid_parallel_model`, `get_mymodel_config`, and `mymodel_model_hp`.

##### 3.1 Getting Hybrid Parallel Configurations

The `get_hybrid_parallel_configs` function is used to obtain hybrid parallel strategies, with the implementation format as follows:

```python
def get_hybrid_parallel_configs(model_config, training_args):
    hybrid_parallel_configs = get_hybrid_parallel_configs_api(model_config, training_args, MyModelModelInfo)
    return hybrid_parallel_configs
```

This function requires no modifications. It obtains hybrid parallel strategies by calling Galvatron's `get_hybrid_parallel_configs_api` function and returns a dictionary containing hybrid parallel strategy information.

##### 3.2 Constructing Hybrid Parallel Model

The `construct_hybrid_parallel_model` function is used to construct a hybrid parallel model, with the implementation format as follows:

```python
def construct_hybrid_parallel_model(model, model_config, training_args, hybrid_parallel_configs):
    # ...
    hp_model = construct_hybrid_parallel_model_api(...)
    return hp_model
```

This function constructs a hybrid parallel model by calling Galvatron's `construct_hybrid_parallel_model_api` function and returns a model instance that supports hybrid parallelism. Specifically, the parameters and format required by this API function are as follows:

```python
def construct_hybrid_parallel_model_api(
    model, # Original model instance   
    model_config, # Model configuration object
    training_args, # Training parameters
    hybrid_parallel_configs, # Hybrid parallel configuration
    model_info, # Model information class
    construct_sequential_model, # Function to construct sequential model
    construct_tensor_parallel_model, # Function to construct tensor parallel model
    wrap_block_name=None, # List of module names to wrap with FSDP
    wrap_checkpoint_block_name=None, # List of module names to add checkpoints
    wrap_other_block_name=None, # List of other module names to wrap with FSDP
    tied_wte_attr_names=None, # List of attribute names for weight tying
    layernorm_name = [], # List of layer normalization names
    all_block_name = None, # List of all module names
    load_module_func = None, # Function to load module
):
    # ...
    pass
```

Parameters can be directly referenced from the implementation of [gpt_hf](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_hf/GPTModel_hybrid_parallel.py) and [gpt_fa](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_fa/GPTModel_hybrid_parallel.py).

Here, we provide additional explanations for some optional parameters that may cause confusion:

- `wrap_block_name`: A list of Transformer layer module classes that need to be wrapped with FSDP.
- `wrap_checkpoint_block_name`: A list of module names that require checkpoints, usually Transformer layers.
- `wrap_other_block_name`: A list of other module names that need to be wrapped with FSDP, usually layers other than Transformer layers. Note that if multiple Embedding classes are defined, all fine-grained Embedding classes need to be added to the list.
- `tied_wte_attr_names`: A list of attribute names for weight tying. For some models, the parameters of the Vocab Embedding layer and the output layer are the same. For models requiring this feature, developers need to inform Galvatron how to access the Vocab Embedding layer in both the first and last layers of the model. For example, in [gpt_hf](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_hf/GPTModel_sequential.py), the Embedding layer accesses the `GPTVocabEmbedding_` class via `self.wte`, while the output layer accesses it directly via `self` in the Cls layer. Therefore, `tied_wte_attr_names` is `['wte', '']`.
- `layernorm_name`: A list of names used to identify how Galvatron should access Layernorm in different layers (only the suffix is needed, not the full name). For example, in [gpt_hf](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_hf), Layernorm is accessed via `self.LayerNorm` in the `GPTAttention_tp` and `GPTMLP_tp` classes, and via `self.ln` in `GPTPreNorm_`. Therefore, `layernorm_name` is `['LayerNorm', 'ln']`.
- `all_block_name`: A list of all module names, usually the union of `wrap_block_name` and `wrap_other_block_name`.
- `load_module_func`: A function to load the module, usually defined as the `load_MyModel_module` function in the `MyModelModel_checkpoint.py` file.

Note: Although `wrap_block_name`, `wrap_checkpoint_block_name`, `wrap_other_block_name`, and `all_block_name` are optional parameters in `construct_hybrid_parallel_model_api`, to ensure that the model can be initialized correctly, these parameters must be provided.

##### 3.3 Getting Model Configuration

The `get_mymodel_config` function is used to get the model configuration, with the implementation format as follows:

```python
def get_mymodel_config(args):
    config = config_from_meta(args.model_size)
    config = set_model_config(config, args, True)
    if args.local_rank == 0:
        print(config)
    return config
```

##### 3.4 Building Hybrid Parallel Model

The `mymodel_model_hp` function is used to build a hybrid parallel model, with the implementation format as follows:

```python
def mymodel_model_hp(config, args):
    hybrid_parallel_configs = get_hybrid_parallel_configs(model_config=config, training_args=args)
    if args.local_rank == 0:
        print("Creating Model...")
    mymodel_model = MyModelModel_huggingface(config)
    model = construct_hybrid_parallel_model(
        model=mymodel_model, 
        model_config=config, 
        training_args=args, 
        hybrid_parallel_configs=hybrid_parallel_configs
    )
    return model
```

Note that `MyModelModel_huggingface` is the model obtained directly through transformers, not the Galvatron model. When selecting a model in huggingface, choose a model that includes the output layer.

#### 4 Model Checkpoint Save and Load Implementation (Experimental, support hf)

The model checkpoint save and load implementation is realized through the `MyModelModel_checkpoint.py` file, which defines the implementation of model checkpoint saving and loading, including checkpoint save and load functions.

This file needs to implement the `save_MyModel_module` and `load_MyModel_module` functions to implement the saving and loading of model checkpoints.

Galvatron stores and loads model checkpoints layer by layer, so pay attention to loading and storing them layer by layer during implementation.

[llama_hf](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/llama_hf/LlamaModel_checkpoint.py) demonstrates how to implement model checkpoint saving and loading.

### Auxiliary File Descriptions

#### 1 Model Configuration Files

Model configuration files define the model's configuration, including the model's structure, parameter size, etc.

##### 1.1 Model Configuration Storage File

`meta_configs/MyModel-{MODEL_SIZE}b.json`: Model configuration file used to store model configuration information.

##### 1.2 Model Configuration Processing File

- **meta_configs/config_utils.py**: This file mainly handles functions related to model configuration, which mainly include three parts:
    - Obtaining model configuration information: Obtain model configuration information by calling the `config_from_meta` function and write it into `TransformerConfig`.
    - Modifying model configuration information: Modify model configuration information based on the passed arguments by calling the `set_model_config` function, and modify the model configuration information in the arguments through the `overwrite_megatron_args` and `overwrite_model_args` functions.
    - Obtaining model-related information: Obtain the model name through the `model_name` function and obtain the configuration information of each layer of the model through the `model_layer_configs` function.

#### 2 Training Files

Training files mainly define functions related to training, including data loading, model training, etc.

##### 2.1 Main Training File

- **train_dist.py**: This file mainly handles functions related to distributed training.

A complete example is as follows:

```python
def train(args):
    # Initialize the distributed training environment
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()

    config = get_mymodel_config(args)
    model = mymodel_model_hp(config, args)

    # Create dataset
    if local_rank == 0:
        print("Creating Dataset...")
    
    # Set dataset-related parameters    
    set_megatron_args_for_dataset(args, model, 
                                 model.sp_groups_whole[0] if args.vocab_sp else model.tp_groups_whole[0], 
                                 model.dp_groups_whole[0])
    if local_rank == 0:
        _print_args("arguments", args)

    # Get data iterators
    train_data_iterator, valid_data_iterator, test_data_iterator = get_train_valid_test_data_iterators()
    
    # Create optimizer and learning rate scheduler
    optimizer, opt_param_scheduler = get_optimizer_and_param_scheduler(model, args)

    # Set profiler
    path = os.path.dirname(os.path.abspath(__file__))
    profiler = GalvatronProfiler(args)
    profiler.set_profiler_dist(path, model_layer_configs(config), model_name(config), start_iter=0)
    
    # Record memory usage after model creation
    profiler.profile_memory(0, "After creating model")
    if local_rank == 0:
        print("Start training...")

    # Training loop
    for iter in range(args.iteration, args.train_iters):
        # Get a batch of data
        tokens, kwargs, loss_func = get_batch(train_data_iterator)
        
        # Record start time and memory usage
        profiler.profile_time_start(iter)
        profiler.profile_memory(iter, "Before Forward")

        # Prepare input data
        input_ids = tokens
        batch = [input_ids]
        
        # Forward and backward propagation
        loss = model.forward_backward(batch, iter, profiler, 
                                      loss_func=loss_func,
                                      **kwargs)
        
        # Record memory usage after backward propagation
        profiler.profile_memory(iter, "After Backward")
        
        # Gradient clipping
        total_norm = clip_grad_norm(model, args.clip_grad)
        
        # Optimizer step
        optimizer.step()
        # Learning rate scheduler step
        opt_param_scheduler.step(increment=args.global_batch_size)
        
        # Record memory usage after optimizer step
        profiler.profile_memory(iter, "After optimizer_step")
        
        # Zero gradients
        optimizer.zero_grad()

        # Update profiler statistics
        profiler.post_profile_memory(iter)
        # Get current learning rate
        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']
        # Record performance metrics for this iteration
        profiler.profile_time_end(iter, loss, learning_rate, total_norm)
        
        # Synchronize all processes
        torch.distributed.barrier()

        # Periodically save model checkpoints
        if args.save != None and (iter + 1) % args.save_interval == 0:
            save_llama_module(args.save, model, optimizer, opt_param_scheduler, iter + 1, args)

if __name__ == '__main__':
    # Initialize Galvatron training environment
    args = initialize_galvatron(model_args, mode='train_dist')
    # Set random seed for reproducibility
    set_seed()
    # Start training
    train(args)
```

- **train_dist_random.py**: This file mainly handles functions related to distributed training, similar to `train_dist.py`, but uses random data for training.

##### 2.2 Data Loading Files

- **dataloader.py**: This file mainly handles functions related to data loading, which mainly include two parts:
    - Random Data Loading: Create a dataset that generates random tokens and create a `collate_fn` function to convert random tokens into model inputs. Below is an example of random data loading:
    ```python
    def random_get_ltor_masks_and_position_ids(data):
    """Build masks and position id for left to right model."""
        micro_batch_size, seq_length = data.size()
        att_mask_batch = 1
        attention_mask = torch.tril(torch.ones(
            (att_mask_batch, seq_length, seq_length), device=data.device)).view(
                att_mask_batch, 1, seq_length, seq_length)
        attention_mask = (attention_mask < 0.5)

        return attention_mask

    def random_collate_fn(batch):
        # Stack data in the batch and return data in the corresponding format
        tokens_ = torch.stack(batch, dim=0)
        labels = tokens_[:, 1:].contiguous()
        tokens = tokens_[:, :-1].contiguous()
        args = get_args()
        if not args.use_flash_attn:
            attention_mask = random_get_ltor_masks_and_position_ids(tokens)
        else:
            attention_mask = None
        return tokens, {"attention_mask":attention_mask, "labels" : labels}, None

    class DataLoaderForMyModel(Dataset):
        def __init__(self, args, device, dataset_size = 2560 * 16):
            self.vocab_size = args.vocab_size
            self.sentence_length = args.seq_length
            self.dataset_size = dataset_size
            # Randomly generate the actual length of each sample (between 1 and the maximum length)
            self.data_length = np.random.randint(1,self.sentence_length+1,(self.dataset_size,))
            self.device = device

            # Generate random input data
            self.input_ids = []
            for i in range(self.dataset_size):
                sentence = np.random.randint(0,self.vocab_size,(self.sentence_length,))
                sentence[self.data_length[i]:] = 0
                mask = np.ones((self.sentence_length,))
                mask[self.data_length[i]:] = 0
                
                padding_sentence = np.zeros(self.sentence_length + 1, dtype=sentence.dtype)
                padding_sentence[:self.sentence_length] = sentence
                self.input_ids.append(padding_sentence)
            
            self.input_ids = np.array(self.input_ids)

        def __len__(self):
            return self.dataset_size

        def __getitem__(self, idx):
            if idx >= self.dataset_size:
                raise IndexError
            input_ids = torch.LongTensor(self.input_ids[idx]).to(self.device)
            return input_ids
    ```

    The specific `trainloader` is created by the following code:

    ```python
    trainloader = distributed_dataloader(
        dataset=DataLoaderForGPT(args, device),
        global_bsz=args.global_train_batch_size,
        shuffle=True,
        args=args,
        group = model.dp_groups_whole[0].group,
        collate_fn = random_collate_fn
    )
    ```

    The `distributed_dataloader` function is a distributed data loader provided by Galvatron, used to create distributed data loaders.

    - Real Data Loading: Create a real data loader and design a loss calculation function.

    The implementation of real data loading is based on the Megatron dataset and mainly includes functions such as `train_valid_test_datasets_provider`, `get_train_valid_test_data_iterators`, `get_batch`, and `loss_func`. A concrete implementation example can be found in [gpt_hf](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_hf/dataloader.py).

    The main point to note is that the `get_batch` function returns a tuple with three elements:

    - Input Data: Usually a sequence of tokens, of type `torch.Tensor`.
    - Other Input Data: Usually a dictionary type, containing `position_ids`, `attention_mask`, `labels`, etc.
    - Loss Calculation Function: The loss can be calculated directly by calling the `loss_func(output_tensor)` function.

    Note: The input data here should be consistent with the input data format of the Embedding layer in the `MyModelModel_sequential.py` file. Other data is passed between model layers as `**kwargs`.

##### 2.3 Profiling File

- **profiler.py**: This file mainly handles functions related to profiling, with content as follows:

```python
if __name__ == '__main__':
    # Initialize Galvatron profiling environment
    args = initialize_galvatron(model_args, mode='profile')
    
    # Load model configuration
    config = config_from_meta(args.model_size)
    config = set_model_config(config, args, overwrite_args=False)
    
    # Create profiler instance
    profiler = GalvatronProfiler(args)
    
    # Get the directory path of the current file
    path = os.path.dirname(os.path.abspath(__file__))
    
    # Set profiler launcher
    profiler.set_profiler_launcher(path, layernum_arg_names(), model_name(config))
    
    # Launch profiling scripts
    profiler.launch_profiling_scripts()
    
    # Process collected profiling data
    profiler.process_profiled_data()
```

##### 2.4 Strategy Search File

- **search_dist.py**: This file is primarily responsible for functions related to strategy search. Its contents are as follows:

```python
if __name__ == '__main__':
    args = initialize_galvatron(model_args, mode='search')
    config = config_from_meta(args.model_size)
    config = set_model_config(config, args, overwrite_args=True)
    path = os.path.dirname(os.path.abspath(__file__))
    print(args)
    print(config)
    # Create an instance of the strategy search engine
    search_engine = GalvatronSearchEngine(args)
    
    # Set basic information for the search engine
    search_engine.set_search_engine_info(path, model_layer_configs(config), model_name(config))
    
    # Initialize the search engine
    search_engine.initialize_search_engine()

    # Perform strategy search
    search_engine.parallelism_optimization()
```

#### 3 Script Files

The `scripts` folder mainly contains script files used to implement model training, performance analysis, strategy search, and other functions.

It mainly includes five different scripts:
- `profile_computation.sh`: Used for performance analysis, calculating the computational performance of the model under different configurations.
- `profile_memory.sh`: Used for performance analysis, calculating the memory usage of the model under different configurations.
- `search_dist.sh`: Used for strategy search, finding the optimal strategy for the model under different configurations.
- `train_dist.sh`: Used for model training.
- `train_dist_random.sh`: Used for model training with random data.
