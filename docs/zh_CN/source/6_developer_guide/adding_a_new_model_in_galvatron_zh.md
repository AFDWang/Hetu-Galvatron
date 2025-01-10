## 在Galvatron中添加新模型

本指南将教你如何在Galvatron中添加新模型。

### 目录结构

一个模型在Galvatron中的目录结构如下；

```
MyModel/
├── meta_configs/                              # 模型配置文件目录
│   ├── __init__.py                            
│   ├── config_utils.py                        # 配置工具函数
│   ├── MyModel-{MODEL_SIZE}b.json        # 模型配置
│   └── ...                                    # 其他规模模型配置
│
├── scripts/                                   # 运行脚本目录
│   ├── profile.sh                             # 性能分析脚本
│   ├── train.sh                               # 训练脚本
│   └── search.sh                              # 并行策略搜索脚本
│
├── __init__.py                                
├── arguments.py                               # 参数定义
├── dataloader.py                              # 数据加载实现
├── profiler.py                                # 性能分析入口
├── search_dist.py                             # 并行策略搜索入口
├── train.py                                   # 单机训练入口
├── train_dist.py                              # 分布式训练入口
├── train_dist_random.py                       # 随机数据训练入口
│
├── MyModelModel_checkpoint.py            # 检查点保存加载
├── MyModelModel_hybrid_parallel.py       # 混合并行实现
├── MyModelModel_sequential.py            # 序列化模型实现
└── MyModelModel_tensor_parallel.py       # 张量并行实现
```

### Galvatron构建混合并行模型过程

在介绍如何加入新模型之前，我们先来了解一下Galvatron构建混合并行模型的大致过程。

Galvatron构建模型不需要手动定义模型整体结构，而是通过使用[transformers](https://github.com/huggingface/transformers)或[flash attention](https://github.com/Dao-AILab/flash-attention)中相应的模型结构，你可以在MyModel中添加`hf`或`fa`后缀来区分你所选择的模型结构后端。如果你不知道该选择什么样的模型结构后端，我们推荐你选择`hf`，因为Galvatron对`hf`的支持更加全面（`fa`模型不支持Ulysses-SP并行方法）。接着基于得到的模型结构构件混合并行模型的流程在[`construct_hybrid_parallel_model_api`](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/core/hybrid_parallel/model.py)中。其具体的流程如下：

1. **预处理配置**：获取混合并行策略、模型配置等信息

2. **通信组生成** （Step 0）：生成各种并行策略需要的通信组

3. **构建张量并行模型** （Step 1）：使用模型特定的 TP 函数（定义在`MyModelModel_tensor_parallel.py`中）构建张量并行模型

4. **构建序列模型** （Step 2）：使用模型特定的序列化函数重构模型（定义在`MyModelModel_sequential.py`中）

5. **包装重分布模块** （Step 3）：为模型添加数据重分布功能，保证每层的数据分布和并行策略对应

6. **构建流水线并行** （Step 4）：构建流水线并行模型，将不同的stage放置在对应设备上

7. **包装数据并行模块** （Step 5）：基于FSDP库包装数据并行模块

8. **添加检查点包装** （Step 6）：根据检查点配置为模块添加检查点功能

其中，只有该API的调用，以及Step1和Step2实现需要使用模型特定的函数完成，其他步骤都是Galvatron的通用实现。

### 核心文件说明

添加新模型的核心是模型实现文件，这是开发者需要实现的最主要的部分，它定义了模型的结构和实现。

#### 1 张量并行实现 

张量并行实现通过`MyModelModel_tensor_parallel.py`文件实现，该文件定义了模型的张量并行实现，需要将Sequential中的模块替换成支持张量并行的模块，这里Galvatron根据不同的模型后端，提供了不同的张量并行实现，具体来说，`hf`使用Megatron-TP，`fa`使用flash-attn提供的TP。

对于`hf`，你需要实现`MyModelLayer_tp`类，并实现`MyModelAttention_tp`和`MyModelMLP_tp`类，对于`fa`，则可以直接调用flash_attn的`create_mixer_cls`和 `create_mlp_cls`方法。同时你还需要定义`construct_tensor_parallel_model`函数，用于将完整模型进行TP模型替换。这方面的详细例子可以参考[gpt_hf](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_hf/GPTModel_tensor_parallel.py)和[gpt_fa](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_fa/GPTModel_tensor_parallel.py)。

##### 1.1 Transformer层 （`hf`模型格式）

Transformer层通过`MyModelLayer_tp`类实现:

```python
class MyModelLayer_tp(nn.Module):
    def __init__(self, config, layer_number, tp_group=None, sp_group=None):
        """
        参数:
            config: 模型配置对象，TransformerConfig
            layer_number: 当前层的索引编号
            tp_group: 当前层张量并行通信组，CommGroup
            sp_group: 当前层序列并行通信组，CommGroup
        """
        super().__init__()
        self.attention = MyModelAttention_tp(config, layer_number, tp_group, sp_group)
        self.mlp = MyModelMLP_tp(config, tp_group)
        self.idx = layer_number
        
    def forward(self, hidden_states, attention_mask=None):
        # ...
        pass
```

该类主要负责定义一层Transformer的实现，包括注意力机制和前馈神经网络，需要注意的是`self.idx`的定义是必要的，这关乎后面如何区分层，`config`则直接使用创建Transformer库中的模型时使用的`TransformerConfig`类。

##### 1.2 注意力层 （`hf`模型格式）

注意力层通过`MyModelAttention_tp`类实现:

```python
class MyModelAttention_tp(nn.Module):
    def __init__(self, config, layer_number, tp_group=None, sp_group=None):
        """
        参数:
            config: 模型配置对象，TransformerConfig
            layer_number: 当前层的索引编号
            tp_group: 张量并行通信组，CommGroup
            sp_group: 序列并行通信组，CommGroup
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

`ParallelAttention`是Galvatron修改后的Megatron-TP中的注意力层实现，在原版Megatron-TP的注意力层实现中，增加了tp_group、sp_group、use_ulysses三个参数，分别表示张量并行通信组、序列并行通信组、是否使用Ulysses序列并行，通常来说你可以直接参考[gpt_hf](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_hf/GPTModel_tensor_parallel.py)的例子实现这部分。

##### 1.3 前馈神经网络层（`hf`模型格式）

前馈神经网络层通过`MyModelMLP_tp`类实现:
```python
class MyModelMLP_tp(nn.Module):
    def __init__(self, config, tp_group=None):
        """
        参数:
            config: 模型配置对象，TransformerConfig
            tp_group: 张量并行通信组，CommGroup
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

`ParallelMLP`是Galvatron修改后的Megatron-TP中的前馈神经网络层实现，在原版Megatron-TP的注意力层实现中，增加了tp_group这个参数，用于表示张量并行通信组，通常来说你可以直接参考[gpt_hf](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_hf/GPTModel_tensor_parallel.py)的例子实现这部分。

##### 1.4 构造张量并行模型（`hf`模型格式）

构造张量并行模型通过`construct_tensor_parallel_model`函数实现:

```python
def construct_tensor_parallel_model(model, config, tp_groups_enc, sp_groups_enc):
    """
    将模型转换为张量并行版本
    
    参数:
        model: 原始模型实例
        config: 模型配置对象，TransformerConfig
        tp_groups_enc: 每一层的张量并行通信组列表，List[CommGroup]
        sp_groups_enc: 每一层的序列并行通信组列表，List[CommGroup]
        
    返回:
        转换后的张量并行模型
    """
    # ...
    pass
```

该函数主要完成三件事：将模型中的Transformer Layer替换为`MyModelLayer_tp`，将模型中的embedding层替换为`VocabParallelEmbedding`，将模型中的lm_head替换为`ColumnParallelLinear`。`VocabParallelEmbedding`和`ColumnParallelLinear`是同样是Galvatron修改后的Megatron-TP中的嵌入层和线性层实现，增加了tp_group和sp_group这两个参数，用于表示张量并行通信组和序列并行通信组，你也可以直接参考[gpt_hf](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_hf/GPTModel_tensor_parallel.py)的例子实现这部分。

注意：这些类和函数中用到的通信组是Galvatron自定义的CommGroup类，如果你想访问torch生成的通信组，请使用`tp_group.group`和`sp_group.group`。

##### 1.5 构造张量并行模型（`fa`模型格式）

对于`fa`，你只需要实现`construct_tensor_parallel_model`函数即可，在该函数中你需要将Transformer Layer中的attention和mlp模块分别替换为flash_attn的`create_mixer_cls`和 `create_mlp_cls`方法，将embedding层替换为flash_attn的`ParallelGPT2Embeddings`方法，将lm_head替换为flash_attn的`ColumnParallelLinear`方法。详细的例子请参考[gpt_fa](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_fa/GPTModel_tensor_parallel.py)。

#### 2 序列化模型实现

`MyModelModel_sequential.py`定义了模型的序列化实现，包括模型的前向传播和反向传播实现。

对于传统的Transformer模型，你需要实现`MyModelEmbeddings_`, `MyModelLayers_`, `MyModelPreNorm_`, `MyModelCls_` 等类。

此外，还需要实现`construct_sequential_model`函数，用于将模型转换为序列化模型。以及`MyModelModelInfo`类，用于定义模型相关信息。

具体来说，每个类的定义和格式如下：

##### 2.1 嵌入层

嵌入层通过`MyModelEmbeddings_`类实现:

```python
class MyModelEmbeddings_(nn.Module):
    def __init__(self, model):
            """
            参数:
                model: 模型实例
            """
            super().__init__()
            # ...
        def forward(self, tokens, **kwargs):
            # ...
            pass
```

该类主要用于定义模型中的嵌入层，包括词嵌入、位置嵌入等。

这里`__init__`函数中需要传入的`model`是直接通过调用transformers或flash-attn获取到的模型（所有API中`model`都需要传入transformers或flash-attn获取到的模型）。

为了增强代码的健壮性，该函数还需要支持一些额外的特性：Megatron序列并行、Ulysses序列并行（`fa`不支持）,这方面的详细例子可以参考[gpt_hf](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_hf/GPTModel_sequential.py)和[gpt_fa](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_fa/GPTModel_sequential.py)。

注意：当使用`hf`后端时，对于有多种Embedding类型的文件（比如GPT同时拥有Vocab和Position Embedding），需要额外定义不同的Embedding类以区分这两种不同的Embedding参数，[gpt_hf](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_hf/GPTModel_sequential.py)中展示了这样的一个例子。

##### 2.2 Transformer层

Transformer层通过`MyModelLayers_`类实现:

```python
class MyModelLayers_(nn.Module):
    def __init__(self, model, layer_idx):
        """
        参数:
            model: 模型实例
            layer_idx: 当前层的索引编号
        """
        super().__init__()
        # ...
    def forward(self, hidden_states, **kwargs):
        # ...
        pass
```

该类主要用于定义模型中的Transformer层，包括自注意力层、前馈神经网络层等。

对于`fa`后端，需要根据代码中实际的模型结构，决定是否添加残差和dropout。

##### 2.3 归一化层

归一化层通过`MyModelPreNorm_`类实现:

```python
class MyModelPreNorm_(nn.Module):
    def __init__(self, model):
        """
        参数:
            model: 模型实例
        """
        super().__init__()
        # ...
    def forward(self, hidden_states, **kwargs):
        # ...
        pass
```

该类主要用于定义模型中输出层前的归一化层。

##### 2.4 输出层

输出层通过`MyModelCls_`类实现:

```python
class MyModelCls_(nn.Module):
    def __init__(self, model):
        """
        参数:
            model: 模型实例
        """
        super().__init__()
        # ...
    def forward(self, hidden_states, **kwargs):
        # ...
        pass
```

该类主要用于定义模型的输出层。

为了增强代码的健壮性，该函数还需要支持一些额外的特性：Megatron序列并行、Ulysses序列并行（`fa`不支持）、并行求loss（`fa`不支持）,这方面的详细例子可以参考[gpt_hf](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_hf/GPTModel_sequential.py)和[gpt_fa](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_fa/GPTModel_sequential.py)。

注意：当使用`hf`后端时，获取`logits_parallel`需要直接引用原模型的`.weight`变量，这一点在FSDP中是不允许的，因此可以单独将获取`logits_parallel`的代码放在一个单独的函数中，用`MyModelLoss_`来表示，[gpt_hf](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_hf/GPTModel_sequential.py)中展示了这样的一个例子。

在实现这些层时，需要特别注意，Transformer层中相同种类的层的forward函数输入张量（`kwargs`除外）和输出张量的格式和大小相同，这是为了方便更新模型信息，以保证流水线并行的正确性。例如在[gpt_hf](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_hf/GPTModel_sequential.py)中，Transformer层的forward函数输入张量和输出张量的格式和大小相同，都是hidden_states。

##### 2.5 构造序列化模型

构造序列化模型通过`construct_sequential_model`函数实现:

```python
def construct_sequential_model(model, config):
    """
    将模型转换为序列化版本
    
    参数:
        model: 原始模型实例
        config: 模型配置对象，TransformerConfig
        
    返回:
        转换后的序列化模型
    """
    model_ = PipeSequential()
    # ...
```

这个函数将模型转化为`PipeSequential` 格式，它是一个特殊的序列容器，专门用于流水线并行。开发者只需要把模型按照顺序顺次通过`add_module`方法添加到`PipeSequential`中即可。

注意：如果使用了`MyModelLoss_`，还需要给其增加reset_parameters方法，以保证模型可以正确初始化。

##### 2.6 模型信息

模型信息通过`MyModelModelInfo`类实现:

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

在该类中，需要赋值四个变量：`layernums`、`shapes`、`dtypes`、`module_types`，分别表示每种不同类型的Transformer层数，每种类型层的输入输出张量形状、每种类型层输入输出张量的数据类型、模型每一层的模型名称。

对于`layernums`，需要赋值一个列表，列表中的每个元素表示每种类型Transformer层的数量，例如对于GPT，列表的长度为1，因为GPT只有一种Decoder层，但对于T5，列表的长度为2，因为T5同时包含Encoder和Decoder层，这两种层的结构是不同的。

对于`shapes`，需要赋值一个列表，列表中的每个元素表示每种类型Transformer层的输入输出张量形状，通常是一个大小为`[x,y]`的列表，x表示Transformer层的种类，y表示每层输入输出张量的数量，列表中的每个值存储的是输入输出张量的形状。

对于`dtypes`，需要赋值一个列表，列表中的每个元素表示每种类型Transformer层的输入输出张量的数据类型，通常是一个大小为`[x,y]`的列表，x表示Transformer层的种类，y表示每层输入输出张量的数量，列表中的每个值存储的是输入输出张量的数据类型。

对于`module_types`，需要赋值一个列表，列表中的每个元素顺次表示模型中每一层的名称。

#### 3 混合并行实现

混合并行实现通过`MyModelModel_hybrid_parallel.py`文件实现，该文件是连接模型与Galvatron并行系统的桥梁，主要负责构建支持混合并行的模型实例。

该文件主要实现了四个函数：`get_hybrid_parallel_configs`，`construct_hybrid_parallel_model`，`get_mymodel_config`，`mymodel_model_hp`。

##### 3.1 获取混合并行配置

`get_hybrid_parallel_configs`函数用于获取混合并行策略，其实现格式如下：

```python
def get_hybrid_parallel_configs(model_config, training_args):
    hybrid_parallel_configs = get_hybrid_parallel_configs_api(model_config, training_args, MyModelModelInfo)
    return hybrid_parallel_configs
```

该函数不需要任何改动，通过调用Galvatron的`get_hybrid_parallel_configs_api`函数获取混合并行策略，并返回一个字典，字典中包含混合并行策略信息。

##### 3.2 构建混合并行模型

`construct_hybrid_parallel_model`函数用于构建混合并行模型，其实现格式如下：

```python
def construct_hybrid_parallel_model(model, model_config, training_args, hybrid_parallel_configs):
    # ...
    hp_model = construct_hybrid_parallel_model_api(...)
    return hp_model
```

该函数通过调用Galvatron的`construct_hybrid_parallel_model_api`函数构建混合并行模型，并返回一个支持混合并行的模型实例。具体来说，该API函数具体需要的参数和格式如下：

```python
def construct_hybrid_parallel_model_api(
    model, # 原始模型实例   
    model_config, # 模型配置对象
    training_args, # 训练参数
    hybrid_parallel_configs, # 混合并行配置
    model_info, # 模型信息类
    construct_sequential_model, # 构建序列化模型的函数
    construct_tensor_parallel_model, # 构建张量并行模型的函数
    wrap_block_name=None, # 需要包装FSDP的模块名称列��
    wrap_checkpoint_block_name=None, # 需要添加检查点的模块名称列表
    wrap_other_block_name=None, # 需要包装FSDP的其他模块名称列表
    tied_wte_attr_names=None, # 权重绑定的属性名称列表
    layernorm_name = [], # 层归一化的名称列表
    all_block_name = None, # 所有模块的名称列表
    load_module_func = None, # 加载模块的函数
):
    # ...
    pass
```

参数可以直接参考[gpt_hf](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_hf/GPTModel_hybrid_parallel.py)和[gpt_fa](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_fa/GPTModel_hybrid_parallel.py)的实现。

在此，我们额外对一些可能感到疑惑的可选参数进行解释：

- `wrap_block_name`：需要包装FSDP的Transfomer层模块类列表。
- `wrap_checkpoint_block_name`：需要添加检查点的模块名称列表，通常是Transformer层。
- `wrap_other_block_name`：需要包装FSDP的其他模块名称列表，通常是Transformer层以外的其它层，注意这里如果定义了多个Embedding类，需要将所有细粒度Embedding类都添加到列表中。
- `tied_wte_attr_names`：权重绑定的属性名称列表，部分模型Vocab Embedding层和输出层的参数是相同的，对于需要这种需求的模型，开发者需要将模型第一层和最后一层中如何访问Vocab Embedding层的方式告诉Galvatron，例如对于[gpt_hf](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_hf/GPTModel_sequential.py)，`GPTVocabEmbedding_`类在Embedding层通过self.wte访问，而输出层在Cls层直接通过self访问即可，因此tied_wte_attr_names为`['wte'，'']`。
- `layernorm_name`：用于标识Galvatron在不同的层该如何访问Layernorm的名称列表（不需要完整名称，只需要知道后缀名词即可），例如对于[gpt_hf](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_hf)，Layernorm在`GPTAttention_tp`和`GPTMLP_tp`类中通过`self.LayerNorm`访问，在`GPTPreNorm_`中通过`self.ln`访问，因此`layernorm_name`为`['LayerNorm', 'ln']` 。
- `all_block_name`：所有模块的名称列表，通常是`wrap_block_name`和`wrap_other_block_name`的并集。
- `load_module_func`：加载模块的函数，通常是定义在`MyModelModel_checkpoint.py`文件中的`load_MyModel_module`函数。

注意：虽然`wrap_block_name`、`wrap_checkpoint_block_name`、`wrap_other_block_name`、`all_block_name`这些参数在`construct_hybrid_parallel_model_api`中是可选参数，但为了保证模型可以正确初始化，这些参数必须传入。

##### 3.3 获取模型配置

`get_mymodel_config`函数用于获取模型配置，其实现格式如下：

```python
def get_mymodel_config(args, overwrite_args=True):
    config = config_from_meta(args.model_size)
    config = set_model_config(config, args, overwrite_args)
    if hasattr(args, 'local_rank') and args.local_rank == 0:
        print(config)
    return config
```

##### 3.4 构建混合并行模型

`mymodel_model_hp`函数用于构建混合并行模型，其实现格式如下：

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

注意这里`MyModelModel_huggingface`是直接通过transformers获取到的模型，而不是Galvatron的模型。在huggingface中选择模型时，需要选择包含输出层的模型。

#### 4 模型检查点保存加载实现（Experimental, 支持hf）

模型检查点保存加载实现通过`MyModelModel_checkpoint.py`文件实现，该文件定义了模型的检查点保存和加载实现，包括检查点的保存和加载函数。

该文件需要实现`save_MyModel_module`和`load_MyModel_module`函数。用于实现模型检查点的保存和加载。

Galvatron是按层存储和加载模型检查点的，因此在实现时需要注意按层进行加载和存储。

[llama_hf](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/llama_hf/LlamaModel_checkpoint.py)中展示了如何实现模型检查点的保存和加载。

### 辅助文件说明

#### 1 模型配置文件

模型配置文件定义了模型的配置，包括模型的结构、参数量等。

##### 1.1 模型配置存储文件

`meta_configs/MyModel-{MODEL_SIZE}b.json`：模型配置文件，用于存储模型配置信息。

##### 1.2 模型配置处理文件

- **meta_configs/config_utils.py**：该文件主要负责处理模型配置相关的功能，其主要包括三部分：
    - 获取模型配置信息：通过调用`config_from_meta`函数获取模型配置信息，并写入到`TransformerConfig`中。
    - 修改模型配置信息：通过调用`set_model_config`函数，根据传入的arguments修改模型配置信息，并通过`overwrite_megatron_args`和`overwrite_model_args`函数修改arguments中的模型配置信息。
    - 获取模型相关信息：通过`model_name`函数获取模型名称，通过`model_layer_configs`函数获取模型每一层的配置信息。

#### 2 训练文件

训练文件主要定义了训练相关的功能，包括数据加载、模型训练等。

##### 2.1 训练主文件

- **train_dist.py**：该文件主要负责分布式训练相关的功能。

一个完整的示例如下：

```python
def train(args):
    # 初始化分布式训练环境
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()

    config = get_mymodel_config(args)
    model = mymodel_model_hp(config, args)

    # 创建数据集
    if local_rank == 0:
        print("Creating Dataset...")
    
    # 设置数据集相关参数    
    set_megatron_args_for_dataset(args, model, 
                                 model.sp_groups_whole[0] if args.vocab_sp else model.tp_groups_whole[0], 
                                 model.dp_groups_whole[0])
    if local_rank == 0:
        _print_args("arguments", args)

    # 获取数据迭代器
    train_data_iterator, valid_data_iterator, test_data_iterator = get_train_valid_test_data_iterators()
    
    # 创建优化器和学习率调度器
    optimizer, opt_param_scheduler = get_optimizer_and_param_scheduler(model, args)

    # 设置性能分析器
    path = os.path.dirname(os.path.abspath(__file__))
    profiler = GalvatronProfiler(args)
    profiler.set_profiler_dist(path, model_layer_configs(config), model_name(config), start_iter=0)
    
    # 记录模型创建后的内存使用情况
    profiler.profile_memory(0, "After creating model")
    if local_rank == 0:
        print("Start training...")

    # 训练循环
    for iter in range(args.iteration, args.train_iters):
        # 获取一个批次的数据
        tokens, kwargs, loss_func = get_batch(train_data_iterator)
        
        # 记录开始时间和内存使用
        profiler.profile_time_start(iter)
        profiler.profile_memory(iter, "Before Forward")

        # 准备输入数据
        input_ids = tokens
        batch = [input_ids]
        
        # 前向传播和反向传播
        loss = model.forward_backward(batch, iter, profiler, 
                                      loss_func=loss_func,
                                      **kwargs)
        
        # 记录反向传播后的内存使用
        profiler.profile_memory(iter, "After Backward")
        
        # 梯度裁剪
        total_norm = clip_grad_norm(model, args.clip_grad)
        
        # 优化器步骤
        optimizer.step()
        # 学习率调度器步骤
        opt_param_scheduler.step(increment=args.global_batch_size)
        
        # 记录优化器步骤后的内存使用
        profiler.profile_memory(iter, "After optimizer_step")
        
        # 清零梯度
        optimizer.zero_grad()

        # 更新性能统计信息
        profiler.post_profile_memory(iter)
        # 获取当前学习率
        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']
        # 记录本次迭代的性能指标
        profiler.profile_time_end(iter, loss, learning_rate, total_norm)
        
        # 同步所有进程
        torch.distributed.barrier()

        # 定期保存模型检查点
        if args.save != None and (iter + 1) % args.save_interval == 0:
            save_llama_module(args.save, model, optimizer, opt_param_scheduler, iter + 1, args)

if __name__ == '__main__':
    # 初始化Galvatron训练环境
    args = initialize_galvatron(model_args, mode='train_dist')
    # 设置随机种子以确保可重复性
    set_seed()
    # 开始训练
    train(args)
```

- **train_dist_random.py**：该文件主要负责分布式训练相关的功能，与`train_dist.py`类似，但使用随机数据进行训练。

##### 2.2 数据加载文件

- **dataloader.py**：该文件主要负责数据加载相关的功能，其主要包括两部分：
    - 随机数据加载：创建生成随机token的dataset，并创建collate_fn函数，将随机token转换为模型输入。
    如下是一个随机数据加载的示例：
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
        # 将batch中的数据堆叠，并返回对应格式的数据
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
            # 随机生成每个样本的实际长度（1到最大长度之间）
            self.data_length = np.random.randint(1,self.sentence_length+1,(self.dataset_size,))
            self.device = device

            # 生成随机输入数据
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

    具体的trainloader由以下代码创建：
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

    其中`distributed_dataloader`函数是Galvatron提供的分布式数据加载器，用于创建分布式数据加载器。

    - 真实数据加载：创建真实数据加载器，并设计loss计算函数。

    真实数据加载的实现基于Megatron dataset，主要包含`train_valid_test_datasets_provider`、`get_train_valid_test_data_iterators`、`get_batch`、`loss_func`等函数。一个具体实现的例子可以参考[gpt_hf](https://github.com/PKU-DAIR/Hetu-Galvatron/blob/main/galvatron/models/gpt_hf/dataloader.py)。

    主要注意的是，`get_batch`函数返回一个tuple，tuple中包含三个元素，分别是：
    - 输入数据：通常是一个token序列，torch.Tensor类型。
    - 其他输入数据：通常是字典类型，包含position_ids、attention_mask、labels等。
    - loss计算函数：通过调用`loss_func(output_tensor)`函数可以直接计算出loss。

    注意：这里的输入数据要和`MyModelModel_sequential.py`文件中Embedding层的输入数据格式保持一致。而其他数据则作为`**kwargs`在模型层之间传递。

##### 2.3 性能分析文件

- **profiler.py**：该文件主要负责性能分析相关的功能，其内容如下：

```python
if __name__ == '__main__':
    # 初始化Galvatron性能分析环境
    args = initialize_galvatron(model_args, mode='profile')
    
    # 加载模型配置
    config = get_mymodel_config(args, overwrite_args=False)
    
    # 创建性能分析器实例
    profiler = GalvatronProfiler(args)
    
    # 获取当前文件的目录路径
    path = os.path.dirname(os.path.abspath(__file__))
    
    # 设置性能分析器启动器
    profiler.set_profiler_launcher(path, layernum_arg_names(), model_name(config))
    
    # 启动性能分析脚本
    profiler.launch_profiling_scripts()
    
    # 处理收集到的性能数据
    profiler.process_profiled_data()
```
##### 2.4 策略搜索文件

- **search_dist.py**：该文件主要负责策略搜索相关的功能，其内容如下：

```python
if __name__ == '__main__':
    args = initialize_galvatron(model_args, mode='search')
    config = get_mymodel_config(args, overwrite_args=True)
    path = os.path.dirname(os.path.abspath(__file__))
    print(args)
    print(config)
    # 创建策略搜索引擎实例
    search_engine = GalvatronSearchEngine(args)
    
    # 设置搜索引擎的基本信息
    search_engine.set_search_engine_info(path, model_layer_configs(config), model_name(config))
    
    # 初始化搜索引擎
    search_engine.initialize_search_engine()

    # 进行策略搜索
    search_engine.parallelism_optimization()
```

#### 3 脚本文件

scripst文件夹中主要包含一些脚本文件，用于实现模型训练、性能分析、策略搜索等功能。

主要包含五种不同的脚本：
- profile_computation.sh：用于性能分析，计算模型在不同配置下的计算性能。
- profile_memory.sh：用于性能分析，计算模型在不同配置下的内存使用情况。
- search_dist.sh：用于策略搜索，搜索模型在不同配置下的最优策略。
- train_dist.sh：用于模型训练，训练模型。
- train_dist_random.sh：用于模型训练，使用随机数据训练模型。
