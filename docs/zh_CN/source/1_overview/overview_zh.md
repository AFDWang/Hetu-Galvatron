# 概述

Galvatron 是一个为 Transformer 模型（包括大语言模型 LLMs）设计的自动分布式训练系统。它利用先进的自动并行技术提供卓越的训练效率。本仓库包含了 Galvatron-2 的官方实现，这是我们最新版本，增加了多项新特性。

## 主要特点
### (1) 通过自动并行提升效率

#### 扩展的并行搜索空间
整合了分布式训练中多个流行的并行维度，包括 DP（数据并行）、SDP（分片数据并行，支持 ZeRO-2 和 ZeRO-3）、PP（流水线并行，支持 GPipe 和 Pipedream-flush / 1F1B-flush）、TP（张量并行）、SP（序列并行，支持 Megatron-SP 和 Deepspeed-Ulysses）。同时将 CKPT（激活检查点）作为一个特殊的并行维度。

#### 细粒度混合并行
对每个 Transformer 层，支持灵活和细粒度的混合并行策略，有助于提高训练效率。

#### 高效的自动并行优化
对于任何给定的 Transformer 模型，自动且高效地搜索最优并行策略，提供最佳训练效率。

### (2) 通用性
适用于广泛的 Transformer 架构，包括语言模型、大语言模型、视觉模型、多模态模型等。

### (3) 用户友好界面
易于使用，即使对分布式训练不熟悉的用户也能轻松上手。

## 系统架构
Galvatron 由四个模块组成，包括自动 Galvatron 性能分析器、策略代价估计器、提供并行优化的 Galvatron 搜索引擎，以及 Galvatron 运行时框架。使用 Galvatron 在多个 GPU 上通过自动并行训练 Transformer 模型时，用户只需要提供硬件环境和 Transformer 模型配置即可。

<div align=center> <img src="../_static/api.jpg" width="800" /> </div>