# Overview

Galvatron is an automatic distributed training system designed for Transformer models, including Large Language Models (LLMs). It leverages advanced automatic parallelism techniques to deliver exceptional training efficiency. This repository houses the official implementation of Galvatron-2, our latest version enriched with several new features.

## Key Features
### (1) Enhanced Efficiency via Automatic Parallelism

#### Enlarged Parallelism Search Space
Incorporate multiple popular parallelism dimensions of distributed training, including DP (Data Parallelism), SDP (Sharded Data Parallelism, support both ZeRO-2 & ZeRO-3), PP (Pipeline Parallelism, support both GPipe & Pipedream-flush / 1F1B-flush), TP (Tensor Parallelism). Also incorporate CKPT (Activation Checkpointing) as a special parallelism dimension.

#### Fine-grained Hybrid Parallelism
For each Transformer layer, support flexible and fine-grained hybrid parallelism strategies, contributing to the enhanced training efficiency.

#### Efficient Automatic Parallelism Optimization
For any given Transformer model, automatically and efficiently search for the optimal parallelism strategies, which provides the optimal training efficiency.

### (2) Versatility
Suitable for a wide range of Transformer architectures, including language models, LLMs, vision models, multi-modality models, etc.

### (3) User-Friendly Interface
Easy to use, even for those new to distributed training.

## System Architecture
Galvatron is consisted of four modules, including an automatic Galvatron Profiler, a strategy cost estimator, Galvatron Search Engine that provides parallelism optimization, and Galvatron runtime framework. To train Transformer models over multiple GPUs using automatic parallelism with Galvatron, users only need to provide with hardware environment and the Transformer model configuration.

<div align=center> <img src="../_static/api.jpg" width="800" /> </div>
