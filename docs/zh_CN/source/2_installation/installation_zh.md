# 安装

## 系统要求
- Python >= 3.8
- Pytorch >= 2.1
- Linux 操作系统

## 准备工作

建议使用 conda 创建 Python 3.8 虚拟环境。命令如下：
````shell
conda create -n galvatron python=3.8
conda activate galvatron
````


首先，根据系统环境中的 CUDA 版本，在 [PyTorch 官网](https://pytorch.org/get-started/previous-versions/) 找到对应的 torch 安装命令。
````shell
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
````


接下来，从源代码安装 [apex](https://github.com/NVIDIA/apex)：
````shell
git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
````


## 安装 Galvatron
### 从 PyPI 安装

你可以通过运行以下命令从 PyPI 安装 Galvatron：

```` shell
pip install hetu-galvatron
````


### 从源代码安装

要从源代码安装最新版本的 Galvatron，运行以下命令：

```` shell
git clone https://github.com/PKU-DAIR/Hetu-Galvatron.git
cd Hetu-Galvatron
pip install .
````


要在 Galvatron-2 中使用 FlashAttention-2 功能，你可以：
- 手动安装 [FlashAttention-2](https://github.com/Dao-AILab/flash-attention)，然后运行 ```pip install hetu-galvatron```。
- 或者，你可以按照以下步骤安装带有 FlashAttention-2 的 Galvatron-2：

    1. 确保已安装 PyTorch、`packaging`（`pip install packaging`）和 `ninja`。
    2. 安装带有 FlashAttention-2 的 Galvatron：
    ```sh
    GALVATRON_FLASH_ATTN_INSTALL=TRUE pip install hetu-galvatron
    ```
