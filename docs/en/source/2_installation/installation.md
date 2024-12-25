# Installation

## System Requirements
- Python >= 3.8
- Pytorch >= 2.1
- Linux OS

## Preparations

It is recommended to create a Python 3.8 virtual environment using conda. The command is as follows:
```shell
conda create -n galvatron python=3.8
conda activate galvatron
```

First, based on the CUDA version in your system environment, find the specific installation command for torch on the [PyTorch official website](https://pytorch.org/get-started/previous-versions/).
```shell
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
```

Next, install [apex](https://github.com/NVIDIA/apex) from source code:
```shell
git clone https://github.com/NVIDIA/apex
cd apex
# if pip >= 23.1 (ref: https://pip.pypa.io/en/stable/news/#v23-1) which supports multiple `--config-settings` with the same key... 
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
# otherwise
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Install Galvatron
### Installation from PyPI

You can install Galvatron from PyPI by running the following command:

``` shell
pip install hetu-galvatron
```

### Installation from Source Code

To install the latest version of Galvatron from the source code, run the following commands:

``` shell
git clone https://github.com/PKU-DAIR/Hetu-Galvatron.git
cd Hetu-Galvatron
pip install .
```

To use FlashAttention-2 features in Galvatron-2, you can either:
- Install the [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) manually and then ```pip install hetu-galvatron```.
- Alternatively, you can install Galvatron-2 with FlashAttention-2 as follows:

    1. Make sure that PyTorch, `packaging` (`pip install packaging`), `ninja` is installed.
    2. Install Galvatron with FlashAttention-2:
    ```sh
    GALVATRON_FLASH_ATTN_INSTALL=TRUE pip install hetu-galvatron
    ```
