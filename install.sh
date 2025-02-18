#!/bin/bash
pip install -r requirements.txt
pip install ninja
MAX_JOBS=128 pip install flash-attn
pip install -e .