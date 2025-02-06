import os

import torch
from transformers import T5Config, T5ForConditionalGeneration

from galvatron.core import ModelProfiler, initialize_galvatron
from galvatron.models.T5.arguments import layernum_arg_names, model_args, seqlen_arg_names
from galvatron.models.T5.meta_configs import model_name
from galvatron.models.T5.T5Model_hybrid_parallel import get_t5_config

if __name__ == "__main__":
    args = initialize_galvatron(model_args, mode="profile")
    config = get_t5_config(args, overwrite_args=False)
    profiler = ModelProfiler(args)
    path = os.path.dirname(os.path.abspath(__file__))
    profiler.set_profiler_launcher(path, layernum_arg_names(), model_name(config), seqlen_arg_names())

    profiler.launch_profiling_scripts()
    profiler.process_profiled_data()
