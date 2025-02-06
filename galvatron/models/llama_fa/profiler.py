import os

from galvatron.core import ModelProfiler, initialize_galvatron
from galvatron.models.llama_fa.arguments import layernum_arg_names, model_args
from galvatron.models.llama_fa.LlamaModel_hybrid_parallel import get_llama_config
from galvatron.models.llama_fa.meta_configs import model_name

if __name__ == "__main__":
    args = initialize_galvatron(model_args, mode="profile")
    config = get_llama_config(args, overwrite_args=False)

    profiler = ModelProfiler(args)
    path = os.path.dirname(os.path.abspath(__file__))
    profiler.set_profiler_launcher(path, layernum_arg_names(), model_name(config))

    profiler.launch_profiling_scripts()
    profiler.process_profiled_data()
