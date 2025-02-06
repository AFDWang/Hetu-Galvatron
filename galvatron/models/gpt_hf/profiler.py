import os

from galvatron.core import ModelProfiler, initialize_galvatron
from galvatron.models.gpt_hf.arguments import layernum_arg_names, model_args
from galvatron.models.gpt_hf.GPTModel_hybrid_parallel import get_gpt_config
from galvatron.models.gpt_hf.meta_configs import model_name

if __name__ == "__main__":
    args = initialize_galvatron(model_args, mode="profile")
    config = get_gpt_config(args, overwrite_args=False)

    profiler = ModelProfiler(args)
    path = os.path.dirname(os.path.abspath(__file__))
    profiler.set_profiler_launcher(path, layernum_arg_names(), model_name(config))

    profiler.launch_profiling_scripts()
    profiler.process_profiled_data()
