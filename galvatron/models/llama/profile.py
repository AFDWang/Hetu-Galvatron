from galvatron.core import GalvatronProfiler, initialize_galvatron
from galvatron.models.llama.arguments import model_args, layernum_arg_names
from galvatron.models.llama.meta_configs import config_from_meta, llama_config_to_gpt2_config, set_model_config, model_name
import os

if __name__ == '__main__':
    args = initialize_galvatron(model_args, mode='profile')
    llama_config = config_from_meta(args.model_size)
    config = llama_config_to_gpt2_config(llama_config)
    config = set_model_config(config, args, overwrite_args=False)
    
    profiler = GalvatronProfiler(args)
    path = os.path.dirname(os.path.abspath(__file__))
    profiler.set_profiler_launcher(path, layernum_arg_names(), model_name(config))
    
    profiler.launch_profiling_scripts()
    profiler.process_profiled_data()