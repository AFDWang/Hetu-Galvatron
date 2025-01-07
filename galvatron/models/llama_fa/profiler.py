from galvatron.core import GalvatronProfiler, initialize_galvatron
from galvatron.models.llama_fa.arguments import model_args, layernum_arg_names
from galvatron.models.llama_fa.meta_configs import model_name
from galvatron.models.llama_fa.LlamaModel_hybrid_parallel import get_llama_config
import os

if __name__ == '__main__':
    args = initialize_galvatron(model_args, mode='profile')
    config = get_llama_config(args, overwrite_args=False)
    
    profiler = GalvatronProfiler(args)
    path = os.path.dirname(os.path.abspath(__file__))
    profiler.set_profiler_launcher(path, layernum_arg_names(), model_name(config))
    
    profiler.launch_profiling_scripts()
    profiler.process_profiled_data()