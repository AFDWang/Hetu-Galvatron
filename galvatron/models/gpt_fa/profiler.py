from galvatron.core import GalvatronProfiler, initialize_galvatron
from galvatron.models.gpt_fa.arguments import model_args, layernum_arg_names
from galvatron.models.gpt_fa.GPTModel_hybrid_parallel import get_gpt_config
from galvatron.models.gpt_fa.meta_configs import model_name
import os

if __name__ == '__main__':
    args = initialize_galvatron(model_args, mode='profile')
    config = get_gpt_config(args, overwrite_args=False)
    
    profiler = GalvatronProfiler(args)
    path = os.path.dirname(os.path.abspath(__file__))
    profiler.set_profiler_launcher(path, layernum_arg_names(), model_name(config))
    
    profiler.launch_profiling_scripts()
    profiler.process_profiled_data()