from galvatron.core import initialize_galvatron, GalvatronSearchEngine
from galvatron.models.llama_fa.arguments import model_args
from galvatron.models.llama_fa.meta_configs import model_name, model_layer_configs
from galvatron.models.llama_fa.LlamaModel_hybrid_parallel import get_llama_config
import os

if __name__ == '__main__':
    args = initialize_galvatron(model_args, mode='search')
    config = get_llama_config(args)
    path = os.path.dirname(os.path.abspath(__file__))
    print(args)
    print(config)
    
    search_engine = GalvatronSearchEngine(args)
    search_engine.set_search_engine_info(path, model_layer_configs(config), model_name(config))
    # search_engine.set_microbatch_func(microbatch_size=4, max_chunk=8) # Optional
    search_engine.set_model_type('gpt') # Optional
    
    search_engine.initialize_search_engine()
    # search_engine.check_cost_model(bsz=128)
    search_engine.parallelism_optimization()