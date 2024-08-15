from .LlamaModel_hybrid_parallel import get_hybrid_parallel_configs, construct_hybrid_parallel_model, llama_model_hp

from transformers.models.llama.modeling_llama import LlamaRMSNorm
import torch

def rms_reset_parameters(self):
    with torch.no_grad():
        torch.nn.init.ones_(self.weight)
LlamaRMSNorm.reset_parameters = rms_reset_parameters