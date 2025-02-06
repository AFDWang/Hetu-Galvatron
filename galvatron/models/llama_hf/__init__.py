import torch
from megatron.legacy.model.rms_norm import RMSNorm as LlamaRMSNorm

from .LlamaModel_hybrid_parallel import construct_hybrid_parallel_model, get_hybrid_parallel_configs, llama_model_hp


def rms_reset_parameters(self):
    with torch.no_grad():
        torch.nn.init.ones_(self.weight)


LlamaRMSNorm.reset_parameters = rms_reset_parameters
