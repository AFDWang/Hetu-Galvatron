from .pipeline import PipelineParallel, PipeSequential
from .sp_grad_reduce import _post_backward_hook_sp
import torch.distributed.fsdp as fsdp
fsdp._runtime_utils._post_backward_hook = _post_backward_hook_sp