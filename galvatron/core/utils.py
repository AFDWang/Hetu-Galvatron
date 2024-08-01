from functools import reduce
from torch.nn import ModuleList
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch import distributed as dist

# utility functions, support on nested attributes for getattr, setattr, and setattr
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
# https://stackoverflow.com/questions/24779483/hasattr-for-nested-attributes
def rgetattr(obj, attr):
    def _getattr_fsdp(obj, attr):
        if isinstance(obj, FSDP):
            return getattr(obj._fsdp_wrapped_module, attr)
        else:
            return getattr(obj, attr)
    return reduce(_getattr_fsdp, [obj] + attr.split('.'))

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rhasattr(obj, attr):
    try:
        rgetattr(obj, attr)
        return True
    except AttributeError:
        return False