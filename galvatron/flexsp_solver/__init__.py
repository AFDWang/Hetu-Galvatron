import os
import sys
for p in ['build/lib']:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), p))
from .solver import flexSPCostModel, flexSPOptimizer
# from .sequence_module_py import Sequence, SeqBucket
from sequence_module import Sequence, SeqBucket