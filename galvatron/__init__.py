import os
import sys
import types
import torch
for p in ['site_package', 'build/lib']:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), p))
    
sys.modules['transformer_engine'] = types.ModuleType('transformer_engine')
sys.modules['transformer_engine'].__spec__ = 'te'
setattr(sys.modules['transformer_engine'], 'pytorch', torch.nn)
setattr(sys.modules['transformer_engine'].pytorch, 'LayerNormLinear', torch.nn.Module)
setattr(sys.modules['transformer_engine'].pytorch, 'DotProductAttention', torch.nn.Module)