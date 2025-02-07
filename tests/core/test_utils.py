# tests/core/test_utils.py
import pytest
import torch
import torch.nn as nn
from galvatron.core.runtime.utils import rgetattr, rsetattr, rhasattr

class DummyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.sub = nn.Linear(10, 10)
        self.sub.weight.data.fill_(1.0)

@pytest.fixture
def dummy_module():
    return DummyModule()

def test_rgetattr(dummy_module):
    # Test basic attribute access
    assert isinstance(rgetattr(dummy_module, "sub"), nn.Linear)
    
    # Test nested attribute access
    weight = rgetattr(dummy_module, "sub.weight")
    assert isinstance(weight, torch.Tensor)
    assert torch.all(weight == 1.0)

def test_rsetattr(dummy_module):
    # Test setting nested attribute
    new_weight = nn.Parameter(torch.zeros(10, 10))
    rsetattr(dummy_module, "sub.weight", new_weight)
    assert torch.all(dummy_module.sub.weight == 0.0)

def test_rhasattr(dummy_module):
    # Test existing attributes
    assert rhasattr(dummy_module, "sub")
    assert rhasattr(dummy_module, "sub.weight")
    assert rhasattr(dummy_module, "sub.weight.data")
    
    # Test non-existing attributes
    assert not rhasattr(dummy_module, "nonexistent")
    assert not rhasattr(dummy_module, "sub.nonexistent")
    assert not rhasattr(dummy_module, "sub.weight.nonexistent")