import pytest
import numpy as np
from tests.utils.search_args import SearchArgs
from galvatron.core.search_engine import GalvatronSearchEngine
from tests.utils.search_configs import initialize_search_engine

@pytest.fixture
def base_engine():
    """Create a base search engine with common settings"""
    args = SearchArgs()
    args.gpu_num = 8
    args.min_bsz = 16
    args.max_bsz = 64
    args.bsz_scale = 8
    args.recommend_min_bsz = False
    engine = GalvatronSearchEngine(args)
    return engine

@pytest.mark.search_engine
def test_settle_bsz(base_engine):
    """Test when settle_bsz is set"""
    base_engine.args.settle_bsz = 20
    base_engine.set_searching_bsz()
    
    assert base_engine.min_bsz == 20
    assert base_engine.max_bsz == 20
    assert base_engine.bsz_scale == 0
    assert base_engine.BSZs == [20]

@pytest.mark.search_engine
def test_normal_bsz_range(base_engine):
    """Test normal batch size range calculation"""
    base_engine.set_searching_bsz()
    
    assert base_engine.min_bsz == 16
    assert base_engine.max_bsz == 64
    assert base_engine.bsz_scale == 8
    assert base_engine.BSZs == [16, 24, 32, 40, 48, 56, 64]

@pytest.mark.search_engine
@pytest.mark.parametrize("min_bsz,max_bsz,bsz_scale,expected_bszs", [
    (20, 50, 10, [20, 30, 40, 50]),  # min_bsz adjusted to nearest multiple
    (15, 45, 15, [15, 30, 45]),      # exact multiples
    (32, 96, 32, [32, 64, 96]),      # larger scale
])
def test_bsz_range_with_different_scales(base_engine, min_bsz, max_bsz, bsz_scale, expected_bszs):
    """Test batch size range with different scales"""
    base_engine.args.min_bsz = min_bsz
    base_engine.args.max_bsz = max_bsz
    base_engine.args.bsz_scale = bsz_scale
    base_engine.set_searching_bsz()
    
    assert base_engine.BSZs == expected_bszs
    assert base_engine.min_bsz == expected_bszs[0]
    assert base_engine.max_bsz == expected_bszs[-1]

@pytest.mark.search_engine
def test_recommend_min_bsz(monkeypatch, base_engine):
    """Test when recommend_min_bsz is enabled"""
    def mock_recommend_min_bsz(bsz_scale):
        return 24
    
    monkeypatch.setattr(base_engine, 'recommend_min_bsz', mock_recommend_min_bsz)
    base_engine.args.recommend_min_bsz = True
    base_engine.set_searching_bsz()
    
    assert base_engine.min_bsz == 24

@pytest.mark.search_engine
def test_max_bsz_adjustment(base_engine):
    """Test maximum batch size adjustment when not divisible by scale"""
    base_engine.args.max_bsz = 50
    base_engine.args.bsz_scale = 16
    base_engine.set_searching_bsz()
    
    expected_max = int(np.ceil(50 / 16) * 16) - 16  # Should round up to 64
    assert base_engine.max_bsz == expected_max

@pytest.mark.search_engine
def test_min_bsz_smaller_than_scale(base_engine):
    """Test when minimum batch size is smaller than scale"""
    base_engine.args.min_bsz = 4
    base_engine.args.bsz_scale = 8
    base_engine.set_searching_bsz()
    
    assert base_engine.min_bsz == 8  # Should be adjusted to bsz_scale

@pytest.mark.search_engine
def test_recommend_min_bsz_negative(monkeypatch, base_engine):
    """Test when recommend_min_bsz returns negative value"""
    def mock_recommend_min_bsz(bsz_scale):
        return -1
    
    monkeypatch.setattr(base_engine, 'recommend_min_bsz', mock_recommend_min_bsz)
    base_engine.args.recommend_min_bsz = True
    base_engine.args.min_bsz = 16
    base_engine.set_searching_bsz()
    
    assert base_engine.min_bsz == 16  # Should keep original min_bsz