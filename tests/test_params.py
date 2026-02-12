"""
Tests for ReconParams dataclass.
"""
import pytest
from Python.params import ReconParams


def test_default_params():
    """Test that default parameters can be instantiated."""
    params = ReconParams()


def test_custom_params():
    """Test that custom parameters can be set."""
    params = ReconParams(
        voxel_size=100.0,
        cube_size=2000.0,
        start_point=(500.0, 500.0, 1000.0),
        chunk_size=1024,
    )
    assert params.voxel_size == 100.0
    assert params.cube_size == 2000.0
    assert params.start_point == (500.0, 500.0, 1000.0)
    assert params.chunk_size == 1024


def test_params_frozen():
    """Test that params are immutable (frozen dataclass)."""
    params = ReconParams()
    with pytest.raises(AttributeError):
        params.voxel_size = 100.0


def test_params_grid_size():
    """Test that grid size calculation makes sense."""
    params = ReconParams(voxel_size=50.0, cube_size=4000.0)
    expected_n = int(round(params.cube_size / params.voxel_size))
    assert expected_n == 80
    
    params2 = ReconParams(voxel_size=100.0, cube_size=1000.0)
    expected_n2 = int(round(params2.cube_size / params2.voxel_size))
    assert expected_n2 == 10

