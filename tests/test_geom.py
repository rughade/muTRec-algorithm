"""
Tests for geometry calculations (MuTRec, scattering angles, etc.).
"""
import pytest
import jax.numpy as jnp
import numpy as np
from Python.muTRec import (
    _angles_xy,
    scattering_angle_signal,
    mutrec_reshma,
    mutrec_reshma_batch,
)
from Python.PoCA import poca_reshma

def test_angles_xy_basic(sample_points):
    """Test basic angle calculation between two points."""
    p1, p2, p3, p4 = sample_points
    theta_x, theta_y = _angles_xy(p1, p2)
    
    # Check that angles are floats
    assert isinstance(theta_x, jnp.ndarray)
    assert isinstance(theta_y, jnp.ndarray)
    
    # For our test data, angles should be non-zero
    assert theta_x != 0.0
    assert theta_y != 0.0


def test_angles_xy_vertical():
    """Test angle calculation for vertical trajectory."""# p1 is now the lower point
    p1 = jnp.array([1000.0, 1000.0, 1000.0])
    p2 = jnp.array([1000.0, 1000.0, 2000.0])
    theta_x, theta_y = _angles_xy(p1, p2)
    
    # Vertical trajectory should have zero angles
    assert jnp.abs(theta_x) < 1e-10
    assert jnp.abs(theta_y) < 1e-10


def test_angles_xy_zero_dz():
    """Test that zero dz is handled without division by zero."""
    p1 = jnp.array([1000.0, 1000.0, 2000.0])
    p2 = jnp.array([1100.0, 1100.0, 2000.0])
    theta_x, theta_y = _angles_xy(p1, p2)
    
    # Should not raise an error and should return valid values
    assert jnp.isfinite(theta_x)
    assert jnp.isfinite(theta_y)


def test_scattering_angle_signal(sample_points):
    """Test scattering angle signal calculation."""
    p1, p2, p3, p4 = sample_points
    s = scattering_angle_signal(p1, p2, p3, p4)
    
    # Check that signal is non-negative
    assert s >= 0.0
    assert jnp.isfinite(s)


def test_scattering_angle_signal_batch(batch_points):
    """Test scattering angle signal for batch of muons."""
    p1, p2, p3, p4 = batch_points
    s = scattering_angle_signal(p1, p2, p3, p4)
    
    # Check shape
    assert s.shape == (p1.shape[0],)
    
    # Check all signals are non-negative and finite
    assert jnp.all(s >= 0.0)
    assert jnp.all(jnp.isfinite(s))


def test_mutrec_single_output_shape(sample_points, default_params):
    """Test that MuTRec returns correct output shapes."""
    p1, p2, p3, p4 = sample_points
    x, y, z, theta_x, theta_y = mutrec_reshma(p1, p2, p3, p4, default_params)
    
    T = int(round(default_params.cube_size / default_params.voxel_size))
    
    # Check shapes
    assert x.shape == (T,)
    assert y.shape == (T,)
    assert z.shape == (T,)
    assert theta_x.shape == (T,)
    assert theta_y.shape == (T,)


def test_mutrec_single_finite(sample_points, default_params):
    """Test that MuTRec returns finite values."""
    p1, p2, p3, p4 = sample_points
    x, y, z, theta_x, theta_y = mutrec_reshma(p1, p2, p3, p4, default_params)
    
    # Check all values are finite
    assert jnp.all(jnp.isfinite(x))
    assert jnp.all(jnp.isfinite(y))
    assert jnp.all(jnp.isfinite(z))
    assert jnp.all(jnp.isfinite(theta_x))
    assert jnp.all(jnp.isfinite(theta_y))


def test_mutrec_batch_output_shape(batch_points, default_params):
    """Test that batched MuTRec returns correct output shapes."""
    p1, p2, p3, p4 = batch_points
    B = p1.shape[0]
    T = int(round(default_params.cube_size / default_params.voxel_size))
    
    x, y, z, theta_x, theta_y = mutrec_reshma_batch(p1, p2, p3, p4, default_params)
    
    # Check shapes
    assert x.shape == (B, T)
    assert y.shape == (B, T)
    assert z.shape == (T,)  # z is collapsed
    assert theta_x.shape == (B, T)
    assert theta_y.shape == (B, T)


def test_poca_basic(sample_points):
    """Test POCA calculation."""
    p1, p2, p3, p4 = sample_points
    poca = poca_reshma(p1, p2, p3, p4)
    
    # Check output shape
    assert poca.shape == (3,)
    
    # Check finite values
    assert jnp.all(jnp.isfinite(poca))


def test_poca_batch(batch_points):
    """Test POCA for batch of trajectories."""
    p1, p2, p3, p4 = batch_points
    
    # POCA is not vectorized by default, so we map it
    from jax import vmap
    poca_vmap = vmap(poca_reshma)
    poca = poca_vmap(p1, p2, p3, p4)
    
    # Check shape
    assert poca.shape == (p1.shape[0], 3)
    assert jnp.all(jnp.isfinite(poca))


def test_mutrec_z_values(sample_points, default_params):
    """Test that z values are correct."""
    p1, p2, p3, p4 = sample_points
    _, _, z, _, _ = mutrec_reshma(p1, p2, p3, p4, default_params)
    
    # Check z range
    start_z = default_params.start_point[2]
    expected_min = start_z - 1e-3
    expected_max = expected_min + default_params.cube_size
    
    assert jnp.all(z >= expected_min)
    assert jnp.all(z <= expected_max)

