"""
Pytest configuration and fixtures for mutrec tests.
"""
import pytest
import numpy as np
import jax.numpy as jnp
from Python.params import ReconParams


@pytest.fixture
def default_params():
    """Provide default reconstruction parameters for testing."""
    return ReconParams(
        voxel_size=50.0,
        cube_size=4000.0,
        start_point=(1000.0, 1000.0, 2000.0),
        X0=6.2,
        p_muon=5000.0,
        DE=0.5,
        skip_rows=21,
        skip_columns=0,
        chunk_size=1024,  # Smaller for testing
        plot_interval=2,
    )


@pytest.fixture
def small_params():
    """Provide small reconstruction parameters for faster testing."""
    return ReconParams(
        voxel_size=100.0,
        cube_size=1000.0,
        start_point=(500.0, 500.0, 1000.0),
        X0=6.2,
        p_muon=5000.0,
        DE=0.5,
        skip_rows=21,
        skip_columns=0,
        chunk_size=512,
        plot_interval=1,
    )


@pytest.fixture
def sample_hit_data():
    """Generate sample hit data for testing (4 points per muon)."""
    # Create a simple muon trajectory
    # p1, p2 (entry), p3, p4 (exit)
    np.random.seed(42)
    n_muons = 10

    data = []
    for i in range(n_muons):
        # Entry points
        p1 = [1000 + np.random.randn() * 10, 1000 + np.random.randn() * 10, 2500]
        p2 = [1100 + np.random.randn() * 10, 1100 + np.random.randn() * 10, 2200]
        # Exit points
        p3 = [1200 + np.random.randn() * 10, 1200 + np.random.randn() * 10, 1800]
        p4 = [1300 + np.random.randn() * 10, 1300 + np.random.randn() * 10, 1500]

        data.append(p1 + p2 + p3 + p4)

    return np.array(data, dtype=np.float64)


@pytest.fixture
def sample_csv_file(tmp_path, sample_hit_data):
    """Create a temporary CSV file with sample hit data."""
    csv_path = tmp_path / "test_hits.csv"

    # Add header lines (21 rows)
    with open(csv_path, 'w') as f:
        for i in range(21):
            f.write(f"# Header line {i}\n")

        # Write data
        np.savetxt(f, sample_hit_data, delimiter=',', fmt='%.6f')

    return str(csv_path)


@pytest.fixture
def sample_points():
    """Generate sample 3D points for geometry tests."""
    p1 = jnp.array([1000.0, 1000.0, 2500.0])
    p2 = jnp.array([1100.0, 1100.0, 2200.0])
    p3 = jnp.array([1200.0, 1200.0, 1800.0])
    p4 = jnp.array([1300.0, 1300.0, 1500.0])
    return p1, p2, p3, p4


@pytest.fixture
def batch_points():
    """Generate batch of 3D points for vectorized tests."""
    np.random.seed(42)
    n = 5
    p1 = jnp.array([[1000 + i * 10, 1000 + i * 10, 2500] for i in range(n)])
    p2 = jnp.array([[1100 + i * 10, 1100 + i * 10, 2200] for i in range(n)])
    p3 = jnp.array([[1200 + i * 10, 1200 + i * 10, 1800] for i in range(n)])
    p4 = jnp.array([[1300 + i * 10, 1300 + i * 10, 1500] for i in range(n)])
    return p1, p2, p3, p4

