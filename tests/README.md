# Tests for mutrec-py

This directory contains the test suite for the mutrec-py project using pytest.

## Test Structure

- `conftest.py` - Pytest configuration and shared fixtures
- `test_params.py` - Tests for ReconParams dataclass
- `test_geom.py` - Tests for geometry calculations (GMTE, scattering angles)

## Running Tests

### Run all tests
```bash
pytest
```

### Run specific test file
```bash
pytest tests/test_geom.py
```

### Run specific test function
```bash
pytest tests/test_geom.py::test_angles_xy_basic
```

### Run with verbose output
```bash
pytest -v
```

### Run with coverage report
```bash
pytest --cov=mutrec --cov-report=html
```

### Run tests in parallel (requires pytest-xdist)
```bash
pytest -n auto
```

## Fixtures

Common fixtures are defined in `conftest.py`:

- `default_params` - Default ReconParams for testing
- `small_params` - Smaller ReconParams for faster tests
- `sample_hit_data` - Sample muon hit data (10 muons)
- `sample_csv_file` - Temporary CSV file with sample data
- `sample_points` - Single set of 4 3D points (p1-p4)
- `batch_points` - Batch of 5 sets of 3D points

## Test Categories

Tests are organized by module:

1. **Parameter Tests** (`test_params.py`)
   - Default parameter values
   - Custom parameter creation
   - Immutability (frozen dataclass)

2. **Geometry Tests** (`test_geom.py`)
   - Angle calculations
   - Scattering angle signals
   - GMTE (single and batch)
   - POCA calculations

## Adding New Tests

When adding new tests:

1. Follow the naming convention `test_*.py` for test files
2. Name test functions as `test_<functionality>`
3. Use appropriate fixtures from `conftest.py`
4. Add new fixtures to `conftest.py` if needed
5. Keep tests focused and independent
6. Use parametrize for testing multiple inputs
7. Mock external dependencies (file I/O, plotting)

## Dependencies

Required packages for testing:
- pytest
- pytest-cov (for coverage)
- pytest-xdist (for parallel execution)
- jax
- numpy
- matplotlib

Install test dependencies:
```bash
pip install pytest pytest-cov pytest-xdist
```
