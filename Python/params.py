from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ReconParams:
    """Sample parameters for muTRec reconstruction."""

    # Geometry (mm)
    voxel_size: float = 50.0
    cube_size: float = 4000.0
    start_point: Tuple[float, float, float] = (1000.0, 1000.0, 2000.0)

    # Physics
    X0: float = 6.2  # radiation length (mm)
    p_muon: float = 5000.0  # MeV
    DE: float = 0.5  # energy-loss constant (same as MATLAB)
    const1: float = 13.6 ** 2  # (MeV)^2, multiple scattering

    # CSV
    skip_rows: int = 21
    skip_columns: int = 0

    # Execution
    chunk_size: int = 65536  # tune for your RAM/GPU memory

    # Animation Plotting
    plot_interval: int = 2  # plot every N batches during reconstruction