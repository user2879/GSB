from __future__ import annotations

from pathlib import Path

from spectral_workbench.config import CubeConfig, DriftConfig, PipelineConfig, UnmixConfig


def small_cube_config(seed: int = 17, drift: float = 0.012) -> CubeConfig:
    return CubeConfig(
        seed=seed,
        frames=8,
        height=32,
        width=32,
        blob_sigma_px=2.2,
        drift_dx_per_frame=drift,
        drift_dy_per_frame=-drift * 2.0 / 3.0,
    )


def small_pipeline_config(*, compensate: bool = True, tolerance: float = 0.08) -> PipelineConfig:
    return PipelineConfig(
        unmix=UnmixConfig(cond_warn=1.0e9),
        drift=DriftConfig(max_shift_px=2.0, compensate=compensate),
        concentration_tolerance=tolerance,
    )
