"""Deterministic configuration and numerical constants.

All reference arithmetic is executed in IEEE-754 float64. The constants below
describe the *equivalent float32* rounding error that a real GPU pipeline would
introduce; the reference keeps float64 precision but budgets the float32 error so
that its decisions stay conservative relative to a GPU implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

# Fixed seeds (single source of truth; never derived from wall-clock time).
HEIGHT_SEED = 20261119
MATERIAL_SEED = 30261119
TRACK_SEED = 40261119

# Heightfield geometry.
WORLD_EXTENT = 4096.0          # world metres covered by the full heightfield
MAX_ELEVATION = 600.0          # uint16 max code maps to this world height
UINT16_MAX = 65535

# Quadtree.
LEAF_SAMPLES = 64              # leaf block edge in height samples (1 sample overlap)
LEAF_OVERLAP = 1

# Numerical precision budgets.
F64_EPS = 2.220446049250313e-16
F32_EPS = 1.1920928955078125e-07
# A chain of ~64 float32 fused matrix/vector ops rounded conservatively.
F32_OP_CHAIN = 64
F32_REL_EPS = F32_EPS * F32_OP_CHAIN
# Absolute float64 floor used for near-zero quantities.
ABS_FLOOR = 1.0e-12

# Depth / Hi-Z defaults.
DEPTH_WIDTH = 192
DEPTH_HEIGHT = 128
RAYMARCH_STEPS = 160
MAX_DEPTH_BIAS = 0.05          # 5% of view depth is the hard safety limit
FRUSTUM_NEAR = 1.0
FRUSTUM_FAR = 12000.0
VERTICAL_FOV_RAD = 1.0471975511965976  # 60 degrees

# Error budget per occlusion decision, expressed as relative view-depth.
DEFAULT_DECISION_BUDGET = 0.02
# Screen-space projected AABB error budget in pixels.
DEFAULT_SCREEN_ERROR_BUDGET_PX = 8.0


@dataclass(frozen=True)
class CullingConfig:
    depth_width: int = DEPTH_WIDTH
    depth_height: int = DEPTH_HEIGHT
    raymarch_steps: int = RAYMARCH_STEPS
    max_hiz_levels: int = 6
    decision_budget: float = DEFAULT_DECISION_BUDGET
    screen_error_budget_px: float = DEFAULT_SCREEN_ERROR_BUDGET_PX
    max_depth_bias: float = MAX_DEPTH_BIAS
    near: float = FRUSTUM_NEAR
    far: float = FRUSTUM_FAR
    fov_rad: float = VERTICAL_FOV_RAD
    # Memory budgets in bytes (per-block payload bookkeeping).
    memory_budget_bytes: int = 256 * 1024 * 1024
    vram_budget_bytes: int = 192 * 1024 * 1024
    # Preload ring: blocks within this many leaf cells beyond the frustum stay resident.
    preload_rings: int = 2
    degradation_cap: int = 3

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = CullingConfig()
