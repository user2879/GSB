"""Incremental differentiable soft rasterizer and inverse-imaging demo."""

from .errors import (
    E_CLOSURE_MISMATCH,
    E_DEPENDENCY_STALE,
    E_GRAD_MISMATCH,
    E_SCENE_INVALID,
    E_SNAPSHOT_VERSION,
    E_TILE_HASH,
    DiffRenderError,
)
from .persistence import StateStore
from .pipeline import IncrementalPipeline, ToleranceConfig
from .renderer import gradient_full, render_full
from .scene import Camera, DirectionalLight, Scene, Sphere, default_scene, target_scene_from

__all__ = [
    "Camera",
    "DiffRenderError",
    "DirectionalLight",
    "E_CLOSURE_MISMATCH",
    "E_DEPENDENCY_STALE",
    "E_GRAD_MISMATCH",
    "E_SCENE_INVALID",
    "E_SNAPSHOT_VERSION",
    "E_TILE_HASH",
    "IncrementalPipeline",
    "Scene",
    "Sphere",
    "StateStore",
    "ToleranceConfig",
    "default_scene",
    "gradient_full",
    "render_full",
    "target_scene_from",
]
