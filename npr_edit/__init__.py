"""Deterministic NPR line-art and stylized rendering editor."""

from .engine import EditRecord, NprEngine
from .errors import *  # noqa: F401,F403
from .errors import BUSINESS_CODES
from .formats import (
    inspect_skeleton_binary,
    migrate_project,
    roundtrip_report,
    scene_to_skeleton_binary,
    skeleton_binary_to_project,
    skeleton_binary_to_scene,
)
from .generator import DEFAULT_SEED, generate_scene
from .geometry import flatten_element, flatten_scene
from .model import Element, Layer, Point, Scene, Segment, StrokeSampling, StyleParams
from .persistence import PersistenceStore
from .snapshot import RenderSnapshot, RenderState, SnapshotStore

__all__ = [
    "DEFAULT_SEED",
    "EditRecord",
    "Element",
    "Layer",
    "NPRError",
    "NprEngine",
    "PersistenceStore",
    "Point",
    "RenderSnapshot",
    "RenderState",
    "Scene",
    "Segment",
    "SnapshotStore",
    "StrokeSampling",
    "StyleParams",
    "flatten_element",
    "flatten_scene",
    "generate_scene",
    "inspect_skeleton_binary",
    "migrate_project",
    "roundtrip_report",
    "scene_to_skeleton_binary",
    "skeleton_binary_to_project",
    "skeleton_binary_to_scene",
]
__all__ += sorted(BUSINESS_CODES)

__version__ = "1.0.0"
