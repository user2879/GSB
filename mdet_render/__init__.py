"""Deterministic fixed-point multimodal condition renderer."""

from .config import RenderConfig, load_render_config
from .errors import BusinessError
from .fixed import FixedPoint
from .formats import (
    NOISE_CODE_TO_NAME,
    NOISE_NAME_TO_CODE,
    StageRecord,
    parse_v1,
    parse_v2,
    to_v1_lines,
    to_v2_document,
    v1_to_v2,
    v2_to_v1,
)
from .lease import LocalExecutor, ManualClock
from .renderer import Renderer, RenderResult
from .samples import create_sample_workspace
from .workspace import Workspace, load_commands

__all__ = [
    "BusinessError",
    "FixedPoint",
    "LocalExecutor",
    "ManualClock",
    "NOISE_CODE_TO_NAME",
    "NOISE_NAME_TO_CODE",
    "RenderConfig",
    "RenderResult",
    "Renderer",
    "StageRecord",
    "Workspace",
    "create_sample_workspace",
    "load_commands",
    "load_render_config",
    "parse_v1",
    "parse_v2",
    "to_v1_lines",
    "to_v2_document",
    "v1_to_v2",
    "v2_to_v1",
]
__version__ = "0.1.0"
