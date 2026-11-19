"""Deterministic offline volume rendering with geometric block invalidation."""

from .errors import PipelineError
from .pipeline import RenderResult, render_task

__all__ = ["PipelineError", "RenderResult", "render_task"]

