"""可集成的语音情感与对话理解定点校准组件。"""
from .config import CalibConfig, load_config
from .errors import ErrorCode, ErrorCollector
from .fixedpoint import FixedPoint
from .pipeline import LoadedInputs, PipelineResult, load_inputs, run_pipeline
from .report import build_report, render_markdown, write_report

__all__ = [
    "CalibConfig",
    "ErrorCode",
    "ErrorCollector",
    "FixedPoint",
    "LoadedInputs",
    "PipelineResult",
    "load_config",
    "load_inputs",
    "run_pipeline",
    "build_report",
    "write_report",
    "render_markdown",
]
