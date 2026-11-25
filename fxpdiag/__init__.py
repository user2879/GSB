"""逐层激活定点化诊断管线（fixed-point diagnostic pipeline）。"""

from .pipeline import PipelineRunner, PipelineState

__all__ = ["PipelineState", "PipelineRunner"]
__version__ = "1.0.0"
