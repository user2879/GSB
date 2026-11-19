"""中文分词与词性标注领域组件。

对外暴露配置、模型、流水线引擎与稳定的 JSON 数据结构，便于作为库集成。
"""

from .config import ComponentConfig
from .errors import ComponentError, ConfigMismatchError, StageError
from .models import Dictionary, DictEntry, ProbabilityTable, SentenceRecord, Token
from .pipeline import IncrementalPipeline
from .cache import DependencyCache

__all__ = [
    "ComponentConfig",
    "ComponentError",
    "ConfigMismatchError",
    "StageError",
    "Dictionary",
    "DictEntry",
    "ProbabilityTable",
    "SentenceRecord",
    "Token",
    "IncrementalPipeline",
    "DependencyCache",
]

__version__ = "1.0.0"
