"""Deterministic vertical ancient-layout recognition engine.

The public API is intentionally small.  Most integration code should use
``RecognitionEngine`` together with ``generate_samples`` for local fixtures.
"""

from .config import DEFAULT_CONFIG, config_hash, normalize_config
from .samples import generate_samples
from .templates import build_templates, template_version

__all__ = [
    "DEFAULT_CONFIG",
    "RecognitionEngine",
    "build_templates",
    "config_hash",
    "generate_samples",
    "normalize_config",
    "template_version",
]

__version__ = "1.0.0"


def __getattr__(name: str):
    if name == "RecognitionEngine":
        from .engine import RecognitionEngine

        return RecognitionEngine
    raise AttributeError(name)
