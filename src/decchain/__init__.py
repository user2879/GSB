"""Deterministic fixed-point per-request decoding sampling chain."""

__version__ = "1.0.0"

from .config import DecodeConfig, FIXED_WIDTHS, QUANT_TIERS  # noqa: F401
from .models import Request, CandidateSequence, Step  # noqa: F401

try:  # pipeline is added later in the build; keep package importable meanwhile
    from .pipeline import DecodePipeline  # noqa: F401
except Exception:  # pragma: no cover - development convenience
    DecodePipeline = None  # type: ignore

__all__ = [
    "DecodeConfig",
    "FIXED_WIDTHS",
    "QUANT_TIERS",
    "DecodePipeline",
    "Request",
    "CandidateSequence",
    "Step",
]
