"""Cache-segment capacity proofs and read-view convergence diagnostics."""

from .content import content_bytes, entry_digest, iter_content
from .diagnostics import dashboard_snapshot, inject_anomaly, render_text_dashboard
from .errors import DiagnosticError, EXIT_CODES
from .models import CacheSpec, EntrySpec, OperationResult, SegmentSpec, ViewSpec
from .paths import WorkspacePaths
from .spec import load_spec, parse_spec
from .store import CacheProofStore

__all__ = [
    "CacheProofStore",
    "CacheSpec",
    "DiagnosticError",
    "EXIT_CODES",
    "EntrySpec",
    "OperationResult",
    "SegmentSpec",
    "ViewSpec",
    "WorkspacePaths",
    "content_bytes",
    "dashboard_snapshot",
    "entry_digest",
    "inject_anomaly",
    "iter_content",
    "load_spec",
    "parse_spec",
    "render_text_dashboard",
]

__version__ = "0.1.0"

