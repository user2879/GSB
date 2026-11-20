"""Connection-level gray routing and mirror-sampling arbiter.

The package is intentionally implemented with the Python standard library for
the runtime path.  NumPy is used only by the optional verification helpers.
"""

LEDGER_FORMAT_VERSION = 1
__version__ = "1.0.0"

from .config import load_config, load_request_stream, validate_config
from .engine import ArbiterEngine
from .errors import BusinessError
from .ledger import Ledger, ReplayMismatch
from .model import Config, PoolSpec, RequestSpec, RuleSpec
from .pool import PoolManager
from .stats import build_error_report

__all__ = [
    "ArbiterEngine",
    "BusinessError",
    "Config",
    "Ledger",
    "PoolManager",
    "PoolSpec",
    "ReplayMismatch",
    "RequestSpec",
    "RuleSpec",
    "build_error_report",
    "load_config",
    "load_request_stream",
    "validate_config",
]
