"""可集成的 CSR 稀疏矩阵乘（SpMM）增量算子组件。"""

from .config import EngineConfig
from .csr import CSRMatrix, fingerprint_inputs, generate_csr, validate_csr
from .engine import IncrementalEngine, RunResult
from .errors import SpmmError

__all__ = [
    "CSRMatrix",
    "EngineConfig",
    "IncrementalEngine",
    "RunResult",
    "SpmmError",
    "fingerprint_inputs",
    "generate_csr",
    "validate_csr",
]
