"""Block storage snapshot & rollback domain component (standard library only).

Public package API covering volume loading, CoW snapshots, three-path
rollback, atomic switch / crash recovery, manifest replacement, auditing and
structured log queries.
"""
from .errors import (
    ALL_CODES,
    DomainError,
    StructuredLogger,
    E_CHAIN,
    E_INPUT,
    E_INTEGRITY,
    E_PATH,
    E_SHAPE,
    E_STATE,
)
from .bitmap import Bitmap
from .manager import (
    DEFAULT_BITMAP_THRESHOLD,
    DEFAULT_CHAIN_SPAN,
    MANIFEST_FORMAT_VERSION,
    SnapshotManager,
    manifest_digest,
)
from .manifest import ManifestService
from .rollback import ALL_PATHS, PATH_BITMAP, PATH_CHAIN, PATH_FULL, RollbackEngine
from .service import BlockSnapService
from .storage import BlockStore, JOB_COMMITTED, JOB_RUNNING, JOB_UNKNOWN, MetadataStore
from .volume import (
    SUPPORTED_BLOCK_SIZES,
    Volume,
    block_digest,
    deterministic_block,
    parse_change_spec,
)

__all__ = [
    "ALL_CODES",
    "ALL_PATHS",
    "DEFAULT_BITMAP_THRESHOLD",
    "DEFAULT_CHAIN_SPAN",
    "DomainError",
    "StructuredLogger",
    "E_CHAIN",
    "E_INPUT",
    "E_INTEGRITY",
    "E_PATH",
    "E_SHAPE",
    "E_STATE",
    "Bitmap",
    "BlockSnapService",
    "BlockStore",
    "JOB_COMMITTED",
    "JOB_RUNNING",
    "JOB_UNKNOWN",
    "ManifestService",
    "MANIFEST_FORMAT_VERSION",
    "MetadataStore",
    "PATH_BITMAP",
    "PATH_CHAIN",
    "PATH_FULL",
    "RollbackEngine",
    "SnapshotManager",
    "SUPPORTED_BLOCK_SIZES",
    "Volume",
    "block_digest",
    "deterministic_block",
    "manifest_digest",
    "parse_change_spec",
]
