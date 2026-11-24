"""Deterministic tunnel-fragment interoperability component."""

from .fragmenter import (
    FragmentBundle,
    FragmentRecord,
    MessageBundle,
    ReassemblyResult,
    build_bundle,
    fragment_message,
    reassemble_fragments,
)
from .errors import (
    E_FIELD,
    E_FORMAT,
    E_REF,
    E_REPLAY,
    E_SNAPSHOT,
    E_SPEC,
    E_TIMEOUT,
    TunnelError,
)
from .invalidation import InvalidationResult, evolve_bundle
from .ledger import OpenResult, ReplayResult, SessionLedger
from .samples import MessagePlan, generate_messages, generate_payload
from .scheduler import ScheduledTask, SchedulerResult, run_scheduler
from .spec import TunnelSpec, load_spec, load_spec_mapping, validate_versions_monotonic
from .wire import (
    NormalizedFragment,
    WireContext,
    decode_fragment,
    encode_fragment,
    migrate_fragment,
)

__version__ = "0.1.0"

__all__ = [
    "E_FIELD",
    "E_FORMAT",
    "E_REF",
    "E_REPLAY",
    "E_SNAPSHOT",
    "E_SPEC",
    "E_TIMEOUT",
    "TunnelError",
    "TunnelSpec",
    "FragmentBundle",
    "FragmentRecord",
    "MessageBundle",
    "MessagePlan",
    "NormalizedFragment",
    "OpenResult",
    "ReassemblyResult",
    "ReplayResult",
    "ScheduledTask",
    "SchedulerResult",
    "SessionLedger",
    "WireContext",
    "InvalidationResult",
    "__version__",
    "build_bundle",
    "decode_fragment",
    "encode_fragment",
    "evolve_bundle",
    "fragment_message",
    "generate_messages",
    "generate_payload",
    "migrate_fragment",
    "load_spec",
    "load_spec_mapping",
    "reassemble_fragments",
    "run_scheduler",
    "validate_versions_monotonic",
]
