"""Local delay queue and retry middleware (stdlib-only, Windows friendly).

Public package API:

* :func:`open_queue` / :func:`recover_queue` - lifecycle and crash recovery
* :func:`load_declaration` / :func:`load_stream` - validated inputs
* :class:`DelayQueue` - write/claim/ack/nack, snapshots and replay
* :func:`replay_ledger` - post-recovery consistency verification
* :func:`launch_dashboard` - Tkinter diagnostic UI
* Error classes carrying stable codes E_SPEC / E_SNAPSHOT / E_TIMEOUT / ...
"""

from __future__ import annotations

from .errors import (
    DelayQueueError,
    IllegalTransitionError,
    NotReproducibleError,
    ReplayError,
    SnapshotMismatchError,
    SpecError,
    TimeoutConfigError,
)
from .ledger import FORMAT_VERSION, Ledger
from .models import (
    ACKED,
    DEAD,
    DELAYED,
    INFLIGHT,
    PENDING,
    READY,
    Delivery,
    Message,
    MessageInput,
    QueueSpec,
    RecoveryReport,
    TopicSpec,
)
from .queue import DelayQueue
from .replay import replay_ledger
from .specio import load_messages as load_stream
from .specio import load_queue_spec as load_declaration
from .ui import launch_dashboard

__all__ = [
    "DelayQueue", "Ledger", "Delivery", "Message", "MessageInput",
    "QueueSpec", "TopicSpec", "RecoveryReport",
    "load_declaration", "load_stream", "replay_ledger",
    "launch_dashboard", "open_queue", "recover_queue",
    "FORMAT_VERSION",
    "PENDING", "DELAYED", "READY", "INFLIGHT", "ACKED", "DEAD",
    "DelayQueueError", "SpecError", "IllegalTransitionError",
    "SnapshotMismatchError", "ReplayError", "TimeoutConfigError",
    "NotReproducibleError",
]


def open_queue(spec_path: str, message_path: str, ledger_path: str):
    """Create a fresh ledger, validate inputs and enqueue the stream."""
    spec = load_declaration(spec_path)
    inputs = load_stream(message_path, spec)
    queue = DelayQueue.open(spec_path, ledger_path)
    queue.load_inputs(inputs)
    return queue, inputs


def recover_queue(ledger_path: str):
    """Reopen, verify and recover a ledger after a hard process termination."""
    return DelayQueue.reopen(ledger_path)
