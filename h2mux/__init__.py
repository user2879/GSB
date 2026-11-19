"""Deterministic, offline HTTP/2 multiplexing state laboratory."""

from .declaration import load_declaration, load_script
from .errors import ToolError
from .frames import Frame, decode_frame, encode_frame
from .replay import replay_from_snapshot
from .session import H2Session
from .snapshot import SnapshotStore, restore_snapshot

__all__ = [
    "Frame",
    "H2Session",
    "SnapshotStore",
    "ToolError",
    "decode_frame",
    "encode_frame",
    "load_declaration",
    "load_script",
    "replay_from_snapshot",
    "restore_snapshot",
]

__version__ = "1.0.0"
