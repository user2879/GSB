"""Local HTTP/2 burst-window diagnostic workbench."""

from .engine import ENGINE_FORMAT_VERSION, Engine, EngineOptions, run_session
from .errors import WorkbenchError
from .frames import (
    Frame,
    STATIC_TABLE,
    decode_frame,
    decode_frames,
    decode_header_block,
    encode_frame,
    encode_frames,
    encode_header_block,
)
from .model import SessionSpec, load_session, parse_session
from .replay import ReplayResult, replay_from_snapshot, replay_from_state
from .snapshot import RestoreResult, SnapshotInfo, SnapshotStore
from .viewmodel import WorkbenchViewModel

__all__ = [
    "ENGINE_FORMAT_VERSION",
    "Engine",
    "EngineOptions",
    "Frame",
    "ReplayResult",
    "RestoreResult",
    "STATIC_TABLE",
    "SessionSpec",
    "SnapshotInfo",
    "SnapshotStore",
    "WorkbenchError",
    "WorkbenchViewModel",
    "decode_frame",
    "decode_frames",
    "decode_header_block",
    "encode_frame",
    "encode_frames",
    "encode_header_block",
    "load_session",
    "parse_session",
    "replay_from_snapshot",
    "replay_from_state",
    "run_session",
]
