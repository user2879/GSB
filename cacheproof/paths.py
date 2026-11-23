"""Workspace path safety and deterministic artifact locations."""

from __future__ import annotations

import base64
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .canonical import canonical_text
from .errors import DiagnosticError


_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def validate_workspace_root(root: str | Path) -> Path:
    """Validate a workspace that will contain all mutable artifacts."""

    raw = Path(root).expanduser()
    resolved = raw.resolve()
    if resolved == Path(resolved.anchor):
        raise DiagnosticError(
            "E_IO",
            "workspace root must not be a filesystem root",
            field="workspace",
            details={"path": str(resolved)},
        )
    if os.name == "nt":
        forbidden = [
            Path(os.environ.get("SystemRoot", r"C:\Windows")),
            Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
            Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
            Path(os.environ.get("ProgramData", r"C:\ProgramData")),
        ]
        for candidate in forbidden:
            try:
                candidate_resolved = candidate.resolve()
            except OSError:
                continue
            if resolved == candidate_resolved or _is_relative_to(
                resolved, candidate_resolved
            ):
                raise DiagnosticError(
                    "E_IO",
                    "workspace is inside a Windows system-reserved path",
                    field="workspace",
                    details={
                        "path": str(resolved),
                        "reserved_root": str(candidate_resolved),
                    },
                )
        for part in resolved.parts:
            if part.rstrip(" .").upper() in _WINDOWS_RESERVED:
                raise DiagnosticError(
                    "E_IO",
                    "workspace contains a Windows reserved name component",
                    field="workspace",
                    details={"path": str(resolved), "component": part},
                )
    return resolved


def encode_component(value: str) -> str:
    encoded = base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii")
    return encoded.rstrip("=")


def decode_component(value: str) -> str:
    padding = "=" * ((4 - len(value) % 4) % 4)
    return base64.urlsafe_b64decode(value + padding).decode("utf-8")


def entry_relative_path(segment_id: str, entry_key: str, digest: str) -> str:
    digest_hex = digest.removeprefix("sha256:")
    return "/".join(
        ("entries", encode_component(segment_id), encode_component(entry_key), digest_hex + ".bin")
    )


def safe_absolute(root: Path, relative_path: str) -> Path:
    if not relative_path or relative_path.startswith(("/", "\\")):
        raise DiagnosticError(
            "E_INTEGRITY",
            "unsafe relative artifact path",
            field="file_relpath",
            details={"path": relative_path},
        )
    candidate = (root / relative_path).resolve()
    if not _is_relative_to(candidate, root):
        raise DiagnosticError(
            "E_INTEGRITY",
            "artifact path escapes the workspace",
            field="file_relpath",
            details={"path": relative_path},
        )
    return candidate


def atomic_write_bytes(destination: str | Path, data: bytes, temp_directory: str | Path) -> None:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    Path(temp_directory).mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=".cacheproof-atomic-", suffix=".tmp", dir=temp_directory
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(handle, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_path, destination_path)
    except BaseException:
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise


def atomic_write_json(destination: str | Path, value: Any, temp_directory: str | Path) -> None:
    atomic_write_bytes(
        destination,
        (canonical_text(value) + "\n").encode("utf-8"),
        temp_directory,
    )


class WorkspacePaths:
    def __init__(self, root: str | Path):
        self.root = validate_workspace_root(root)
        self.ledger = self.root / "ledger.sqlite3"
        self.entries = self.root / "entries"
        self.snapshots = self.root / "snapshots"
        self.logs = self.root / "logs"
        self.tmp = self.root / "tmp"
        self.config = self.root / "config"
        self.session_snapshot = self.snapshots / "session.json"
        self.spec_copy = self.config / "spec.json"
        self.event_log = self.logs / "events.jsonl"
        self.marker = self.root / ".cacheproof-workspace"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in (
            self.entries,
            self.snapshots,
            self.logs,
            self.tmp,
            self.config,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        if not self.marker.exists():
            atomic_write_text(
                self.marker,
                canonical_text(
                    {
                        "format_version": 1,
                        "kind": "cacheproof-workspace",
                        "note": "All mutable files are contained in this directory.",
                    }
                )
                + "\n",
                self.tmp,
            )

    def relative_entry_path(self, segment_id: str, entry_key: str, digest: str) -> str:
        return entry_relative_path(segment_id, entry_key, digest)

    def absolute_entry_path(self, relative_path: str) -> Path:
        return safe_absolute(self.root, relative_path)


def atomic_write_text(destination: str | Path, text: str, temp_directory: str | Path) -> None:
    atomic_write_bytes(destination, text.encode("utf-8"), temp_directory)


def reset_demo_workspace(paths: WorkspacePaths) -> None:
    root = paths.root
    if not root.exists():
        return
    if paths.marker.exists() or not any(root.iterdir()):
        shutil.rmtree(root)
        return
    raise DiagnosticError(
        "E_IO",
        "refusing to reset a non-empty directory without the cacheproof workspace marker",
        field="workspace",
        details={"path": str(root)},
    )
