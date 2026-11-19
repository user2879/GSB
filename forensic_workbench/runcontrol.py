"""Controlled-exit manifests versus forced-termination recovery evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .util import atomic_write_json, read_json


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class RunSession:
    run_dir: Path
    run_id: str
    workspace: Path
    base_version: int | None
    base_hash: str | None

    @classmethod
    def start(
        cls,
        run_dir: Path,
        *,
        run_id: str,
        workspace: Path,
        base_version: int | None,
        base_hash: str | None,
        mode: str = "normal",
    ) -> "RunSession":
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        session = cls(
            run_dir=run_dir,
            run_id=run_id,
            workspace=Path(workspace),
            base_version=base_version,
            base_hash=base_hash,
        )
        atomic_write_json(
            run_dir / "session.json",
            {
                "format": "forensic-run-session-v1",
                "run_id": run_id,
                "mode": mode,
                "status": "running",
                "started_at_utc": _utc_now(),
                "workspace": str(Path(workspace)),
                "base_version": base_version,
                "base_snapshot_hash": base_hash,
            },
        )
        return session

    def close(
        self,
        *,
        snapshot_version: int,
        snapshot_hash: str,
        node_hashes: dict[str, str],
        conclusion_hash: str,
        run_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        manifest = {
            "format": "forensic-exit-manifest-v1",
            "run_id": self.run_id,
            "snapshot_version": snapshot_version,
            "snapshot_hash": snapshot_hash,
            "node_hashes": node_hashes,
            "conclusion_hash": conclusion_hash,
            "run_summary": run_summary or {},
            "closed_at_utc": _utc_now(),
        }
        manifest_path = self.run_dir / "exit_manifest.json"
        manifest_hash = atomic_write_json(manifest_path, manifest)
        close_marker = {
            "format": "forensic-close-v1",
            "clean_exit": True,
            "run_id": self.run_id,
            "snapshot_version": snapshot_version,
            "snapshot_hash": snapshot_hash,
            "conclusion_hash": conclusion_hash,
            "exit_manifest": manifest_path.name,
            "exit_manifest_hash": manifest_hash,
            "closed_at_utc": _utc_now(),
        }
        atomic_write_json(self.run_dir / "close.json", close_marker)
        return close_marker


def inspect_run(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    session_path = run_dir / "session.json"
    close_path = run_dir / "close.json"
    if not session_path.exists():
        return {
            "run_dir": str(run_dir),
            "session_present": False,
            "clean_exit": False,
            "status": "no_session",
        }
    try:
        session = read_json(session_path)
    except Exception as exc:
        return {
            "run_dir": str(run_dir),
            "session_present": True,
            "clean_exit": False,
            "status": "invalid_session",
            "error": str(exc),
        }
    if close_path.exists():
        try:
            close_marker = read_json(close_path)
        except Exception as exc:
            return {
                "run_dir": str(run_dir),
                "session_present": True,
                "close_present": True,
                "clean_exit": False,
                "status": "invalid_close_marker",
                "session": session,
                "error": str(exc),
            }
        return {
            "run_dir": str(run_dir),
            "session_present": True,
            "close_present": True,
            "clean_exit": bool(close_marker.get("clean_exit")),
            "status": "controlled_exit",
            "session": session,
            "close": close_marker,
        }
    return {
        "run_dir": str(run_dir),
        "session_present": True,
        "close_present": False,
        "clean_exit": False,
        "status": "unclean_termination",
        "session": session,
    }


def record_recovery(run_dir: Path, value: dict[str, Any]) -> Path:
    run_dir = Path(run_dir)
    path = run_dir / "recovery-report.json"
    atomic_write_json(
        path,
        {
            "format": "forensic-run-recovery-v1",
            "recovered_at_utc": _utc_now(),
            **value,
        },
    )
    return path

