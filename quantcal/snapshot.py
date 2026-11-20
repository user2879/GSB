"""Atomic stage publication and crash/staleness handling."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .errors import SnapshotStale
from .util import (
    atomic_write_json,
    canonical_json_bytes,
    load_json,
    sha256_bytes,
)


def _seal(payload: dict[str, Any], field: str = "manifest_hash") -> dict[str, Any]:
    sealed = dict(payload)
    sealed.pop(field, None)
    sealed[field] = sha256_bytes(canonical_json_bytes(sealed))
    return sealed


def _verify_seal(payload: Any, field: str = "manifest_hash") -> bool:
    if not isinstance(payload, dict) or field not in payload:
        return False
    body = {key: value for key, value in payload.items() if key != field}
    return payload[field] == sha256_bytes(canonical_json_bytes(body))


class SnapshotManager:
    def __init__(
        self,
        work_dir: str | Path,
        config_fingerprint: str,
        exchange_version: str,
        resume: bool = False,
    ) -> None:
        self.work_dir = Path(work_dir)
        self.state_dir = self.work_dir / "state"
        self.stages_dir = self.work_dir / "stages"
        self.tmp_dir = self.work_dir / "tmp"
        self.snapshot_path = self.state_dir / "snapshot.json"
        self.intent_path = self.state_dir / "run_intent.json"
        self.pointer_path = self.state_dir / "stage_pointer.json"
        self.progress_path = self.state_dir / "progress.json"
        self.config_fingerprint = config_fingerprint
        self.exchange_version = exchange_version
        self.resume = resume
        self.previous_snapshot: dict[str, Any] | None = None
        self.previous_version = 0
        self.version = 1
        self.stage_pointer: dict[str, Any] = {"stages": {}}

    def startup(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.stages_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        if self.snapshot_path.exists():
            snapshot = load_json(self.snapshot_path)
            if not _verify_seal(snapshot):
                raise SnapshotStale("snapshot manifest seal is invalid or was tampered with")
            if snapshot.get("format") != "quantcal.snapshot.v1":
                raise SnapshotStale("snapshot manifest format is not supported")
            version = snapshot.get("global_version")
            if not isinstance(version, int) or isinstance(version, bool) or version < 1:
                raise SnapshotStale("snapshot global_version is invalid")
            self.previous_snapshot = snapshot
            self.previous_version = version
            if self.resume:
                if snapshot.get("config_fingerprint") != self.config_fingerprint:
                    raise SnapshotStale(
                        "resume config fingerprint differs from committed snapshot",
                        expected=snapshot.get("config_fingerprint"),
                        actual=self.config_fingerprint,
                    )
                if snapshot.get("exchange_version") != self.exchange_version:
                    raise SnapshotStale(
                        "resume exchange version differs from committed snapshot",
                        expected=snapshot.get("exchange_version"),
                        actual=self.exchange_version,
                    )
            pointer = {}
            if self.pointer_path.exists():
                pointer = load_json(self.pointer_path)
                if not isinstance(pointer, dict) or not isinstance(pointer.get("stages"), dict):
                    raise SnapshotStale("stage pointer is malformed")
            self.stage_pointer = pointer or {"stages": {}}
        if self.intent_path.exists():
            intent = load_json(self.intent_path)
            if not _verify_seal(intent, "intent_hash"):
                raise SnapshotStale("unfinished run intent seal is invalid")
            intent_version = intent.get("global_version")
            if intent_version == self.previous_version:
                # The snapshot was finalized but the process stopped before removing
                # the intent. This is a completed generation, not an unfinished one.
                os.unlink(self.intent_path)
            elif intent_version != self.previous_version + 1:
                raise SnapshotStale(
                    "unfinished run version does not follow the committed snapshot",
                    committed_version=self.previous_version,
                    intent_version=intent_version,
                )
            else:
                if intent.get("previous_global_version") != self.previous_version:
                    raise SnapshotStale(
                        "unfinished run predecessor version is inconsistent",
                        committed_version=self.previous_version,
                        intent_previous=intent.get("previous_global_version"),
                    )
                if intent.get("config_fingerprint") != self.config_fingerprint:
                    raise SnapshotStale(
                        "unfinished run was started with a different config fingerprint",
                        expected=intent.get("config_fingerprint"),
                        actual=self.config_fingerprint,
                    )
                if intent.get("exchange_version") != self.exchange_version:
                    raise SnapshotStale(
                        "unfinished run was started with a different exchange version",
                        expected=intent.get("exchange_version"),
                        actual=self.exchange_version,
                    )
        self.version = self.previous_version + 1
        intent = _seal(
            {
                "format": "quantcal.run-intent.v1",
                "global_version": self.version,
                "previous_global_version": self.previous_version,
                "config_fingerprint": self.config_fingerprint,
                "exchange_version": self.exchange_version,
            },
            "intent_hash",
        )
        atomic_write_json(self.intent_path, intent)

    def stage_version_name(self, phase: str, content_hash: str) -> str:
        return f"{phase}-g{self.version:06d}-{content_hash[:12]}"

    def publish_stage(
        self, phase: str, files: dict[str, Any], content_hash: str
    ) -> dict[str, Any]:
        final_name = self.stage_version_name(phase, content_hash)
        final_dir = self.stages_dir / final_name
        if not final_dir.exists():
            temporary = Path(tempfile.mkdtemp(prefix=f".{phase}-", dir=self.tmp_dir))
            try:
                for relative, payload in files.items():
                    target = temporary / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    atomic_write_json(target, payload)
                temporary.replace(final_dir)
            except Exception:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
        stages = dict(self.stage_pointer.get("stages", {}))
        stages[phase] = {
            "path": str(final_dir.relative_to(self.work_dir)),
            "content_fingerprint": content_hash,
            "artifact_version": self.version,
        }
        pointer = {
            "format": "quantcal.stage-pointer.v1",
            "global_version": self.version,
            "stages": stages,
        }
        pointer["pointer_hash"] = sha256_bytes(canonical_json_bytes(pointer))
        atomic_write_json(self.pointer_path, pointer)
        self.stage_pointer = pointer
        return stages[phase]

    def checkpoint(self, phase: str, completed: list[str]) -> None:
        payload = {
            "format": "quantcal.progress.v1",
            "global_version": self.version,
            "config_fingerprint": self.config_fingerprint,
            "exchange_version": self.exchange_version,
            "phase": phase,
            "completed_tensors": sorted(completed),
        }
        payload["progress_hash"] = sha256_bytes(canonical_json_bytes(payload))
        atomic_write_json(self.progress_path, payload)

    def finalize(self, details: dict[str, Any]) -> dict[str, Any]:
        snapshot = {
            "format": "quantcal.snapshot.v1",
            "global_version": self.version,
            "previous_global_version": self.previous_version,
            "config_fingerprint": self.config_fingerprint,
            "exchange_version": self.exchange_version,
            "stages": self.stage_pointer.get("stages", {}),
            **details,
        }
        snapshot = _seal(snapshot)
        atomic_write_json(self.snapshot_path, snapshot)
        for path in (self.intent_path, self.progress_path):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        self.previous_snapshot = snapshot
        self.previous_version = self.version
        return snapshot


def snapshot_public_view(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return {
        "global_version": snapshot.get("global_version"),
        "config_fingerprint": snapshot.get("config_fingerprint"),
        "exchange_version": snapshot.get("exchange_version"),
        "exchange_version_fingerprint": snapshot.get("exchange_version_fingerprint"),
        "bypass_metadata_hash": snapshot.get("bypass_metadata_hash"),
        "completed_tensors": snapshot.get("completed_tensors", []),
        "manifest_hash": snapshot.get("manifest_hash"),
    }
