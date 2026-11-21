"""Immutable, self-verifying snapshot directories and wholesale restore."""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from .errors import SnapshotError
from .ledger import Ledger
from .models import DiagnosticRun
from .state import run_from_snapshot_payload, state_content_digest
from .util import canonical_json, read_json


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-.")
    return (normalized or "snapshot")[:40]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class SnapshotInfo:
    snapshot_id: str
    plan_version: int
    plan_digest: str
    content_digest: str
    label: str
    path: str
    created_at: str
    read_only: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "plan_version": self.plan_version,
            "plan_digest": self.plan_digest,
            "content_digest": self.content_digest,
            "label": self.label,
            "path": self.path,
            "created_at": self.created_at,
            "read_only": self.read_only,
        }


class SnapshotStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, snapshot_id: str) -> Path:
        if not snapshot_id or "/" in snapshot_id or "\\" in snapshot_id or snapshot_id in {".", ".."}:
            raise SnapshotError("invalid snapshot id", snapshot_id=snapshot_id)
        exact = self.root / snapshot_id
        if exact.is_dir():
            return exact
        matches = sorted(
            path for path in self.root.iterdir() if path.is_dir() and path.name.startswith(snapshot_id)
        )
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise SnapshotError("snapshot does not exist", snapshot_id=snapshot_id)
        raise SnapshotError(
            "snapshot id prefix is ambiguous",
            snapshot_id=snapshot_id,
            candidates=[path.name for path in matches],
        )

    def save(
        self,
        run: DiagnosticRun,
        *,
        label: str = "manual",
        ledger: Ledger | None = None,
        unique: bool = False,
    ) -> SnapshotInfo:
        payload = run.to_snapshot_dict()
        actual_digest = state_content_digest(payload)
        if run.content_digest and run.content_digest != actual_digest:
            raise SnapshotError(
                "runtime content digest is inconsistent",
                recorded=run.content_digest,
                actual=actual_digest,
            )
        payload["content_digest"] = actual_digest
        base_id = f"v{run.plan.plan_version}-{actual_digest[:12]}-{_slug(label)}"
        snapshot_id = base_id
        if unique:
            snapshot_id = f"{base_id}-{uuid4().hex[:8]}"
        destination = self.root / snapshot_id
        if destination.exists():
            existing, _ = self.load(snapshot_id)
            if existing.content_digest == actual_digest and not unique:
                return self._info(destination)
            raise SnapshotError(
                "snapshot destination already contains different content",
                snapshot_id=snapshot_id,
            )

        temporary = self.root / f".tmp-{uuid4().hex}"
        temporary.mkdir(mode=0o700)
        snapshot_path = temporary / "snapshot.json"
        text = canonical_json(payload) + "\n"
        with snapshot_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(snapshot_path, 0o444)
        metadata_path = temporary / "metadata.json"
        metadata = {
            "snapshot_id": snapshot_id,
            "plan_version": run.plan.plan_version,
            "plan_digest": run.plan.digest,
            "content_digest": actual_digest,
            "label": label,
            "created_at": _now(),
        }
        with metadata_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(metadata) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(metadata_path, 0o444)
        temporary.chmod(0o555)
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(self.root, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
            finally:
                os.close(directory_fd)
        info = self._info(destination)
        if ledger is not None:
            ledger.register_snapshot(
                snapshot_id=info.snapshot_id,
                plan_version=info.plan_version,
                plan_digest=info.plan_digest,
                content_digest=info.content_digest,
                path=info.path,
                label=info.label,
            )
        return info

    def _info(self, path: Path) -> SnapshotInfo:
        payload = read_json(path / "snapshot.json")
        metadata_path = path / "metadata.json"
        metadata = read_json(metadata_path) if metadata_path.is_file() else {}
        actual_digest = state_content_digest(payload)
        if payload.get("content_digest") != actual_digest:
            raise SnapshotError(
                "snapshot file was modified after commit",
                snapshot_id=path.name,
                recorded=payload.get("content_digest"),
                actual=actual_digest,
            )
        mode = path.stat().st_mode & 0o777
        file_mode = (path / "snapshot.json").stat().st_mode & 0o777
        return SnapshotInfo(
            snapshot_id=path.name,
            plan_version=int(payload["plan_version"]),
            plan_digest=str(payload["plan_digest"]),
            content_digest=str(payload["content_digest"]),
            label=str(metadata.get("label", "snapshot")),
            path=str(path),
            created_at=str(metadata.get("created_at", _now())),
            read_only=(mode & 0o222) == 0 and (file_mode & 0o222) == 0,
        )

    def load(self, snapshot_id: str) -> tuple[SnapshotInfo, DiagnosticRun]:
        path = self._resolve(snapshot_id)
        payload = read_json(path / "snapshot.json")
        info = self._info(path)
        run = run_from_snapshot_payload(payload)
        if run.plan.plan_version != info.plan_version or run.plan.digest != info.plan_digest:
            raise SnapshotError(
                "snapshot manifest and payload disagree",
                snapshot_id=info.snapshot_id,
                field="plan_version_or_digest",
            )
        return info, run

    def list(self) -> list[SnapshotInfo]:
        infos: list[SnapshotInfo] = []
        for path in sorted(self.root.iterdir(), key=lambda item: item.name):
            if not path.is_dir() or path.name.startswith("."):
                continue
            if not (path / "snapshot.json").is_file():
                continue
            infos.append(self._info(path))
        infos.sort(key=lambda info: (info.plan_version, info.snapshot_id))
        return infos

    def restore(self, snapshot_id: str, ledger: Ledger) -> dict[str, Any]:
        info, target_run = self.load(snapshot_id)
        current = ledger.load_current_run()
        comparison = {
            "current_plan_version": None if current is None else current.plan.plan_version,
            "snapshot_plan_version": target_run.plan.plan_version,
            "current_content_digest": None if current is None else current.content_digest,
            "snapshot_content_digest": target_run.content_digest,
            "current_plan_digest": None if current is None else current.plan.digest,
            "snapshot_plan_digest": target_run.plan.digest,
        }
        same = (
            current is not None
            and current.plan.plan_version == target_run.plan.plan_version
            and current.plan.digest == target_run.plan.digest
            and current.content_digest == target_run.content_digest
        )
        comparison["identical"] = same
        backup_info: SnapshotInfo | None = None
        if not same and current is not None:
            backup_info = self.save(
                current,
                label="pre-restore-backup",
                ledger=ledger,
                unique=True,
            )
        ledger.replace_current_state(target_run, session_id=f"restore:{info.snapshot_id}")
        ledger.event(
            "SNAPSHOT_RESTORE",
            {
                "snapshot_id": info.snapshot_id,
                "comparison": comparison,
                "backup_snapshot_id": None if backup_info is None else backup_info.snapshot_id,
                "replacement": "wholesale",
            },
        )
        return {
            "ok": True,
            "snapshot": info.to_dict(),
            "comparison": comparison,
            "replaced": not same,
            "replacement_mode": "wholesale" if not same else "none",
            "backup_snapshot": None if backup_info is None else backup_info.to_dict(),
            "restored_content_digest": target_run.content_digest,
        }

    def archive_old(self, snapshot_ids: Iterable[str] | None = None) -> list[str]:
        """Make existing snapshot files and directories read-only.

        Snapshots are already created read-only; this method is an explicit
        audit-friendly reinforcement for imported/legacy directories.
        """

        archived: list[str] = []
        paths = (
            [self._resolve(snapshot_id) for snapshot_id in snapshot_ids]
            if snapshot_ids is not None
            else [path for path in self.root.iterdir() if path.is_dir() and not path.name.startswith(".")]
        )
        for path in paths:
            snapshot_file = path / "snapshot.json"
            if snapshot_file.is_file():
                os.chmod(snapshot_file, 0o444)
            os.chmod(path, 0o555)
            archived.append(path.name)
        return sorted(archived)
