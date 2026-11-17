"""JSON record-library persistence and restart-safe format upgrades."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .codec import parse_library, validate_parent_graph, validate_record_versions
from .errors import FormatError, StateError
from .metrics import record_metric_digest
from .models import CURRENT_FORMAT_VERSION, Library, RunRecord
from .util import (
    atomic_write_json,
    read_json,
    safe_component,
    sha256_json,
    sha256_text,
    workspace_relative,
    write_readonly_once,
    write_readonly_text_once,
)

CHECKPOINT_SCHEMA = "upgrade-checkpoint-v1"


class UpgradeInterrupted(RuntimeError):
    """Raised by the deterministic stop-after test/demo hook."""

    def __init__(self, entry: dict[str, Any], checkpoint_path: Path) -> None:
        super().__init__(f"upgrade interrupted after {entry['key']}")
        self.entry = entry
        self.checkpoint_path = checkpoint_path


@dataclass
class UpgradeResult:
    processed: list[dict[str, Any]] = field(default_factory=list)
    recovered: list[str] = field(default_factory=list)
    checkpoint_path: str = ""
    position: int = 0
    total: int = 0
    status: str = "not_needed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "processed": self.processed,
            "recovered_from_active": self.recovered,
            "checkpoint_path": self.checkpoint_path,
            "position": self.position,
            "total": self.total,
        }


class RecordRepository:
    def __init__(self, workspace: Path, checkpoint_path: Path | None = None) -> None:
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.records_path = self.workspace / "records.json"
        self.backups_dir = self.workspace / "backups"
        self.reports_dir = self.workspace / "reports"
        self.checkpoint_path = (
            checkpoint_path.resolve()
            if checkpoint_path is not None and checkpoint_path.is_absolute()
            else (self.workspace / (checkpoint_path or "upgrade-checkpoint.json")).resolve()
        )
        self.backups_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def initialize(self, library: Library, *, overwrite: bool = False) -> None:
        if self.records_path.exists() and not overwrite:
            raise StateError(
                "record library already exists",
                path=str(self.records_path),
                hint="use --force to replace the workspace library",
            )
        atomic_write_json(self.records_path, library.to_envelope())
        if overwrite and self.checkpoint_path.exists():
            self.checkpoint_path.unlink()

    def read_library(self) -> tuple[Library, str]:
        if not self.records_path.exists():
            raise StateError(
                "record library does not exist",
                path=str(self.records_path),
                hint="run the init command first",
            )
        try:
            raw_text = self.records_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise StateError(
                "cannot read record library", path=str(self.records_path), error=str(exc)
            ) from exc
        try:
            raw = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise FormatError(
                "$",
                f"invalid JSON: {exc.msg}",
                line=exc.lineno,
                column=exc.colno,
            ) from exc
        return parse_library(raw, path="$"), raw_text

    def save_library(self, library: Library) -> None:
        validate_record_versions(
            library.runs,
            [record for records in library.history.values() for record in records],
        )
        validate_parent_graph(library.runs)
        atomic_write_json(self.records_path, library.to_envelope())

    def load(
        self, *, auto_upgrade: bool = True, stop_after: int | None = None
    ) -> tuple[Library, UpgradeResult]:
        library, raw_text = self.read_library()
        result = UpgradeResult()
        if auto_upgrade and any(record.format_version < CURRENT_FORMAT_VERSION for record in library.all_records()):
            result = self._upgrade_library(library, raw_text, stop_after=stop_after)
        self.write_run_reports(library)
        return library, result

    def checkpoint_view(self) -> dict[str, Any] | None:
        if not self.checkpoint_path.exists():
            return None
        return {
            "path": workspace_relative(self.checkpoint_path, self.workspace),
            "content": self._load_checkpoint(),
        }

    def write_run_reports(self, library: Library) -> None:
        for record in library.sorted_runs():
            history = library.history_for(record.run_id)
            report = {
                "run_id": record.run_id,
                "current": self._record_summary(record),
                "history": [self._record_summary(item) for item in history],
            }
            path = self.reports_dir / "runs" / f"{safe_component(record.run_id)}.json"
            atomic_write_json(path, report)

    def write_replacement_report(self, difference: dict[str, Any]) -> str:
        run_id = difference["run_id"]
        run_version = difference["new_version"]
        path = (
            self.reports_dir
            / "replacements"
            / f"{safe_component(run_id)}-v{run_version}.json"
        )
        atomic_write_json(path, difference)
        return workspace_relative(path, self.workspace)

    def _record_summary(self, record: RunRecord) -> dict[str, Any]:
        return {
            "run_version": record.run_version,
            "format_version": record.format_version,
            "source_format_version": record.provenance.source_format_version,
            "upgraded": record.provenance.upgraded,
            "aggregation": record.aggregation.to_dict(),
            "step_count": len(record.step_metrics),
            "final_metrics": dict(record.final_metrics),
            "metric_digest": record_metric_digest(record),
            "content_digest": record.content_digest(),
        }

    def _load_checkpoint(self) -> dict[str, Any]:
        try:
            checkpoint = read_json(self.checkpoint_path)
        except json.JSONDecodeError as exc:
            raise FormatError(
                str(self.checkpoint_path),
                f"invalid upgrade checkpoint JSON: {exc.msg}",
                line=exc.lineno,
                column=exc.colno,
            ) from exc
        if not isinstance(checkpoint, dict):
            raise FormatError(str(self.checkpoint_path), "checkpoint must be an object")
        if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
            raise FormatError(
                f"{self.checkpoint_path}.schema",
                f"unsupported checkpoint schema {checkpoint.get('schema')!r}",
            )
        if checkpoint.get("target_format") != CURRENT_FORMAT_VERSION:
            raise FormatError(
                f"{self.checkpoint_path}.target_format",
                "checkpoint target format is not current",
            )
        if not isinstance(checkpoint.get("completed"), list):
            raise FormatError(
                f"{self.checkpoint_path}.completed", "expected an array"
            )
        active = checkpoint.get("active")
        if active is not None and not isinstance(active, dict):
            raise FormatError(f"{self.checkpoint_path}.active", "expected object or null")
        return checkpoint

    def _save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        atomic_write_json(self.checkpoint_path, checkpoint)

    def _new_checkpoint(
        self, library: Library, total: int, completed: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        completed = completed or []
        return {
            "schema": CHECKPOINT_SCHEMA,
            "library_id": library.library_id,
            "record_path": workspace_relative(self.records_path, self.workspace),
            "target_format": CURRENT_FORMAT_VERSION,
            "status": "running",
            "position": len(completed),
            "total": total,
            "active": None,
            "completed": completed,
        }

    def _legacy_records(self, library: Library) -> list[RunRecord]:
        return [
            record
            for record in library.all_records()
            if record.format_version < CURRENT_FORMAT_VERSION
        ]

    def _find_record(
        self,
        library: Library,
        run_id: str,
        run_version: int,
        *,
        path: str | None = None,
    ) -> RunRecord:
        for record in library.all_records():
            if record.run_id == run_id and record.run_version == run_version:
                return record
        raise FormatError(
            path or f"{self.checkpoint_path}.active.key",
            f"checkpoint references missing record {run_id}@v{run_version}",
        )

    def _replace_record(self, library: Library, old: RunRecord, new: RunRecord) -> None:
        for index, record in enumerate(library.runs):
            if record.key == old.key and record.format_version == old.format_version:
                library.runs[index] = new
                return
        for history in library.history.values():
            for index, record in enumerate(history):
                if record.key == old.key and record.format_version == old.format_version:
                    history[index] = new
                    return
        raise StateError("record to upgrade disappeared from the in-memory library")

    def _upgrade_library(
        self, library: Library, raw_text: str, *, stop_after: int | None
    ) -> UpgradeResult:
        legacy = sorted(
            self._legacy_records(library),
            key=lambda record: (record.run_id, record.run_version, record.format_version),
        )
        raw_digest = sha256_text(raw_text)
        library_backup = self.backups_dir / f"library-before-upgrade-{raw_digest[:16]}.json"
        write_readonly_text_once(library_backup, raw_text)

        if self.checkpoint_path.exists():
            checkpoint = self._load_checkpoint()
            if checkpoint.get("library_id") != library.library_id:
                raise FormatError(
                    f"{self.checkpoint_path}.library_id",
                    "checkpoint belongs to a different library",
                    checkpoint_library_id=checkpoint.get("library_id"),
                    library_id=library.library_id,
                )
            completed = list(checkpoint["completed"])
        else:
            completed = []
            checkpoint = self._new_checkpoint(library, total=len(legacy), completed=completed)
            self._save_checkpoint(checkpoint)

        result = UpgradeResult(
            checkpoint_path=workspace_relative(self.checkpoint_path, self.workspace),
            total=int(checkpoint.get("total", len(legacy))),
            status="running",
        )

        completed_keys = {
            entry.get("key") for entry in completed if isinstance(entry, dict)
        }
        for entry in completed:
            if not isinstance(entry, dict):
                raise FormatError(
                    f"{self.checkpoint_path}.completed",
                    "completed checkpoint entries must be objects",
                )
            run_id = entry.get("run_id")
            run_version = entry.get("run_version")
            if not isinstance(run_id, str) or not isinstance(run_version, int):
                raise FormatError(
                    f"{self.checkpoint_path}.completed",
                    "completed checkpoint entry is incomplete",
                    entry=entry,
                )
            completed_record = self._find_record(
                library,
                run_id,
                run_version,
                path=f"{self.checkpoint_path}.completed.{entry.get('key')}",
            )
            if completed_record.format_version < CURRENT_FORMAT_VERSION:
                raise FormatError(
                    f"{self.checkpoint_path}.completed.{entry.get('key')}",
                    "checkpoint marks a record complete but the disk record is still legacy",
                    run_id=run_id,
                    run_version=run_version,
                )
        active = checkpoint.get("active")
        if active is not None:
            active_key = active.get("key")
            active_run_id = active.get("run_id")
            active_version = active.get("run_version")
            if not isinstance(active_run_id, str) or not isinstance(active_version, int):
                raise FormatError(
                    f"{self.checkpoint_path}.active",
                    "active checkpoint entry is incomplete",
                )
            active_record = self._find_record(library, active_run_id, active_version)
            target_digest = active.get("target_digest")
            source_digest = active.get("source_digest")
            current_digest = sha256_json(active_record.to_storage_dict())
            if (
                active_record.format_version == CURRENT_FORMAT_VERSION
                and active_record.content_digest() == target_digest
            ):
                if active_key not in completed_keys:
                    completed.append(active)
                    completed_keys.add(active_key)
                    result.recovered.append(str(active_key))
                checkpoint["completed"] = completed
                checkpoint["active"] = None
                checkpoint["position"] = len(completed)
                self._save_checkpoint(checkpoint)
            elif active_record.format_version < CURRENT_FORMAT_VERSION and current_digest == source_digest:
                checkpoint["active"] = None
                self._save_checkpoint(checkpoint)
            else:
                raise FormatError(
                    f"{self.checkpoint_path}.active",
                    "active checkpoint entry conflicts with the on-disk record",
                    active_key=active_key,
                )

        checkpoint["status"] = "running"
        checkpoint["position"] = len(completed)
        self._save_checkpoint(checkpoint)

        processed_now = 0
        while True:
            remaining = [
                record
                for record in self._legacy_records(library)
                if record.key not in completed_keys
            ]
            remaining.sort(key=lambda record: (record.run_id, record.run_version))
            if not remaining:
                break
            record = remaining[0]
            source_dict = record.to_storage_dict()
            source_digest = sha256_json(source_dict)
            target_record = record.upgraded(original_digest=source_digest)
            target_digest = target_record.content_digest()
            backup_path = (
                self.backups_dir
                / (
                    f"{safe_component(record.run_id)}-v{record.run_version}"
                    f"-f{record.format_version}-{source_digest[:16]}.json"
                )
            )
            write_readonly_once(backup_path, source_dict)
            entry: dict[str, Any] = {
                "key": record.key,
                "run_id": record.run_id,
                "run_version": record.run_version,
                "source_format": record.format_version,
                "target_format": CURRENT_FORMAT_VERSION,
                "source_digest": source_digest,
                "target_digest": target_digest,
                "backup": workspace_relative(backup_path, self.workspace),
            }

            checkpoint["active"] = entry
            self._save_checkpoint(checkpoint)

            self._replace_record(library, record, target_record)
            self.save_library(library)

            completed.append(entry)
            completed_keys.add(entry["key"])
            checkpoint["completed"] = completed
            checkpoint["active"] = None
            checkpoint["position"] = len(completed)
            checkpoint["total"] = max(int(checkpoint.get("total", 0)), len(completed))
            self._save_checkpoint(checkpoint)

            result.processed.append(entry)
            processed_now += 1
            if stop_after is not None and processed_now >= stop_after:
                raise UpgradeInterrupted(entry, self.checkpoint_path)

        checkpoint["status"] = "complete"
        checkpoint["active"] = None
        checkpoint["position"] = len(completed)
        self._save_checkpoint(checkpoint)
        if self._legacy_records(library):
            raise FormatError(
                str(self.checkpoint_path),
                "upgrade checkpoint completed while legacy records still remain",
            )
        result.status = "complete"
        result.position = len(completed)
        result.checkpoint_path = workspace_relative(self.checkpoint_path, self.workspace)
        return result
