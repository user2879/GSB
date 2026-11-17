"""Application service combining records, upgrades, replacement and snapshots."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

from .codec import (
    parse_import_document,
    parse_record,
    validate_parent_graph,
    validate_record_versions,
)
from .errors import FormatError, ReplayError, RunError, StateError
from .metrics import record_metric_digest
from .models import CURRENT_FORMAT_VERSION, Library, Provenance, RunRecord
from .repository import RecordRepository, UpgradeResult
from .snapshots import SnapshotStore
from .util import (
    atomic_write_json,
    flatten_mapping,
    read_json,
    safe_component,
    workspace_path,
    workspace_relative,
    write_readonly_once,
)


def reset_session_artifacts(workspace: Path, checkpoint_path: Path | None = None) -> None:
    """Remove only files owned by this application, never arbitrary workspace data."""

    workspace = workspace.resolve()
    for name in ("records.json", "snapshots.sqlite3"):
        path = workspace / name
        if path.exists():
            path.unlink()
    checkpoint = checkpoint_path or (workspace / "upgrade-checkpoint.json")
    if checkpoint.exists():
        checkpoint.unlink()
    for name in ("backups", "reports"):
        directory = workspace / name
        if directory.exists():
            shutil.rmtree(directory)


class ExperimentService:
    def __init__(
        self,
        repository: RecordRepository,
        snapshots: SnapshotStore,
        library: Library,
        upgrade_result: UpgradeResult,
    ) -> None:
        self.repository = repository
        self.snapshots = snapshots
        self.library = library
        self.upgrade_result = upgrade_result

    @classmethod
    def open(
        cls,
        workspace: Path,
        *,
        checkpoint_path: Path | None = None,
        database_path: Path | None = None,
        auto_upgrade: bool = True,
        stop_after: int | None = None,
    ) -> "ExperimentService":
        repository = RecordRepository(workspace, checkpoint_path=checkpoint_path)
        library, upgrade_result = repository.load(
            auto_upgrade=auto_upgrade, stop_after=stop_after
        )
        snapshots = SnapshotStore(repository.workspace, database_path=database_path)
        return cls(repository, snapshots, library, upgrade_result)

    @classmethod
    def initialize(
        cls,
        workspace: Path,
        source: Path | str | dict[str, Any],
        *,
        force: bool = False,
        checkpoint_path: Path | None = None,
        database_path: Path | None = None,
    ) -> "ExperimentService":
        workspace = workspace.resolve()
        records_path = workspace / "records.json"
        if records_path.exists() and not force:
            raise StateError(
                "record library already exists",
                path=str(records_path),
                hint="use force=True or --force to replace the session",
            )
        raw, source_name = cls._read_document(source)
        try:
            library = parse_import_document(raw, path="$")
        except FormatError as exc:
            exc.details.setdefault("source", source_name)
            raise

        if force:
            reset_session_artifacts(workspace, checkpoint_path=checkpoint_path)
        repository = RecordRepository(workspace, checkpoint_path=checkpoint_path)
        repository.initialize(library, overwrite=force)
        return cls.open(
            workspace,
            checkpoint_path=checkpoint_path,
            database_path=database_path,
            auto_upgrade=True,
        )

    @staticmethod
    def _read_document(source: Path | str | dict[str, Any]) -> tuple[Any, str]:
        if isinstance(source, dict):
            return source, "<memory>"
        path = Path(source)
        try:
            raw = read_json(path)
        except FileNotFoundError as exc:
            raise StateError("import declaration file does not exist", path=str(path)) from exc
        except json.JSONDecodeError as exc:
            raise FormatError(
                "$",
                f"invalid JSON: {exc.msg}",
                line=exc.lineno,
                column=exc.colno,
                source=str(path),
            ) from exc
        except OSError as exc:
            raise StateError(
                "cannot read import declaration", path=str(path), error=str(exc)
            ) from exc
        return raw, str(path)

    def list_runs(self) -> list[dict[str, Any]]:
        result = []
        for record in self.library.sorted_runs():
            history = self.library.history_for(record.run_id)
            result.append(
                {
                    "run_id": record.run_id,
                    "run_version": record.run_version,
                    "format_version": record.format_version,
                    "source_format_version": record.provenance.source_format_version,
                    "upgraded": record.provenance.upgraded,
                    "aggregation": record.aggregation.to_dict(),
                    "parent_run_id": record.parent_run_id,
                    "final_metrics": dict(record.final_metrics),
                    "step_count": len(record.step_metrics),
                    "metric_digest": record_metric_digest(record),
                    "content_digest": record.content_digest(),
                    "history_versions": [item.run_version for item in history],
                }
            )
        return result

    def run_detail(self, run_id: str) -> dict[str, Any]:
        current = self.library.current_map().get(run_id)
        if current is None:
            raise StateError("unknown run_id", run_id=run_id)
        history = self.library.history_for(run_id)
        detail = {
            "run": self._run_view(current),
            "step_metrics": [point.to_dict() for point in current.step_metrics],
            "history": [self._run_view(record) for record in history],
        }
        if history:
            detail["replacement"] = build_replacement_difference(history[-1], current)
        return detail

    def _run_view(self, record: RunRecord) -> dict[str, Any]:
        return {
            "run_id": record.run_id,
            "run_version": record.run_version,
            "format_version": record.format_version,
            "source_format_version": record.provenance.source_format_version,
            "upgraded": record.provenance.upgraded,
            "parameters": record.parameters.to_dict(),
            "timestamp": record.timestamp,
            "aggregation": record.aggregation.to_dict(),
            "parent_run_id": record.parent_run_id,
            "final_metrics": dict(record.final_metrics),
            "metric_digest": record_metric_digest(record),
            "content_digest": record.content_digest(),
            "read_only": record.run_version < self.library.current_map()[record.run_id].run_version,
        }

    def commit_record(self, raw: dict[str, Any]) -> dict[str, Any]:
        incoming = parse_record(raw, path="$")
        if incoming.format_version != CURRENT_FORMAT_VERSION:
            raise FormatError(
                "$.format_version",
                "replacement submissions must use the current v3 format",
                supported=[CURRENT_FORMAT_VERSION],
            )
        incoming = replace(
            incoming,
            format_version=CURRENT_FORMAT_VERSION,
            provenance=Provenance(source_format_version=CURRENT_FORMAT_VERSION),
        )
        existing = self.library.current_map().get(incoming.run_id)
        if existing is None:
            if incoming.run_version != 1:
                raise RunError(
                    incoming.run_id,
                    [f"{incoming.run_id}@v1", f"{incoming.run_id}@v{incoming.run_version}"],
                    "a new run_id must start at run_version 1",
                    constraint="run_version_monotonic",
                    versions=[1, incoming.run_version],
                )
            candidate_runs = list(self.library.runs) + [incoming]
            validate_parent_graph(candidate_runs)
            self.library.runs = sorted(candidate_runs, key=lambda item: item.run_id)
            self.repository.save_library(self.library)
            self.repository.write_run_reports(self.library)
            return {
                "mode": "new_run",
                "run_id": incoming.run_id,
                "run_version": incoming.run_version,
                "format_version": incoming.format_version,
                "aggregation": incoming.aggregation.to_dict(),
                "whole_record_written": True,
            }

        if incoming.run_version <= existing.run_version:
            raise RunError(
                incoming.run_id,
                [
                    f"{incoming.run_id}@v{existing.run_version}",
                    f"{incoming.run_id}@v{incoming.run_version}",
                ],
                "new run_version must be greater than the current version",
                constraint="run_version_monotonic",
                versions=sorted([existing.run_version, incoming.run_version]),
            )

        candidate_runs = [
            incoming if record.run_id == incoming.run_id else record
            for record in self.library.runs
        ]
        validate_parent_graph(candidate_runs)
        candidate_history = {
            run_id: list(records) for run_id, records in self.library.history.items()
        }
        candidate_history.setdefault(incoming.run_id, []).append(existing)
        candidate_history[incoming.run_id].sort(key=lambda item: item.run_version)
        validate_record_versions(
            candidate_runs,
            [record for records in candidate_history.values() for record in records],
        )

        backup_path = (
            self.repository.backups_dir
            / (
                f"current-{safe_component(existing.run_id)}-v{existing.run_version}"
                f"-{existing.content_digest()[:16]}.json"
            )
        )
        write_readonly_once(backup_path, existing.to_normalized_dict())
        difference = build_replacement_difference(existing, incoming)
        difference["backup"] = workspace_relative(
            backup_path, self.repository.workspace
        )
        difference["report"] = self.repository.write_replacement_report(difference)

        self.library.runs = sorted(candidate_runs, key=lambda item: item.run_id)
        self.library.history = candidate_history
        self.repository.save_library(self.library)
        self.repository.write_run_reports(self.library)
        return difference

    def snapshot(self, label: str) -> dict[str, Any]:
        return self.snapshots.capture(
            self.library,
            self.repository.checkpoint_view(),
            label=label,
        )

    def list_snapshots(self) -> list[dict[str, Any]]:
        return self.snapshots.list()

    def replay_snapshot(self, snapshot_id: int) -> dict[str, Any]:
        return self.snapshots.replay(snapshot_id)

    def restore_snapshot(
        self, snapshot_id: int, *, replay: bool = True
    ) -> dict[str, Any]:
        snapshot_library = self.snapshots.load_library(snapshot_id)
        artifact = self.snapshots.checkpoint_artifact(snapshot_id)
        difference = self.snapshots.diff_against_disk(snapshot_id, self.library)

        recorded_snapshot_digests = self.snapshots.all_content_digests(snapshot_id)
        state_digests = {
            record.key: record.content_digest()
            for record in snapshot_library.all_records()
        }
        if recorded_snapshot_digests != state_digests:
            raise FormatError(
                f"snapshot[{snapshot_id}].library_json",
                "snapshot state digest set does not match its persisted run summaries",
                snapshot_digests=recorded_snapshot_digests,
                state_digests=state_digests,
            )

        self.repository.save_library(snapshot_library)
        self.library = snapshot_library

        checkpoint_value = artifact["path"]
        if not checkpoint_value:
            if self.repository.checkpoint_path.exists():
                self.repository.checkpoint_path.unlink()
        else:
            checkpoint_path = workspace_path(str(checkpoint_value), self.repository.workspace)
            try:
                checkpoint_path.resolve().relative_to(self.repository.workspace)
            except ValueError as exc:
                raise StateError(
                    "snapshot checkpoint path escapes the workspace",
                    path=str(checkpoint_path),
                ) from exc
            self.repository.checkpoint_path = checkpoint_path
            if artifact["content"] is None:
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
            else:
                atomic_write_json(checkpoint_path, artifact["content"])

        self.repository.write_run_reports(self.library)
        report = {
            "restore": difference,
            "state_replaced": True,
            "checkpoint": {
                "path": workspace_relative(
                    self.repository.checkpoint_path, self.repository.workspace
                )
                if artifact["content"] is not None
                else "",
                "position": artifact["position"],
            },
        }
        if replay:
            try:
                report["replay"] = self.snapshots.replay(snapshot_id)
            except ReplayError as exc:
                exc.details.setdefault("restore_diff", difference)
                exc.details.setdefault("state_replaced", True)
                raise
        return report

    def status(self) -> dict[str, Any]:
        checkpoint = self.repository.checkpoint_view()
        return {
            "workspace": str(self.repository.workspace),
            "library_id": self.library.library_id,
            "current_format_version": CURRENT_FORMAT_VERSION,
            "format_versions": self.library.format_versions(),
            "source_format_versions": self.library.source_format_versions(),
            "aggregation_versions": self.library.aggregation_versions(),
            "runs": self.list_runs(),
            "snapshots": self.list_snapshots(),
            "upgrade": self.upgrade_result.to_dict(),
            "checkpoint": {
                "path": checkpoint["path"] if checkpoint else "",
                "position": checkpoint["content"].get("position", 0)
                if checkpoint
                else 0,
                "status": checkpoint["content"].get("status", "absent")
                if checkpoint
                else "absent",
            },
        }


def build_replacement_difference(
    before: RunRecord, after: RunRecord
) -> dict[str, Any]:
    before_fields = {
        **{f"parameters.{key}": value for key, value in flatten_mapping(before.parameters.to_dict()).items()},
        "aggregation.name": before.aggregation.name,
        "aggregation.version": before.aggregation.version,
        "parent_run_id": before.parent_run_id,
    }
    after_fields = {
        **{f"parameters.{key}": value for key, value in flatten_mapping(after.parameters.to_dict()).items()},
        "aggregation.name": after.aggregation.name,
        "aggregation.version": after.aggregation.version,
        "parent_run_id": after.parent_run_id,
    }
    changed_fields = []
    for field in sorted(set(before_fields) | set(after_fields)):
        if before_fields.get(field) != after_fields.get(field):
            changed_fields.append(
                {
                    "field": field,
                    "before": before_fields.get(field),
                    "after": after_fields.get(field),
                }
            )

    common = min(len(before.step_metrics), len(after.step_metrics))
    first_changed_step = None
    changed_steps = 0
    loss_delta_sum = 0.0
    accuracy_delta_sum = 0.0
    for index in range(common):
        left = before.step_metrics[index]
        right = after.step_metrics[index]
        if (left.loss, left.accuracy) != (right.loss, right.accuracy):
            changed_steps += 1
            if first_changed_step is None:
                first_changed_step = right.step
            loss_delta_sum += right.loss - left.loss
            accuracy_delta_sum += right.accuracy - left.accuracy
    if len(before.step_metrics) != len(after.step_metrics):
        if first_changed_step is None:
            first_changed_step = common + 1
        changed_steps += abs(len(after.step_metrics) - len(before.step_metrics))

    final_loss_delta = after.final_metrics["loss"] - before.final_metrics["loss"]
    final_accuracy_delta = (
        after.final_metrics["accuracy"] - before.final_metrics["accuracy"]
    )
    return {
        "run_id": after.run_id,
        "old_version": before.run_version,
        "new_version": after.run_version,
        "whole_record_replaced": True,
        "history_retained": True,
        "changed_fields": changed_fields,
        "aggregation": {
            "before": before.aggregation.to_dict(),
            "after": after.aggregation.to_dict(),
            "version_changed": before.aggregation.version != after.aggregation.version,
        },
        "metric_changes": {
            "old_step_count": len(before.step_metrics),
            "new_step_count": len(after.step_metrics),
            "first_changed_step": first_changed_step,
            "changed_step_count": changed_steps,
            "final_loss": {
                "before": before.final_metrics["loss"],
                "after": after.final_metrics["loss"],
                "delta": final_loss_delta,
            },
            "final_accuracy": {
                "before": before.final_metrics["accuracy"],
                "after": after.final_metrics["accuracy"],
                "delta": final_accuracy_delta,
            },
            "changed_step_loss_delta_sum": loss_delta_sum,
            "changed_step_accuracy_delta_sum": accuracy_delta_sum,
        },
        "before_metric_digest": record_metric_digest(before),
        "after_metric_digest": record_metric_digest(after),
        "before_content_digest": before.content_digest(),
        "after_content_digest": after.content_digest(),
    }
