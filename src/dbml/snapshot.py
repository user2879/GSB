from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any

from .analysis import derivation_graph_payload, derivation_graph_version
from .database import (
    canonical_json,
    connect,
    database_fingerprint,
    read_column_meta,
    write_json_atomic,
)
from .errors import MigrationError, fail
from .ledger import ledger_view
from .manifest import Manifest, manifest_from_plan


SNAPSHOT_FORMAT = 1


def _digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _calibration_version(calibrations: list[dict[str, Any]]) -> str:
    return _digest(calibrations)


def _conclusion_version(conclusions: dict[str, Any]) -> str:
    return _digest(conclusions)


def _session_version(payload: dict[str, Any]) -> str:
    version_input = {
        "schema_version": payload["schema_version"],
        "data_version": payload["data_version"],
        "calibration_version": payload["calibration_version"],
        "derivation_graph_version": payload["derivation_graph_version"],
        "ledger_version": payload["ledger_version"],
        "conclusion_version": payload["conclusion_version"],
    }
    return _digest(version_input)


def _current_session_state(
    connection,
    manifest: Manifest,
    *,
    conclusions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fingerprint = database_fingerprint(connection)
    ledger = ledger_view(connection)
    calibrations = read_column_meta(connection)
    calibration_list = [meta.to_dict() for meta in calibrations.values()]
    graph = derivation_graph_payload(manifest)
    conclusions = conclusions or {}
    session = {
        "schema_version": fingerprint["schema_version"],
        "data_summary": fingerprint["tables"],
        "total_rows": fingerprint["total_rows"],
        "data_version": fingerprint["data_version"],
        "column_calibrations": calibration_list,
        "calibration_version": _calibration_version(calibration_list),
        "derivation_graph": graph,
        "derivation_graph_version": derivation_graph_version(manifest),
        "ledger_view": ledger,
        "ledger_version": ledger["ledger_version"],
        "conclusions": conclusions,
        "conclusion_version": _conclusion_version(conclusions),
    }
    session["session_version"] = _session_version(session)
    return session


class SnapshotManager:
    def __init__(
        self,
        db_path: str | Path,
        manifest: Manifest,
        snapshot_dir: str | Path,
        *,
        base_db_path: str | Path | None = None,
        verification_config: dict[str, Any] | None = None,
    ):
        self.db_path = Path(db_path)
        self.manifest = manifest
        self.snapshot_dir = Path(snapshot_dir)
        self.base_db_path = Path(base_db_path) if base_db_path is not None else None
        self.verification_config = verification_config or {}

    def save(
        self,
        *,
        name: str = "session.json",
        conclusions: dict[str, Any] | None = None,
        base_db_path: str | Path | None = None,
    ) -> dict[str, Any]:
        connection = connect(self.db_path, readonly=True)
        try:
            session = _current_session_state(
                connection, self.manifest, conclusions=conclusions or {}
            )
        finally:
            connection.close()
        base_path = Path(base_db_path) if base_db_path is not None else self.base_db_path
        base_summary = None
        if base_path is not None:
            base_connection = connect(base_path, readonly=True)
            try:
                base_summary = database_fingerprint(base_connection)
            finally:
                base_connection.close()
        payload = {
            "snapshot_format": SNAPSHOT_FORMAT,
            "session_version": session["session_version"],
            "manifest_id": self.manifest.manifest_id,
            "manifest_sha256": self.manifest.sha256,
            "manifest": self.manifest.raw,
            "manifest_plan": self.manifest.to_dict(),
            "schema_version": session["schema_version"],
            "data_summary": session["data_summary"],
            "total_rows": session["total_rows"],
            "data_version": session["data_version"],
            "column_calibrations": session["column_calibrations"],
            "calibration_version": session["calibration_version"],
            "derivation_graph": session["derivation_graph"],
            "derivation_graph_version": session["derivation_graph_version"],
            "ledger_view": session["ledger_view"],
            "ledger_version": session["ledger_version"],
            "conclusions": session["conclusions"],
            "conclusion_version": session["conclusion_version"],
            "base_database_summary": base_summary,
            "verification_config": self.verification_config,
        }
        write_json_atomic(self.snapshot_dir / name, payload)
        return payload


def _load_snapshot(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise MigrationError(
            "E_MIGRATE", f"cannot load snapshot: {exc}", field_path="$.snapshot"
        ) from exc
    if payload.get("snapshot_format") != SNAPSHOT_FORMAT:
        fail(
            "E_MIGRATE",
            "unsupported snapshot format",
            step_id=None,
            field_path="$.snapshot_format",
        )
    return payload


def _calibration_map(calibrations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        f"{item['table']}.{item['column']}": item for item in calibrations
    }


def _step_map(steps: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(step["step_id"]): step for step in steps}


def _stable_difference(
    snapshot_values: dict[str, Any], current_values: dict[str, Any]
) -> list[str]:
    return sorted(
        key
        for key in set(snapshot_values) | set(current_values)
        if snapshot_values.get(key) != current_values.get(key)
    )


def restore_snapshot(
    snapshot_path: str | Path,
    db_path: str | Path,
    *,
    manifest: Manifest | None = None,
    ledger_dir: str | Path | None = None,
) -> dict[str, Any]:
    snapshot = _load_snapshot(snapshot_path)
    raw_manifest = snapshot.get("manifest")
    plan = snapshot.get("manifest_plan", {})
    if not isinstance(raw_manifest, dict):
        fail(
            "E_MIGRATE",
            "snapshot does not contain a replayable manifest",
            step_id=None,
            field_path="$.manifest",
        )
    reconstructed = manifest_from_plan(
        raw_manifest,
        plan,
        str(snapshot.get("manifest_sha256", "")),
    )
    if manifest is not None and manifest.sha256 != reconstructed.sha256:
        fail(
            "E_MIGRATE",
            "snapshot manifest does not match the supplied manifest",
            step_id=None,
            field_path="$.manifest_sha256",
        )
    active_manifest = manifest or reconstructed
    connection = connect(db_path, readonly=True)
    try:
        current = _current_session_state(connection, active_manifest, conclusions={})
    finally:
        connection.close()
    snapshot_steps = list(snapshot.get("ledger_view", {}).get("steps", []))
    current_steps = list(current["ledger_view"]["steps"])
    snapshot_step_map = _step_map(snapshot_steps)
    current_step_map = _step_map(current_steps)
    affected_steps = _stable_difference(snapshot_step_map, current_step_map)
    snapshot_calibrations = _calibration_map(snapshot.get("column_calibrations", []))
    current_calibrations = _calibration_map(current["column_calibrations"])
    affected_columns = _stable_difference(snapshot_calibrations, current_calibrations)
    snapshot_recal = snapshot.get("ledger_view", {}).get("recalibrations", [])
    current_recal = current["ledger_view"].get("recalibrations", [])
    if snapshot_recal != current_recal:
        for event in sorted(
            [*snapshot_recal, *current_recal], key=lambda item: str(item.get("event_id"))
        ):
            for step_id in event.get("affected_steps", []):
                if step_id not in affected_steps:
                    affected_steps.append(step_id)
            affected_columns.append(f"{event.get('table')}.{event.get('column')}")
            affected_columns.extend(
                str(column) for column in event.get("report", {}).get("affected_columns", [])
            )
        affected_steps.sort()
        affected_columns = sorted(set(affected_columns))
    version_names = [
        "schema_version",
        "data_version",
        "calibration_version",
        "derivation_graph_version",
        "ledger_version",
    ]
    differences = [
        {
            "version": name,
            "snapshot": snapshot.get(name),
            "current": current.get(name),
        }
        for name in version_names
        if snapshot.get(name) != current.get(name)
    ]
    replaced = bool(differences)
    if not replaced:
        state = {
            key: snapshot.get(key)
            for key in [
                "schema_version",
                "data_summary",
                "total_rows",
                "data_version",
                "column_calibrations",
                "calibration_version",
                "derivation_graph",
                "derivation_graph_version",
                "ledger_view",
                "ledger_version",
                "conclusions",
                "conclusion_version",
                "session_version",
            ]
        }
        result = {
            "restored": True,
            "replaced": False,
            "session_version": snapshot.get("session_version"),
            "version_differences": [],
            "affected_columns": [],
            "affected_steps": [],
            "session_state": state,
        }
    else:
        if (
            snapshot.get("data_version") != current.get("data_version")
            and not current_recal
            and not snapshot_recal
        ):
            affected_steps = sorted(set(affected_steps) | set(current_step_map))
            affected_columns = sorted(
                set(affected_columns)
                | set(snapshot_calibrations)
                | set(current_calibrations)
            )
        state = {
            key: current[key]
            for key in [
                "schema_version",
                "data_summary",
                "total_rows",
                "data_version",
                "column_calibrations",
                "calibration_version",
                "derivation_graph",
                "derivation_graph_version",
                "ledger_view",
                "ledger_version",
                "conclusions",
                "conclusion_version",
                "session_version",
            ]
        }
        result = {
            "restored": False,
            "replaced": True,
            "session_version": current["session_version"],
            "version_differences": differences,
            "affected_columns": sorted(set(affected_columns)),
            "affected_steps": sorted(set(affected_steps)),
            "session_state": state,
            "merge_policy": "replace_all",
        }
    if ledger_dir is not None:
        write_json_atomic(
            Path(ledger_dir) / "restored_ledger_view.json",
            result["session_state"]["ledger_view"],
        )
    return result
