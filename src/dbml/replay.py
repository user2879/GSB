from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Sequence

from .database import canonical_json, connect, database_fingerprint
from .errors import MigrationError, fail
from .manifest import Manifest, manifest_from_plan
from .migrations import MigrationRunner, perturb_scale
from .snapshot import _load_snapshot
from .verify import verify_drift


def _replay_fail(
    message: str,
    *,
    step_id: str | None,
    column: str | None,
    field: str | None = None,
    **details: Any,
) -> None:
    raise MigrationError(
        "E_REPLAY",
        message,
        step_id=step_id,
        field_path=(
            f"steps.{step_id}.columns.{column}" if step_id and column else "replay"
        ),
        details={"column": column, "field": field, **details},
    )


def _calibration_maps(step: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    before = {
        f"{item['table']}.{item['column']}": item
        for item in step.get("before_columns", [])
    }
    after = {
        f"{item['table']}.{item['column']}": item
        for item in step.get("after_columns", [])
    }
    return before, after


def _result_column_map(step: dict[str, Any]) -> dict[str, Any]:
    return {
        str(item.get("column")): item
        for item in step.get("result", {}).get("column_errors", [])
    }


def _compare_ledgers(snapshot: dict[str, Any], replay: dict[str, Any]) -> None:
    snapshot_steps = list(snapshot.get("ledger_view", {}).get("steps", []))
    replay_steps = list(replay.get("ledger_view", {}).get("steps", []))
    if len(snapshot_steps) != len(replay_steps):
        index = min(len(snapshot_steps), len(replay_steps))
        expected = (
            snapshot_steps[index].get("step_id") if index < len(snapshot_steps) else None
        )
        actual = replay_steps[index].get("step_id") if index < len(replay_steps) else None
        _replay_fail(
            "replayed step count differs from snapshot",
            step_id=str(expected or actual or "__steps__"),
            column="*",
            field="step_count",
            expected=len(snapshot_steps),
            actual=len(replay_steps),
        )
    for snapshot_step, replay_step in zip(snapshot_steps, replay_steps):
        snapshot_step_id = str(snapshot_step.get("step_id"))
        replay_step_id = str(replay_step.get("step_id"))
        if snapshot_step_id != replay_step_id:
            _replay_fail(
                "replayed step identifier differs from snapshot",
                step_id=snapshot_step_id,
                column="*",
                field="step_id",
                expected=snapshot_step_id,
                actual=replay_step_id,
            )
        for field in ("version", "row_count", "content_digest", "operation", "table"):
            if snapshot_step.get(field) != replay_step.get(field):
                _replay_fail(
                    "replayed step metadata differs from snapshot",
                    step_id=snapshot_step_id,
                    column="*",
                    field=field,
                    expected=snapshot_step.get(field),
                    actual=replay_step.get(field),
                )
        snapshot_before, snapshot_after = _calibration_maps(snapshot_step)
        replay_before, replay_after = _calibration_maps(replay_step)
        for phase, snapshot_map, replay_map in (
            ("before_columns", snapshot_before, replay_before),
            ("after_columns", snapshot_after, replay_after),
        ):
            for column in sorted(set(snapshot_map) | set(replay_map)):
                if snapshot_map.get(column) != replay_map.get(column):
                    _replay_fail(
                        "replayed column calibration differs from snapshot",
                        step_id=snapshot_step_id,
                        column=column,
                        field=phase,
                        expected=snapshot_map.get(column),
                        actual=replay_map.get(column),
                    )
        snapshot_results = _result_column_map(snapshot_step)
        replay_results = _result_column_map(replay_step)
        for column in sorted(set(snapshot_results) | set(replay_results)):
            if snapshot_results.get(column) != replay_results.get(column):
                _replay_fail(
                    "replayed column error report differs from snapshot",
                    step_id=snapshot_step_id,
                    column=column,
                    field="column_errors",
                    expected=snapshot_results.get(column),
                    actual=replay_results.get(column),
                )
        if snapshot_step.get("result") != replay_step.get("result"):
            first_column = sorted(
                set(snapshot_results) | set(replay_results)
                or {"*"}
            )[0]
            _replay_fail(
                "replayed step result differs from snapshot",
                step_id=snapshot_step_id,
                column=first_column,
                field="result",
            )
    snapshot_recal = snapshot.get("ledger_view", {}).get("recalibrations", [])
    replay_recal = replay.get("ledger_view", {}).get("recalibrations", [])
    if snapshot_recal != replay_recal:
        first_snapshot = snapshot_recal[0] if snapshot_recal else {}
        first_replay = replay_recal[0] if replay_recal else {}
        _replay_fail(
            "replayed recalibration ledger differs from snapshot",
            step_id=first_snapshot.get("event_id") or first_replay.get("event_id"),
            column=first_snapshot.get("column") or first_replay.get("column") or "*",
            field="recalibrations",
        )


def _compare_drift_conclusion(
    snapshot: dict[str, Any],
    replay_report: dict[str, Any],
    *,
    last_step_id: str,
) -> None:
    snapshot_drift = snapshot.get("conclusions", {}).get("drift")
    if snapshot_drift is None:
        return
    snapshot_columns = {
        str(item.get("column")): item for item in snapshot_drift.get("columns", [])
    }
    replay_columns = {
        str(item.get("column")): item for item in replay_report.get("columns", [])
    }
    for column in sorted(set(snapshot_columns) | set(replay_columns)):
        if snapshot_columns.get(column) != replay_columns.get(column):
            _replay_fail(
                "replayed drift conclusion differs from snapshot",
                step_id=last_step_id,
                column=column,
                field="conclusions.drift",
                expected=snapshot_columns.get(column),
                actual=replay_columns.get(column),
            )
    for field in ("conclusion", "row_set_mismatch_count"):
        if snapshot_drift.get(field) != replay_report.get(field):
            _replay_fail(
                "replayed drift summary differs from snapshot",
                step_id=last_step_id,
                column="*",
                field=f"conclusions.drift.{field}",
            )


def replay_snapshot(
    snapshot_path: str | Path,
    base_db_path: str | Path,
    work_dir: str | Path,
    *,
    manifest: Manifest | None = None,
    batch_size: int = 256,
) -> dict[str, Any]:
    snapshot = _load_snapshot(snapshot_path)
    raw_manifest = snapshot.get("manifest")
    plan = snapshot.get("manifest_plan", {})
    if not isinstance(raw_manifest, dict):
        _replay_fail(
            "snapshot has no embedded migration manifest",
            step_id="__manifest__",
            column="*",
            field="manifest",
        )
    reconstructed = manifest_from_plan(
        raw_manifest,
        plan,
        str(snapshot.get("manifest_sha256", "")),
    )
    if manifest is not None and manifest.sha256 != reconstructed.sha256:
        _replay_fail(
            "supplied manifest differs from the snapshot manifest",
            step_id="__manifest__",
            column="*",
            field="manifest_sha256",
            expected=snapshot.get("manifest_sha256"),
            actual=manifest.sha256,
        )
    base_path = Path(base_db_path)
    if not base_path.exists():
        _replay_fail(
            "base database is missing",
            step_id="__base__",
            column="*",
            field="base_db_path",
        )
    base_connection = connect(base_path, readonly=True)
    try:
        base_fingerprint = database_fingerprint(base_connection)
    finally:
        base_connection.close()
    expected_base = snapshot.get("base_database_summary")
    if expected_base is not None:
        for field in ("schema_version", "data_version"):
            if expected_base.get(field) != base_fingerprint.get(field):
                _replay_fail(
                    "base database digest differs from snapshot",
                    step_id="__base__",
                    column="*",
                    field=field,
                    expected=expected_base.get(field),
                    actual=base_fingerprint.get(field),
                )
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)
    replay_db = work_path / "replay.sqlite3"
    for suffix in ("", "-wal", "-shm", "-journal"):
        candidate = Path(str(replay_db) + suffix)
        if candidate.exists():
            candidate.unlink()
    shutil.copy2(base_path, replay_db)
    try:
        runner = MigrationRunner(
            replay_db, reconstructed, batch_size=max(1, int(batch_size))
        )
        run_result = runner.run(batch_size=max(1, int(batch_size)))
    except MigrationError as exc:
        raise MigrationError(
            "E_REPLAY",
            f"migration replay failed: {exc.message}",
            step_id=exc.step_id,
            field_path=exc.field_path,
            details={"cause": exc.to_dict()},
        ) from exc
    replayed_manifest = runner.manifest or reconstructed
    for event in sorted(
        snapshot.get("ledger_view", {}).get("recalibrations", []),
        key=lambda item: int(item.get("seq", 0)),
    ):
        perturb_scale(
            replay_db,
            replayed_manifest,
            table=str(event["table"]),
            column=str(event["column"]),
            new_scale=int(event["new_scale"]),
            new_truncation=str(event["new_truncation"]),
            batch_size=max(1, int(batch_size)),
        )
    replay_connection = connect(replay_db, readonly=True)
    try:
        from .ledger import ledger_view

        replay_ledger = ledger_view(replay_connection)
    finally:
        replay_connection.close()
    replay_bundle = {"ledger_view": replay_ledger}
    _compare_ledgers(snapshot, replay_bundle)
    drift_report = None
    verification_config = snapshot.get("verification_config", {})
    if snapshot.get("conclusions", {}).get("drift") is not None:
        drift_report = verify_drift(
            base_path,
            replay_db,
            table=str(verification_config.get("table", "orders")),
            columns=verification_config.get("columns"),
            before_calibrations=replayed_manifest,
            row_key=verification_config.get("row_key"),
            batch_size=max(1, int(verification_config.get("batch_size", batch_size))),
            raise_on_drift=False,
        )
        last_step_id = (
            str(replayed_manifest.execution_order[-1])
            if replayed_manifest.execution_order
            else "__conclusion__"
        )
        _compare_drift_conclusion(
            snapshot, drift_report, last_step_id=last_step_id
        )
    replay_connection = connect(replay_db, readonly=True)
    try:
        replay_fingerprint = database_fingerprint(replay_connection)
    finally:
        replay_connection.close()
    return {
        "reproducible": True,
        "manifest_id": reconstructed.manifest_id,
        "manifest_sha256": reconstructed.sha256,
        "snapshot_session_version": snapshot.get("session_version"),
        "replayed_data_version": replay_fingerprint["data_version"],
        "replayed_ledger_version": replay_ledger["ledger_version"],
        "steps_compared": len(replay_ledger["steps"]),
        "columns_compared_per_step": True,
        "drift": drift_report,
        "replay_database": str(replay_db),
        "run": run_result.to_dict(),
    }
