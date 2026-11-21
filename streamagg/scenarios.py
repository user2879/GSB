from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import digest_obj
from .engine import Engine
from .errors import BusinessError
from .generator import generate_session
from .ledger import Ledger
from .logging_utils import JsonlLogger
from .replay import validate_replay
from .snapshot import SnapshotStore, restore_engine
from .spec import load_spec, parse_spec


@dataclass(frozen=True)
class DemoPaths:
    workspace: Path
    demo_root: Path
    ledger: Path
    snapshots: Path
    logs: Path
    reports: Path
    specs: Path

    @classmethod
    def from_values(
        cls,
        workspace: str | Path,
        demo_root: str | Path,
        ledger: str | Path | None = None,
        snapshots: str | Path | None = None,
        logs: str | Path | None = None,
        reports: str | Path | None = None,
    ) -> "DemoPaths":
        workspace_path = Path(workspace).resolve()
        root = Path(demo_root).resolve()
        return cls(
            workspace=workspace_path,
            demo_root=root,
            ledger=Path(ledger).resolve() if ledger else root / "ledger.sqlite3",
            snapshots=Path(snapshots).resolve() if snapshots else root / "snapshots",
            logs=Path(logs).resolve() if logs else root / "logs",
            reports=Path(reports).resolve() if reports else root / "reports",
            specs=workspace_path / "samples",
        )

    def prepare(self) -> None:
        for directory in (self.demo_root, self.snapshots, self.logs, self.reports):
            directory.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _expected_order_aggregates(
    spec: Any,
    events: list[Any],
    watermarks: dict[str, int],
) -> dict[tuple[Any, ...], tuple[int, int, int]]:
    from .engine import floor_window_start

    result: dict[tuple[Any, ...], tuple[int, int, int]] = {}
    for rule in spec.rules:
        for event in events:
            if event.partition_key not in rule.partition_keys:
                continue
            start = floor_window_start(event.event_time_ms, rule.width_ms)
            key = (rule.rule_id, event.partition_key, start, start + rule.width_ms)
            count, total = result.get(key, (0, 0, 0))[:2]
            result[key] = (count + 1, total + event.measure * spec.fixed_scale, 0)
    finalized: dict[tuple[Any, ...], tuple[int, int, int]] = {}
    for key, (count, total, _) in result.items():
        rule = next(rule for rule in spec.rules if rule.rule_id == key[0])
        partition_watermark = watermarks.get(key[1])
        if partition_watermark is None or key[3] > partition_watermark - rule.lateness_ms:
            continue
        finalized[key] = (count, total, int(total / count) if total >= 0 else -((-total) // count))
    return finalized


def scenario1(paths: DemoPaths) -> dict[str, Any]:
    logger = JsonlLogger(paths.logs, "scenario1")
    logger.info("OK", "scenario_start", "ordered aggregation scenario started")
    spec = load_spec(paths.specs / "scenario1_ordered.json")
    events_a, generations_a = generate_session(spec)
    events_b, generations_b = generate_session(spec)
    source_id = spec.sources[0].source_id
    if generations_a[source_id].digest != generations_b[source_id].digest:
        raise BusinessError(
            code="E_SPEC",
            message="source generation digest is not reproducible",
            field=f"sources.{source_id}.seed",
        )
    ordered_events = sorted(events_a, key=lambda event: (event.arrival_no, event.event_id))
    with Ledger(paths.ledger) as ledger:
        engine = Engine(
            spec,
            session_id="scenario1-ordered",
            ledger=ledger,
            logger=logger,
        )
        summary = engine.run_events(ordered_events)
        replay_report = validate_replay(engine, ledger)
        current_windows = ledger.current_windows(engine.session_id, engine.view_id)
        attributions = {
            (row["rule_id"], row["partition_key"], row["window_start_ms"], row["window_end_ms"]): dict(row)
            for row in ledger.current_attributions(engine.session_id, engine.view_id)
        }
        if summary["drop_count"] != 0:
            raise BusinessError(
                code="E_REPLAY",
                message="ordered scenario unexpectedly produced dropped events",
                details={"drop_count": summary["drop_count"]},
            )
        if not summary["finalized_windows"]:
            raise BusinessError(
                code="E_REPLAY",
                message="ordered scenario produced no finalized windows",
            )
        expected = _expected_order_aggregates(spec, events_a, summary["watermarks"])
        actual = {
            (
                row["rule_id"],
                row["partition_key"],
                row["window_start_ms"],
                row["window_end_ms"],
            ): (row["count"], row["sum_fixed"], row["mean_fixed"])
            for row in current_windows
        }
        if set(actual) != set(expected):
            missing = sorted(set(expected) - set(actual))[:3]
            extra = sorted(set(actual) - set(expected))[:3]
            raise BusinessError(
                code="E_REPLAY",
                message="ordered window set differs from independent event-time truth",
                details={"missing": missing, "extra": extra},
            )
        for key, expected_aggregate in expected.items():
            if actual[key] != expected_aggregate:
                raise BusinessError(
                    code="E_REPLAY",
                    message="ordered fixed-point aggregate differs from independent truth",
                    details={"window": key, "expected": expected_aggregate, "actual": actual[key]},
                )
        for row in attributions.values():
            if row["sum_bound"] < 0 or row["mean_bound"] < 0:
                raise BusinessError(
                    code="E_REPLAY",
                    message="error bound is negative",
                    details={"window": [row["partition_key"], row["window_start_ms"]]},
                )
        report_windows = []
        for row in current_windows:
            payload = dict(row)
            payload["attribution"] = attributions[
                (
                    row["rule_id"],
                    row["partition_key"],
                    row["window_start_ms"],
                    row["window_end_ms"],
                )
            ]
            report_windows.append(payload)
        report = {
            "business_code": "OK_SCENARIO_1_ORDERED",
            "scenario": 1,
            "source_digest": generations_a[source_id].digest,
            "digest_reproducible": True,
            "event_count": len(events_a),
            "finalized_window_count": len(current_windows),
            "drop_count": summary["drop_count"],
            "watermarks": summary["watermarks"],
            "fixed_scale": spec.fixed_scale,
            "mean_rounding": "toward_zero",
            "overflow_rule": "saturate_signed_int64",
            "replay": replay_report,
            "windows": report_windows,
        }
        _write_json(paths.reports / "scenario1.json", report)
        logger.info(
            "OK",
            "scenario_complete",
            "ordered aggregation scenario completed",
            {"finalized_window_count": len(current_windows), "event_count": len(events_a)},
        )
        return report


def scenario2(paths: DemoPaths) -> dict[str, Any]:
    logger = JsonlLogger(paths.logs, "scenario2")
    logger.info("OK", "scenario_start", "late-budget scenario started")
    spec = load_spec(paths.specs / "sample_spec.json")
    with Ledger(paths.ledger) as ledger:
        engine = Engine(
            spec,
            session_id="scenario2-late-budget",
            ledger=ledger,
            logger=logger,
        )
        summary = engine.run_all()
        replay_report = validate_replay(engine, ledger)
        reasons: dict[str, int] = {}
        for decision in summary["drop_decisions"]:
            reasons[decision["reason"]] = reasons.get(decision["reason"], 0) + 1
        required_reasons = {
            "LATE_BUDGET_COUNT_EXHAUSTED",
            "LATE_BUDGET_AMOUNT_EXCEEDED",
            "WINDOW_ALREADY_FINALIZED",
        }
        missing_reasons = sorted(required_reasons - set(reasons))
        if missing_reasons:
            raise BusinessError(
                code="E_REPLAY",
                message="late-budget scenario did not exercise all required drop reasons",
                details={"missing_reasons": missing_reasons, "actual": reasons},
            )
        if summary["late_accept_count"] <= 0:
            raise BusinessError(
                code="E_REPLAY",
                message="late-budget scenario produced no accepted late event",
            )
        attributed = [
            window
            for window in summary["finalized_windows"]
            if window["attribution"]["sum_bound"] > 0
        ]
        if not attributed:
            raise BusinessError(
                code="E_REPLAY",
                message="late-budget scenario produced no non-zero error attribution",
            )
        if not any(window["attribution"]["dominant_segment"] != "NONE" for window in attributed):
            raise BusinessError(
                code="E_REPLAY",
                message="late-budget scenario produced no dominant error segment",
            )
        report = {
            "business_code": "OK_SCENARIO_2_LATE_BUDGET",
            "scenario": 2,
            "event_count": summary["event_cursor"],
            "finalized_window_count": len(summary["finalized_windows"]),
            "drop_count": summary["drop_count"],
            "drop_reasons": reasons,
            "late_accept_count": summary["late_accept_count"],
            "late_accept_reasons": _count_reasons(summary["late_accept_decisions"]),
            "ambiguity_pair_count": len(summary["ambiguity_pairs"]),
            "ambiguity_groups": summary["ambiguity_groups"],
            "dominant_segments": _count_dominant(summary["finalized_windows"]),
            "open_window_count": summary["open_window_count"],
            "watermarks": summary["watermarks"],
            "replay": replay_report,
            "sample_drops": summary["drop_decisions"][:20],
            "sample_late_accepts": summary["late_accept_decisions"][:20],
            "windows": summary["finalized_windows"],
        }
        _write_json(paths.reports / "scenario2.json", report)
        logger.info(
            "OK",
            "scenario_complete",
            "late-budget scenario completed",
            {
                "drop_count": summary["drop_count"],
                "late_accept_count": summary["late_accept_count"],
                "reasons": reasons,
            },
        )
        return report


def _count_reasons(decisions: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for decision in decisions:
        reason = decision["reason"]
        result[reason] = result.get(reason, 0) + 1
    return result


def _count_dominant(windows: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for window in windows:
        segment = window["attribution"]["dominant_segment"]
        result[segment] = result.get(segment, 0) + 1
    return result


def prepare_scenario3_crash(paths: DemoPaths, metadata_path: Path) -> dict[str, Any]:
    logger = JsonlLogger(paths.logs, "scenario3")
    logger.info("OK", "crash_writer_start", "preparing a mid-stream snapshot before abrupt termination")
    store = SnapshotStore(paths.snapshots)
    initial_spec = load_spec(paths.specs / "scenario3_initial.json")
    evolved_spec = load_spec(paths.specs / "scenario3_evolved.json")
    ledger = Ledger(paths.ledger)
    engine = Engine(
        initial_spec,
        session_id="scenario3-replace-restore",
        ledger=ledger,
        logger=logger,
    )
    version1_summary = engine.run_all()
    version1_view = engine.view_id
    version1_manifest = dict(ledger.get_view_manifest(engine.session_id, version1_view))
    snapshot_v1 = store.save(engine, snapshot_id="scenario3-v1")
    evolution = engine.evolve_spec(
        evolved_spec,
        reason="width, lateness, fixed scale, and partition key set changed",
    )
    all_events, _ = generate_session(engine.spec)
    sorted_events = sorted(all_events, key=lambda event: (event.arrival_no, event.event_id))
    partial_cursor = min(90, len(sorted_events))
    partial_summary = engine.run_events(sorted_events[:partial_cursor])
    version2_view = engine.view_id
    version2_manifest = dict(ledger.get_view_manifest(engine.session_id, version2_view))
    snapshot_v2 = store.save(engine, snapshot_id="scenario3-v2-mid")
    pending_before = ledger.connection.execute(
        "SELECT COUNT(*) AS c FROM window_state WHERE session_id = ? AND view_id = ? AND status = 'PENDING' AND stale = 0",
        (engine.session_id, version2_view),
    ).fetchone()["c"]
    metadata = {
        "session_id": engine.session_id,
        "version1": {
            "view_id": version1_view,
            "manifest": version1_manifest,
            "event_count": version1_summary["event_cursor"],
            "finalized_window_count": len(version1_summary["finalized_windows"]),
            "snapshot": str(snapshot_v1.path),
        },
        "evolution": evolution,
        "version2": {
            "view_id": version2_view,
            "manifest": version2_manifest,
            "event_cursor_before_crash": partial_cursor,
            "finalized_window_count": len(partial_summary["finalized_windows"]),
            "snapshot": str(snapshot_v2.path),
            "pending_windows_before": pending_before,
        },
    }
    _write_json(metadata_path, metadata)
    logger.info(
        "OK",
        "crash_writer_ready_for_abrupt_exit",
        "snapshot and ledger durable; process will terminate without Python cleanup",
        {"pending_windows": pending_before, "event_cursor": engine.event_cursor},
    )
    return metadata


def scenario3(paths: DemoPaths) -> dict[str, Any]:
    logger = JsonlLogger(paths.logs, "scenario3")
    logger.info("OK", "scenario_start", "replacement, crash recovery, and replay scenario started")
    metadata_path = paths.reports / "scenario3_crash_writer.json"
    command = [
        sys.executable,
        "-m",
        "streamagg.cli",
        "crash-writer",
        "--workspace",
        str(paths.workspace),
        "--demo-root",
        str(paths.demo_root),
        "--ledger",
        str(paths.ledger),
        "--snapshots",
        str(paths.snapshots),
        "--logs",
        str(paths.logs),
        "--reports",
        str(paths.reports),
        "--metadata",
        str(metadata_path),
    ]
    environment = os.environ.copy()
    existing_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(paths.workspace) + (
        os.pathsep + existing_path if existing_path else ""
    )
    process = subprocess.run(
        command,
        cwd=paths.workspace,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    (paths.reports / "scenario3_crash_writer.stdout.log").write_text(process.stdout, encoding="utf-8")
    (paths.reports / "scenario3_crash_writer.stderr.log").write_text(process.stderr, encoding="utf-8")
    if process.returncode != 17:
        raise BusinessError(
            code="E_SNAPSHOT",
            message="crash writer did not terminate with the expected forced exit code",
            details={"expected_exit_code": 17, "actual_exit_code": process.returncode},
        )
    if not metadata_path.exists():
        raise BusinessError(
            code="E_SNAPSHOT",
            message="crash writer did not persist recovery metadata",
            details={"path": str(metadata_path)},
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    version1 = metadata["version1"]
    version2 = metadata["version2"]
    version1_view = version1["view_id"]
    version2_view = version2["view_id"]
    evolution = metadata["evolution"]
    store = SnapshotStore(paths.snapshots)
    ledger = Ledger(paths.ledger)
    try:
        if engine_rule_version_from_ledger(ledger, metadata["session_id"], version2_view) <= version1["manifest"]["rules_version"]:
            raise BusinessError(
                code="E_SPEC",
                message="rules version did not increase monotonically",
            )
        if not evolution["affected_windows"]:
            raise BusinessError(code="E_SPEC", message="evolution did not identify affected windows")
        stale_window_rows = ledger.all_windows(
            metadata["session_id"], version1_view, include_stale=True
        )
        if not stale_window_rows or any(row["stale"] != 1 for row in stale_window_rows):
            raise BusinessError(code="E_SPEC", message="old view windows were not marked stale")
        if ledger.current_windows(metadata["session_id"], version1_view):
            raise BusinessError(code="E_SPEC", message="stale view leaked into the current window view")

        restored_engine, loaded_v2, unknown_rows = restore_engine(
            version2["snapshot"],
            store=store,
            ledger=ledger,
            current_manifest=version2["manifest"],
            logger=logger,
        )
        if not unknown_rows:
            raise BusinessError(
                code="E_SNAPSHOT",
                message="restore did not mark any pending window unknown",
            )
        if restored_engine.spec.rules_version != evolution["rules_version"]:
            raise BusinessError(
                code="E_SNAPSHOT",
                message="restored rule version differs from the replacement view",
            )
        replay_report = validate_replay(restored_engine, ledger)

        loaded_v1 = store.load(version1["snapshot"], current_manifest=version2["manifest"])
        if loaded_v1.comparison is None or not loaded_v1.comparison.replacement_required:
            raise BusinessError(
                code="E_SNAPSHOT",
                message="old snapshot was not identified as requiring wholesale replacement",
            )
        isolated_v1, _ = Engine.from_snapshot_state(
            loaded_v1.state,
            ledger=None,
            logger=None,
            mark_pending_unknown=False,
        )
        if isolated_v1.spec.rules_version != 1 or isolated_v1.view_id != version1_view:
            raise BusinessError(
                code="E_SNAPSHOT",
                message="snapshot restore did not replace rules and view wholesale",
            )
        if any(key.rule_version != 1 for key in isolated_v1.finalized_windows):
            raise BusinessError(
                code="E_SNAPSHOT",
                message="snapshot restore merged conclusions from another rule version",
            )
        unknown_after = ledger.connection.execute(
            "SELECT COUNT(*) AS c FROM window_state WHERE session_id = ? AND view_id = ? AND status = 'UNKNOWN' AND stale = 0",
            (restored_engine.session_id, restored_engine.view_id),
        ).fetchone()["c"]
        report = {
            "business_code": "OK_SCENARIO_3_REPLACE_SNAPSHOT_REPLAY",
            "scenario": 3,
            "version1": version1,
            "evolution": evolution,
            "version2": version2,
            "forced_termination": {
                "simulated": False,
                "method": "os._exit_before_python_cleanup",
                "crash_writer_exit_code": process.returncode,
                "pending_windows_before": version2["pending_windows_before"],
                "unknown_windows_after_restore": unknown_after,
                "unknown_window_rows": unknown_rows,
            },
            "replacement_comparison": loaded_v1.comparison.to_dict(),
            "replacement_isolated_rules_version": isolated_v1.spec.rules_version,
            "restored_rules_version": restored_engine.spec.rules_version,
            "replay": replay_report,
            "ledger_batch_digest": loaded_v2.manifest["ledger_content_digest"],
            "fixed_scale": restored_engine.spec.fixed_scale,
        }
        _write_json(paths.reports / "scenario3.json", report)
        logger.info(
            "OK",
            "scenario_complete",
            "replacement, crash recovery, and replay scenario completed",
            {
                "old_rules_version": version1["manifest"]["rules_version"],
                "new_rules_version": version2["manifest"]["rules_version"],
                "crash_writer_exit_code": process.returncode,
                "unknown_windows": unknown_after,
                "replay_consistent": replay_report["consistent"],
            },
        )
        return report
    finally:
        ledger.close()


def engine_rule_version_from_ledger(ledger: Ledger, session_id: str, view_id: str) -> int:
    return int(ledger.get_view_manifest(session_id, view_id)["rules_version"])


def run_scenario(number: int, paths: DemoPaths) -> dict[str, Any]:
    paths.prepare()
    if number == 1:
        return scenario1(paths)
    if number == 2:
        return scenario2(paths)
    if number == 3:
        return scenario3(paths)
    raise BusinessError(
        code="E_SPEC",
        message="scenario number must be 1, 2, or 3",
        field="scenario",
    )


def inject_order(paths: DemoPaths) -> dict[str, Any]:
    logger = JsonlLogger(paths.logs, "injection_order")
    spec = load_spec(paths.specs / "scenario1_ordered.json")
    engine = Engine(spec, session_id="inject-order", ledger=None, logger=logger)
    events, _ = generate_session(spec)
    first = sorted(events, key=lambda event: event.arrival_no)[0]
    engine.ingest(first)
    from .model import Event

    regressed = Event(
        event_id="inject:order-regression",
        source_id=first.source_id,
        source_seq=10**9,
        arrival_no=10**9,
        partition_key=first.partition_key,
        event_time_ms=engine.watermarks[first.partition_key] - spec.order_backward_limit_ms - 1,
        measure=1,
    )
    engine.ingest(regressed)
    raise BusinessError(code="E_ORDER", message="order injection did not trigger E_ORDER")


def inject_grid(paths: DemoPaths) -> dict[str, Any]:
    raw = json.loads((paths.specs / "scenario1_ordered.json").read_text(encoding="utf-8"))
    raw["session"]["fixed_scale"] = 0
    injected_path = paths.demo_root / "injected" / "invalid_scale.json"
    _write_json(injected_path, raw)
    parse_spec(raw)
    raise BusinessError(code="E_GRID", message="grid injection did not trigger E_GRID")


def inject_snapshot(paths: DemoPaths) -> dict[str, Any]:
    logger = JsonlLogger(paths.logs, "injection_snapshot")
    store = SnapshotStore(paths.snapshots)
    spec = load_spec(paths.specs / "scenario1_ordered.json")
    with Ledger(paths.ledger) as ledger:
        engine = Engine(
            spec,
            session_id="inject-snapshot",
            ledger=ledger,
            logger=logger,
        )
        engine.run_all()
        saved = store.save(engine, snapshot_id="inject-snapshot-tampered")
        ledger.connection.execute(
            "UPDATE ledger_batches SET content_digest = ? WHERE batch_id = (SELECT batch_id FROM ledger_batches WHERE session_id = ? AND view_id = ? ORDER BY batch_no LIMIT 1)",
            ("0" * 64, engine.session_id, engine.view_id),
        )
        ledger.connection.commit()
        ledger.verify_content_digest(engine.session_id, engine.view_id)
    raise BusinessError(code="E_SNAPSHOT", message="snapshot injection did not trigger E_SNAPSHOT")


def inject_replay(paths: DemoPaths) -> dict[str, Any]:
    logger = JsonlLogger(paths.logs, "injection_replay")
    spec = load_spec(paths.specs / "scenario1_ordered.json")
    with Ledger(paths.ledger) as ledger:
        engine = Engine(
            spec,
            session_id="inject-replay",
            ledger=ledger,
            logger=logger,
        )
        engine.run_all()
        row = ledger.connection.execute(
            "SELECT rule_id, partition_key, window_start_ms, window_end_ms FROM window_state WHERE session_id = ? AND view_id = ? AND status = 'COMMITTED' AND stale = 0 ORDER BY partition_key, window_start_ms LIMIT 1",
            (engine.session_id, engine.view_id),
        ).fetchone()
        if row is None:
            raise BusinessError(code="E_REPLAY", message="no window available for replay injection")
        ledger.connection.execute(
            "UPDATE window_state SET sum_fixed = sum_fixed + 1 WHERE session_id = ? AND view_id = ? AND rule_id = ? AND partition_key = ? AND window_start_ms = ? AND window_end_ms = ?",
            (
                engine.session_id,
                engine.view_id,
                row["rule_id"],
                row["partition_key"],
                row["window_start_ms"],
                row["window_end_ms"],
            ),
        )
        ledger.connection.commit()
        validate_replay(engine, ledger)
    raise BusinessError(code="E_REPLAY", message="replay injection did not trigger E_REPLAY")
