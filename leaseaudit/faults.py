from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .clock import ManualClock
from .engine import LeaseEngine
from .errors import BusinessError, ErrorCode
from .ledger import AuditLedger
from .logging_utils import StructuredLogger
from .scenario import load_scenario
from .util import read_json_strict, write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
DEMO_SCENE = ROOT / "scenarios" / "demo_scene.json"


def _fault_scenario_document() -> dict:
    return {
        "format_version": 1,
        "scenario_version": "fault-base-v1",
        "lease_params_version": 1,
        "cluster_capacity": 8,
        "lease_duration_ms": 5000,
        "renew_interval_ms": 100,
        "reclaim_threshold": 250000,
        "reclaim_window_rounds": 2,
        "observation_window_ms": 100,
        "watermark_scale": 1000000,
        "load_duration_ms": 3000,
        "sources": [
            {
                "source_id": "fault-a",
                "address": "127.0.0.1",
                "port": 55701,
                "lease_share": 3,
                "reserve_quota": 1,
                "mode": "success",
                "processing_ms": 0,
                "request_interval_ms": 100,
            },
            {
                "source_id": "fault-b",
                "address": "127.0.0.1",
                "port": 55702,
                "lease_share": 3,
                "reserve_quota": 1,
                "mode": "success",
                "processing_ms": 0,
                "request_interval_ms": 100,
            },
        ],
    }


def _build_base_ledger(workdir: Path) -> tuple[Path, Path]:
    scenario_path = workdir / "fault-base-scenario.json"
    write_json_atomic(scenario_path, _fault_scenario_document())
    scenario = load_scenario(scenario_path)
    ledger_path = workdir / "fault-base.sqlite"
    for suffix in ("", "-wal", "-shm"):
        path = Path(str(ledger_path) + suffix)
        if path.exists():
            path.unlink()
    ledger = AuditLedger(ledger_path, scenario=scenario, verify=True)
    engine = LeaseEngine(scenario, ledger, clock=ManualClock(value_ms=1_000_000))
    try:
        lease = engine.issue("fault-a", 2, now_ms=1_000_010)
        engine.renew(lease.lease_id, "fault-a", now_ms=1_000_050)
        engine.observe_all(now_ms=1_000_100)
    finally:
        engine.close()
    return scenario_path, ledger_path


def _inject_scene_corrupt(workdir: Path) -> tuple[Path, BusinessError]:
    document = read_json_strict(DEMO_SCENE)
    document["sources"][0]["address"] = "0.0.0.0"
    scene_path = workdir / "fault-scene-corrupt.json"
    write_json_atomic(scene_path, document)
    try:
        load_scenario(scene_path)
    except BusinessError as exc:
        return scene_path, exc
    raise AssertionError("scene corruption did not fail validation")


def _inject_quota_overflow(workdir: Path) -> tuple[Path, BusinessError]:
    document = _fault_scenario_document()
    for source in document["sources"]:
        source["lease_share"] = 5
        source["reserve_quota"] = 5
    scene_path = workdir / "fault-quota-overflow.json"
    write_json_atomic(scene_path, document)
    try:
        load_scenario(scene_path)
    except BusinessError as exc:
        return scene_path, exc
    raise AssertionError("quota overflow did not fail validation")


def _inject_ledger_digest(workdir: Path) -> tuple[Path, BusinessError]:
    _scenario_path, base_ledger = _build_base_ledger(workdir)
    faulty = workdir / "fault-ledger-digest.sqlite"
    for suffix in ("", "-wal", "-shm"):
        path = Path(str(faulty) + suffix)
        if path.exists():
            path.unlink()
    source = sqlite3.connect(base_ledger)
    target = sqlite3.connect(faulty)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()
    conn = sqlite3.connect(faulty)
    try:
        row = conn.execute("SELECT seq, payload_json FROM events ORDER BY seq LIMIT 1").fetchone()
        if row is None:
            raise AssertionError("base ledger has no event to corrupt")
        payload = json.loads(row[1])
        payload["tamper_marker"] = "ledger-content-digest-injection"
        payload["basis"] = payload.get("basis", "") + ";tampered"
        conn.execute(
            "UPDATE events SET payload_json=? WHERE seq=?",
            (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")), row[0]),
        )
        conn.commit()
    finally:
        conn.close()
    try:
        ledger = AuditLedger(faulty, readonly=True, verify=True)
        ledger.close()
    except BusinessError as exc:
        return faulty, exc
    raise AssertionError("ledger digest corruption was not detected")


def run_fault_injection(workdir: str | Path, kind: str) -> dict:
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    logger = StructuredLogger(workdir / "structured.jsonl", echo=False)
    if kind == "scene_corrupt":
        artifact, error = _inject_scene_corrupt(workdir)
    elif kind == "quota_overflow":
        artifact, error = _inject_quota_overflow(workdir)
    elif kind == "ledger_digest":
        artifact, error = _inject_ledger_digest(workdir)
    else:
        raise BusinessError(
            ErrorCode.E_ENV,
            "unknown fault kind",
            field_path="--kind",
            details={"allowed": ["scene_corrupt", "quota_overflow", "ledger_digest"]},
        )
    logger.info(
        "FAULT_INJECTED",
        "fault artifact written without modifying source code",
        kind=kind,
        artifact=str(artifact),
    )
    logger.error(error)
    expected = {
        "scene_corrupt": ErrorCode.E_SCENE,
        "quota_overflow": ErrorCode.E_QUOTA,
        "ledger_digest": ErrorCode.E_SNAPSHOT,
    }[kind]
    result = {
        "fault_kind": kind,
        "status": "EXPECTED_FAILURE" if error.code == expected else "UNEXPECTED_FAILURE",
        "business_code": error.code.value,
        "expected_code": expected.value,
        "field_path": error.field_path,
        "message": error.message,
        "details": error.details,
        "artifact": str(artifact),
        "source_code_modified": False,
        "partially_written_ledger": False,
        "log_path": str(workdir / "structured.jsonl"),
    }
    write_json_atomic(workdir / f"fault-{kind}.json", result)
    return result
