from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from .engine import ConfigEngine
from .errors import DiagnosticError
from .structured_log import JsonlLogger
from .validation import load_bundle, load_config_spec


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _record_diagnostic(logger: JsonlLogger, kind: str, error: DiagnosticError) -> None:
    logger.emit(
        error.code,
        f"injection_{kind}",
        severity="ERROR",
        message=error.detail,
        injection=kind,
        key=error.key,
        field=error.field,
        diagnostic_context=error.context,
    )


def inject_spec_corruption(samples_dir: Path, root: Path, logger: JsonlLogger) -> dict[str, Any]:
    source = json.loads((samples_dir / "config_keys.json").read_text(encoding="utf-8"))
    source["keys"][0]["ttl_ms"] = 0
    target = root / "config_keys.corrupt.json"
    _write_json(target, source)
    try:
        load_config_spec(target)
    except DiagnosticError as exc:
        _record_diagnostic(logger, "spec", exc)
        return {
            "ok": True,
            "injected": "spec",
            "code": "E_SPEC",
            "expected_code": "E_SPEC",
            "diagnostic": exc.to_dict(),
            "artifact": str(target),
            "structured_log": str(logger.path),
        }
    raise AssertionError("spec corruption injection unexpectedly validated")


def inject_cycle(samples_dir: Path, root: Path, logger: JsonlLogger) -> dict[str, Any]:
    source = json.loads((samples_dir / "config_keys.json").read_text(encoding="utf-8"))
    by_key = {item["key"]: item for item in source["keys"]}
    first = "pricing.net_fee"
    second = "tax.amount"
    by_key[first]["refs"] = [second]
    by_key[first]["formula"] = {
        "op": "linear",
        "terms": [{"ref": second, "weight": 1000}],
        "divisor": 1000,
        "bias": 0,
    }
    by_key[second]["refs"] = [first]
    by_key[second]["formula"] = {
        "op": "linear",
        "terms": [{"ref": first, "weight": 1000}],
        "divisor": 1000,
        "bias": 0,
    }
    target = root / "config_keys.cycle.json"
    _write_json(target, source)
    try:
        load_config_spec(target)
    except DiagnosticError as exc:
        _record_diagnostic(logger, "cycle", exc)
        return {
            "ok": True,
            "injected": "cycle",
            "code": "E_CYCLE",
            "expected_code": "E_CYCLE",
            "diagnostic": exc.to_dict(),
            "artifact": str(target),
            "structured_log": str(logger.path),
        }
    raise AssertionError("cycle injection unexpectedly validated")


def inject_snapshot_mismatch(
    samples_dir: Path,
    root: Path,
    logger: JsonlLogger,
) -> dict[str, Any]:
    spec, clients, _ = load_bundle(samples_dir)
    db_path = root / "runtime" / "injected.sqlite3"
    engine = ConfigEngine(root, db_path=db_path, logger=logger)
    try:
        engine.initialize(spec, clients)
    finally:
        engine.close()

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("UPDATE ledger_meta SET content_digest = ? WHERE id = 1", ("0" * 64,))
        conn.commit()
    finally:
        conn.close()

    reopened = ConfigEngine(root, db_path=db_path, logger=logger)
    try:
        event = None
        for candidate in reopened.open_events:
            if candidate.get("code") == "E_SNAPSHOT":
                event = candidate
                break
        if event is None:
            raise AssertionError("snapshot mismatch was not detected")
        backup = Path(event["backup"])
        if not backup.exists():
            raise AssertionError("read-only backup was not created")
        readonly = (os.stat(backup).st_mode & 0o222) == 0
        return {
            "ok": True,
            "injected": "snapshot",
            "code": "E_SNAPSHOT",
            "expected_code": "E_SNAPSHOT",
            "diagnostic": {
                "ok": False,
                "code": "E_SNAPSHOT",
                "detail": event["message"],
                "field": "ledger_meta.content_digest",
                "context": {
                    "mismatch": event["mismatch"],
                    "backup": event["backup"],
                    "snapshot_id": event["snapshot_id"],
                    "mutation_mode": event["mutation_mode"],
                },
            },
            "recovered_ledger_version": reopened.ledger.meta()["ledger_version"],
            "content_digest": reopened.ledger.meta()["content_digest"],
            "backup_readonly": readonly,
            "artifact": str(db_path),
            "structured_log": str(logger.path),
        }
    finally:
        reopened.close()


def run_injection(kind: str, samples_dir: Path | str, workspace: Path | str) -> dict[str, Any]:
    samples = Path(samples_dir).resolve()
    workspace_path = Path(workspace).resolve()
    root = workspace_path / "runtime" / "injections" / f"{kind}-{time.monotonic_ns()}"
    root.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(workspace_path / "logs" / "diagnostics.jsonl")
    if kind == "spec":
        return inject_spec_corruption(samples, root, logger)
    if kind == "cycle":
        return inject_cycle(samples, root, logger)
    if kind == "snapshot":
        return inject_snapshot_mismatch(samples, root, logger)
    raise DiagnosticError("E_STATE", "未知异常注入类型", field="kind", context={"kind": kind})
