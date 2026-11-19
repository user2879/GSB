from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from audit_forensics.canonical import canonical_json
from audit_forensics.engine import AuditEngine
from audit_forensics.rules import RuleConfig


ROOT = Path(__file__).resolve().parents[1]
BASE_TS = 1_795_046_400_000


def event(
    event_id: str,
    *,
    trace_id: str = "T-1",
    actor: str = "alice",
    object_name: str = "vault",
    action: str = "read",
    ts: int = BASE_TS,
    seq: int = 1,
    parent_event_id: str | None = None,
    severity: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "event_id": event_id,
        "trace_id": trace_id,
        "actor": actor,
        "object": object_name,
        "action": action,
        "ts": ts,
        "seq": seq,
    }
    if parent_event_id is not None:
        value["parent_event_id"] = parent_event_id
    if severity is not None:
        value["severity"] = severity
    if source is not None:
        value["source"] = source
    return value


def write_v1(
    path: Path,
    events: Iterable[dict[str, Any]],
    *,
    malformed_lines: Iterable[str] = (),
) -> Path:
    lines = [json.dumps(item, separators=(",", ":")) for item in events]
    lines.extend(malformed_lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_v2(path: Path, events: Iterable[Any]) -> Path:
    path.write_text(
        json.dumps(list(events), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def rules_config(
    version: str = "1",
    rules: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {"rules_version": version, "rules": rules or []}


def write_rules(path: Path, value: dict[str, Any]) -> Path:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def import_workspace(
    workspace: Path,
    files: Iterable[Path],
    rules: dict[str, Any] | None = None,
    *,
    workers: int = 1,
):
    engine = AuditEngine(
        workspace,
        rules=RuleConfig.from_dict(rules) if rules is not None else None,
    )
    return engine.import_streams([str(path) for path in files], workers=workers)


def run_cli(
    *args: str,
    env: dict[str, str] | None = None,
    timeout: float = 20,
) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    merged["PYTHONPATH"] = str(ROOT) + os.pathsep + merged.get("PYTHONPATH", "")
    if env:
        merged.update(env)
    return subprocess.run(
        [sys.executable, "-m", "audit_forensics", *args],
        cwd=str(ROOT),
        env=merged,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def parse_stdout(process: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    try:
        return json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"stdout is not JSON (rc={process.returncode})\n"
            f"stdout={process.stdout}\nstderr={process.stderr}"
        ) from exc


def codes(payload: dict[str, Any]) -> set[str]:
    return {error["code"] for error in payload.get("errors", [])}


def case_signature(payload: dict[str, Any]) -> str:
    return canonical_json(
        [
            {
                "case_id": case["case_id"],
                "rule_id": case["rule_id"],
                "members": case["member_event_ids"],
            }
            for case in payload.get("cases", [])
        ]
    )
