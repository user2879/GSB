from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
CONFIG = [
    "--global-concurrency",
    "3",
    "--per-node-concurrency",
    "1",
    "--batch-size",
    "2",
    "--request-timeout-ms",
    "2500",
    "--retries",
    "2",
]


def _run(args: list[str]) -> dict[str, Any]:
    command = [sys.executable, "-m", "trace_arbiter", *args]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"command did not return JSON:\n{completed.stdout}") from exc


def _assert_safe_workdir(path: Path) -> None:
    text = str(path)
    if any(ord(character) > 127 for character in text) or any(
        character.isspace() for character in text
    ):
        raise SystemExit("workdir must not contain spaces or non-ASCII characters")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local end-to-end acceptance demo")
    parser.add_argument("--workdir", default=str(ROOT / "demo_output"))
    args = parser.parse_args()
    base_workdir = Path(args.workdir).resolve()
    _assert_safe_workdir(base_workdir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    workdir = base_workdir / f"run-{stamp}"
    workdir.mkdir(parents=True, exist_ok=False)
    db = workdir / "ledger.sqlite3"
    log = workdir / "events.log.jsonl"

    first = _run(
        [
            "adjudicate",
            "--nodes",
            str(ROOT / "sample_data" / "nodes_v1.json"),
            "--spans",
            str(ROOT / "sample_data" / "spans.ndjson"),
            "--db",
            str(db),
            "--log",
            str(log),
            *CONFIG,
        ],
    )
    first_review = _run(
        ["review", "--db", str(db), "--log", str(log), *CONFIG],
    )
    first_replay = _run(
        ["replay", "--db", str(db), "--log", str(log), *CONFIG],
    )
    first_view = _run(
        ["view", "--db", str(db), "--log", str(log), "--json"],
    )
    static_before = next(
        trace for trace in first_view["traces"] if trace["trace_id"] == "trace-static"
    )
    if not all(trace["entry_digest_valid"] for trace in first_view["traces"]):
        raise RuntimeError("ruleset 1 contains an invalid ledger entry digest")

    second = _run(
        [
            "adjudicate",
            "--nodes",
            str(ROOT / "sample_data" / "nodes_v2.json"),
            "--spans",
            str(ROOT / "sample_data" / "spans.ndjson"),
            "--db",
            str(db),
            "--log",
            str(log),
            *CONFIG,
        ],
    )
    second_review = _run(
        ["review", "--db", str(db), "--log", str(log), *CONFIG],
    )
    second_replay = _run(
        ["replay", "--db", str(db), "--log", str(log), *CONFIG],
    )
    second_view = _run(
        ["view", "--db", str(db), "--log", str(log), "--json"],
    )
    static_after = next(
        trace for trace in second_view["traces"] if trace["trace_id"] == "trace-static"
    )
    if not all(trace["entry_digest_valid"] for trace in second_view["traces"]):
        raise RuntimeError("ruleset 2 contains an invalid ledger entry digest")
    if first["ledger"]["trace_count"] != 5 or second["ledger"]["trace_count"] != 5:
        raise RuntimeError("expected five traces in both ruleset snapshots")
    if len(second_view["flips"]) != 4:
        raise RuntimeError(f"expected four flips, got {len(second_view['flips'])}")
    unchanged = static_before["entry"] == static_after["entry"]
    if not unchanged:
        raise RuntimeError("unchanged trace-static ledger bytes changed across ruleset replacement")
    if not all(
        result["ok"]
        for result in (first_review, first_replay, second_review, second_replay)
    ):
        raise RuntimeError("authoritative review or replay did not succeed")

    report = {
        "ok": True,
        "workdir": str(workdir),
        "ruleset_1": {
            "snapshot_id": first["snapshot_id"],
            "trace_count": first["ledger"]["trace_count"],
            "review": first_review["ok"],
            "replay": first_replay["ok"],
        },
        "ruleset_2": {
            "snapshot_id": second["snapshot_id"],
            "trace_count": second["ledger"]["trace_count"],
            "review": second_review["ok"],
            "replay": second_replay["ok"],
            "changed_traces": [
                {
                    "trace_id": flip["trace_id"],
                    "change_kind": flip["change_kind"],
                    "diff_fields": flip["diff_fields"],
                    "before_rule_versions": flip["before_rule_versions"],
                    "after_rule_versions": flip["after_rule_versions"],
                }
                for flip in second_view["flips"]
            ],
            "unchanged_trace": "trace-static" if unchanged else None,
            "stale_sources_excluded": second_view["stale_sources_excluded"],
            "view_mode": second_view["view_mode"],
        },
    }
    report_path = workdir / "validation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
