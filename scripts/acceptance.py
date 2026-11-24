#!/usr/bin/env python3
"""Cross-platform local acceptance run for the diagnostics workbench."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lateness_ledger.canonical import canonical_json, digest_object  # noqa: E402
from lateness_ledger.engine import run_engine  # noqa: E402
from lateness_ledger.ledger import (  # noqa: E402
    LedgerReader,
    LedgerWriter,
    create_snapshot,
    recover_unknown_windows,
    restore_snapshot,
)
from lateness_ledger.probe import LoopbackSourceSubprocess  # noqa: E402
from lateness_ledger.replay import replay_and_compare  # noqa: E402
from lateness_ledger.simulate import generate_events  # noqa: E402
from lateness_ledger.spec import load_bundle  # noqa: E402
from lateness_ledger.ui_model import load_dashboard  # noqa: E402


def bundle_for(strategy: str):
    return load_bundle(
        [ROOT / "samples" / f"sources_{strategy}.json"],
        ROOT / "samples" / "window_rules.json",
        ROOT / "samples" / "derivation_budget.json",
    )


def budget_audit(result) -> dict[str, Any]:
    accepted = 0
    discarded = 0
    exact_zero_count_accepts = 0
    exact_zero_abs_accepts = 0
    for window in result.windows:
        late = window["late_decisions"]
        accepted += len(late["accepted"])
        discarded += len(late["discarded"])
        exact_zero_count_accepts += sum(
            1 for item in late["accepted"] if item["remaining_event_count"] == 0
        )
        exact_zero_abs_accepts += sum(
            1
            for item in late["accepted"]
            if item["remaining_absolute_measure_fixed"] == 0
        )
    return {
        "accepted": accepted,
        "discarded": discarded,
        "exact_count_boundary_accepts": exact_zero_count_accepts,
        "exact_absolute_boundary_accepts": exact_zero_abs_accepts,
    }


def run_strategy(strategy: str, directory: Path) -> dict[str, Any]:
    bundle = bundle_for(strategy)
    ledger = directory / f"{strategy}.sqlite"
    snapshot = directory / f"{strategy}.snapshot.sqlite"
    with LedgerWriter(ledger, reset_existing=True) as writer:
        result = run_engine(bundle, writer=writer)
    snapshot_record = create_snapshot(ledger, snapshot)
    replay = replay_and_compare(bundle, str(ledger))
    restored_target = directory / f"{strategy}-restore.sqlite"
    shutil.copy2(snapshot, restored_target)
    connection = sqlite3.connect(restored_target)
    deleted = connection.execute(
        "SELECT window_id FROM windows ORDER BY window_id LIMIT 1"
    ).fetchone()[0]
    connection.execute("DELETE FROM windows WHERE window_id=?", (deleted,))
    connection.commit()
    connection.close()
    restore_report = restore_snapshot(restored_target, snapshot)
    dashboard = load_dashboard(restored_target)
    window_set_digest = digest_object(
        sorted(window["window_id"] for window in result.windows)
    )
    return {
        "strategy": strategy,
        "summary": result.summary(),
        "budget_audit": budget_audit(result),
        "window_set_digest": window_set_digest,
        "first_finalization_order": [
            {
                "order": window["finalization_order"],
                "window_id": window["window_id"],
                "trigger_seq": window["trigger_seq"],
            }
            for window in result.windows[:5]
        ],
        "displacement_runs": len(result.displacement_runs),
        "error_envelope_sample": result.windows[0]["errors"],
        "snapshot": snapshot_record,
        "replay": replay,
        "restore": restore_report,
        "dashboard_counts": {
            "partitions": len(dashboard["partition_rows"]),
            "windows": len(dashboard["window_rows"]),
            "late_decisions": len(dashboard["late_rows"]),
            "displacements": len(dashboard["displacements"]),
            "error_attributions": len(dashboard["error_rows"]),
            "snapshots": len(dashboard["snapshots"]),
        },
    }


def make_small_spec(directory: Path) -> tuple[Path, Path, Path]:
    source_doc = json.loads(
        (ROOT / "samples" / "sources_source_max.json").read_text(encoding="utf-8")
    )
    for source in source_doc["sources"]:
        source["event_count"] = 40
    source_path = directory / "small-sources.json"
    source_path.write_text(canonical_json(source_doc), encoding="utf-8")
    return (
        source_path,
        ROOT / "samples" / "window_rules.json",
        ROOT / "samples" / "derivation_budget.json",
    )


def injection_check(
    directory: Path,
    sources: Path,
    rules: Path,
    budgets: Path,
    mode: str,
    expected: str,
) -> dict[str, Any]:
    ledger = directory / f"inject-{mode}.sqlite"
    log_path = directory / f"inject-{mode}.jsonl"
    command = [
        sys.executable,
        "-m",
        "lateness_ledger",
        "run",
        "--sources",
        str(sources),
        "--rules",
        str(rules),
        "--budgets",
        str(budgets),
        "--ledger",
        str(ledger),
        "--inject",
        mode,
        "--log-json",
        str(log_path),
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    expected_exit = {
        "E_ORDER": 3,
        "E_BUDGET": 4,
        "E_WATERMARK": 5,
    }[expected]
    log_records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        completed.returncode != expected_exit
        or payload.get("error", {}).get("code") != expected
    ):
        raise RuntimeError(f"injection {mode} failed: {completed.returncode} {payload}")
    if not any(record["code"] == expected for record in log_records):
        raise RuntimeError(f"structured log does not contain {expected}")
    return {
        "mode": mode,
        "exit_code": completed.returncode,
        "report": payload["error"],
        "structured_log_codes": [record["code"] for record in log_records],
    }


def crash_check(directory: Path) -> dict[str, Any]:
    ledger = directory / "crash.sqlite"
    command = [
        sys.executable,
        "-m",
        "lateness_ledger",
        "run",
        "--sources",
        str(ROOT / "samples" / "sources_source_max.json"),
        "--rules",
        str(ROOT / "samples" / "window_rules.json"),
        "--budgets",
        str(ROOT / "samples" / "derivation_budget.json"),
        "--ledger",
        str(ledger),
        "--crash-after",
        "2",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode != 137:
        raise RuntimeError(f"forced crash returned {completed.returncode}")
    unknown = recover_unknown_windows(ledger)
    report = LedgerReader(ledger).status_report()
    return {
        "forced_exit_code": completed.returncode,
        "committed_windows": len(report["finalized"]),
        "unknown_window_count": len(unknown),
        "run_status": report["meta"].get("run_status"),
        "exit_report_claimed": False,
        "first_unknown": unknown[0] if unknown else None,
    }


def probe_check() -> dict[str, Any]:
    with LoopbackSourceSubprocess(server_delay_ms=0) as source:
        success = source.probe(timeout_ms=1000, attempts=1, backoff_ms=0)
    with LoopbackSourceSubprocess(server_delay_ms=250) as source:
        timeout = source.probe(timeout_ms=30, attempts=2, backoff_ms=5)
    return {
        "normal": success.to_dict(),
        "forced_real_timeout": timeout.to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    started = time.monotonic()
    report: dict[str, Any] = {"ok": True, "python": sys.version, "checks": []}
    try:
        with tempfile.TemporaryDirectory(prefix="lateness-acceptance-") as temp:
            directory = Path(temp)
            stream_digests = {}
            for strategy in ("source_max", "partition_min", "fixed_step"):
                bundle = bundle_for(strategy)
                stream_digests[strategy] = digest_object(
                    [event.to_public_dict() for event in generate_events(bundle)]
                )
            report["event_stream_digests"] = stream_digests
            report["same_stream_across_strategies"] = len(set(stream_digests.values())) == 1
            if not report["same_stream_across_strategies"]:
                raise RuntimeError("strategy samples do not share one event stream")

            for strategy in ("source_max", "partition_min", "fixed_step"):
                report["checks"].append(run_strategy(strategy, directory))

            window_sets = {
                check["strategy"]: check["window_set_digest"]
                for check in report["checks"]
            }
            report["same_window_key_set"] = len(set(window_sets.values())) == 1
            if not report["same_window_key_set"]:
                raise RuntimeError("window key sets diverged across strategies")

            sources, rules, budgets = make_small_spec(directory)
            report["anomalies"] = [
                injection_check(directory, sources, rules, budgets, "order", "E_ORDER"),
                injection_check(directory, sources, rules, budgets, "budget", "E_BUDGET"),
                injection_check(
                    directory, sources, rules, budgets, "watermark", "E_WATERMARK"
                ),
            ]
            report["crash_recovery"] = crash_check(directory)
            report["loopback_probe"] = probe_check()
            report["elapsed_ms"] = round((time.monotonic() - started) * 1000.0, 3)
    except Exception as exc:  # noqa: BLE001 - acceptance must report any failure
        report["ok"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
    text = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
