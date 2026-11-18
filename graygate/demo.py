from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .network import send_json_line
from .routing import decide_route
from .service import WorkbenchService
from .spec import load_spec
from .util import canonical_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "samples" / "gateway_spec.json"
DEFAULT_OUTPUT = ROOT / "demo-output"


def _sample(index: int, size: int = 8) -> list[float]:
    return [((index * 5 + offset * 3) % 23) / 23.0 for offset in range(size)]


def _stat_rows(snapshot: Any) -> list[dict[str, Any]]:
    return [
        {
            "rule_id": row["rule_id"],
            "stage_id": row["stage_id"],
            "stage_weight_bp": row["stage_weight_bp"],
            "stable_hits": row["stable_hits"],
            "canary_hits": row["canary_hits"],
            "canary_ratio_bp": row["canary_ratio_bp"],
            "stable_total_us": row["stable_total_us"],
            "canary_total_us": row["canary_total_us"],
            "stable_max_us": row["stable_max_us"],
            "canary_max_us": row["canary_max_us"],
        }
        for row in snapshot.stats
    ]


def _clear_db(path: Path) -> None:
    for candidate in path.parent.glob(path.name + "*"):
        candidate.unlink()


def _write_scenario(output_dir: Path, name: str, payload: dict[str, Any]) -> None:
    path = output_dir / f"{name}.json"
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def scenario_cold_start(spec_path: Path, output_dir: Path) -> dict[str, Any]:
    db_path = output_dir / "scenario-cold.db"
    _clear_db(db_path)
    spec = load_spec(spec_path)
    service = WorkbenchService(spec, db_path)
    from .gateway import GatewayStack

    stack = GatewayStack(spec, service.ledger)
    request_count = 480
    try:
        stack.start()

        def call(index: int) -> dict[str, Any]:
            return send_json_line(
                spec.port,
                {
                    "client_key": f"ramp-client-{index % 17}",
                    "feature": "alpha:ramp" if index % 2 == 0 else "beta:ramp",
                    "sample": _sample(index),
                },
                timeout=20.0,
            )

        with ThreadPoolExecutor(max_workers=16) as gate:
            responses = list(gate.map(call, range(request_count)))
        failures = [response for response in responses if not response.get("ok")]
        if failures:
            raise RuntimeError(f"gateway rejected {len(failures)} requests: {failures[:2]}")
        snapshot = service.rebuild_now()
        consistency = service.check_consistency()
        dashboard = service.dashboard()
        by_stage: dict[str, dict[str, int]] = {}
        for row in snapshot.stats:
            by_stage[f"{row['rule_id']}/{row['stage_id']}"] = {
                "stable_hits": row["stable_hits"],
                "canary_hits": row["canary_hits"],
                "canary_ratio_bp": row["canary_ratio_bp"],
            }
        result = {
            "scenario": "cold_start_ramp",
            "ok": consistency.ok and dashboard["ok"],
            "requests": request_count,
            "ledger_version": service.ledger.current_version(),
            "snapshot": {
                "snapshot_id": snapshot.snapshot_id,
                "source_ledger_version": snapshot.source_ledger_version,
                "observed_ledger_version": snapshot.observed_ledger_version,
                "excluded_hits": snapshot.excluded_hits,
                "content_digest": snapshot.content_digest,
                "aggregation_path": snapshot.aggregation_path,
                "rebuild_elapsed_ms": snapshot.rebuild_elapsed_ms,
                "budget_ms": snapshot.budget_ms,
            },
            "stage_usage": [by_stage[key] | {"rule_stage": key} for key in sorted(by_stage)],
            "consistency": consistency.canonical(),
            "latency": {
                "rebuild": dashboard["latency"]["rebuild"]["summary"],
                "query": dashboard["latency"]["ui_query"]["summary"],
            },
            "errors": dashboard["errors"],
        }
        _write_scenario(output_dir, "scenario-1-cold-start", result)
        return result
    finally:
        stack.stop()
        service.close()


def scenario_rebuild_during_traffic(spec_path: Path, output_dir: Path) -> dict[str, Any]:
    db_path = output_dir / "scenario-traffic.db"
    _clear_db(db_path)
    spec = load_spec(spec_path)
    service = WorkbenchService(spec, db_path)
    try:
        for index in range(120):
            service.ingest(
                f"traffic-client-{index % 13}",
                "alpha:live" if index % 2 == 0 else "beta:live",
                _sample(index),
            )
        rebuild_id = service.start_rebuild(pause_ms=250)
        during = []
        for index in range(120, 160):
            response = service.ingest(
                f"traffic-client-{index % 13}",
                "alpha:live" if index % 2 == 0 else "beta:live",
                _sample(index),
            )
            route = spec.routes_by_id[response["rule_id"]]
            expected = decide_route(
                spec, route, response["request_seq"], f"traffic-client-{index % 13}"
            )
            during.append(
                {
                    "request_seq": response["request_seq"],
                    "stage_id": response["stage_id"],
                    "side": response["side"],
                    "expected_side": expected.side,
                    "matched": response["side"] == expected.side,
                }
            )
        info = service.wait_rebuild(rebuild_id, timeout=20.0)
        snapshot = service.current_snapshot
        assert snapshot is not None
        consistency = service.check_consistency()
        total_in_snapshot = sum(
            row["stable_hits"] + row["canary_hits"] for row in snapshot.stats
        )
        final_snapshot = service.rebuild_now()
        final_consistency = service.check_consistency()
        dashboard = service.dashboard()
        result = {
            "scenario": "rebuild_during_traffic",
            "ok": (
                consistency.ok
                and final_consistency.ok
                and snapshot.excluded_hits >= 40
                and total_in_snapshot == 120
                and all(item["matched"] for item in during)
            ),
            "rebuild_id": rebuild_id,
            "rebuild_status": info["status"],
            "snapshot_while_running": {
                "snapshot_id": snapshot.snapshot_id,
                "source_ledger_version": snapshot.source_ledger_version,
                "observed_ledger_version": snapshot.observed_ledger_version,
                "excluded_hits": snapshot.excluded_hits,
                "hits_in_snapshot": total_in_snapshot,
                "rebuild_elapsed_ms": snapshot.rebuild_elapsed_ms,
                "budget_ms": snapshot.budget_ms,
            },
            "admitted_during_rebuild": len(during),
            "routing_unchanged": all(item["matched"] for item in during),
            "consistency_during": consistency.canonical(),
            "final_snapshot": {
                "snapshot_id": final_snapshot.snapshot_id,
                "source_ledger_version": final_snapshot.source_ledger_version,
                "excluded_hits": final_snapshot.excluded_hits,
            },
            "consistency_final": final_consistency.canonical(),
            "latency": {
                "rebuild": dashboard["latency"]["rebuild"]["summary"],
                "query": dashboard["latency"]["ui_query"]["summary"],
            },
            "errors": dashboard["errors"],
        }
        _write_scenario(output_dir, "scenario-2-rebuild-traffic", result)
        return result
    finally:
        service.close()


def scenario_forced_termination(spec_path: Path, output_dir: Path) -> dict[str, Any]:
    db_path = output_dir / "scenario-crash.db"
    marker_path = output_dir / "scenario-crash.marker.json"
    log_path = output_dir / "scenario-crash.worker.log"
    _clear_db(db_path)
    marker_path.unlink(missing_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    with log_path.open("wb") as log_file:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "graygate.demo_worker",
                "--spec",
                str(spec_path),
                "--db",
                str(db_path),
                "--marker",
                str(marker_path),
            ],
            cwd=str(ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    marker: dict[str, Any] | None = None
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if marker_path.exists():
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                break
            except json.JSONDecodeError:
                pass
        if process.poll() is not None:
            break
        time.sleep(0.05)
    if marker is None:
        process.kill()
        process.wait(timeout=5)
        log = log_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(f"crash worker did not reach marker: {log[-2000:]}")
    process.kill()
    process.wait(timeout=10)

    spec = load_spec(spec_path)
    recovered = WorkbenchService(spec, db_path)
    try:
        unknown = recovered.ledger.recovered_unknown
        report = recovered.recover_and_replay()
        replay = report["replays"][0] if report["replays"] else None
        retriggered = recovered.rebuild_now()
        retrigger_check = recovered.check_consistency()
        dashboard = recovered.dashboard()
        result = {
            "scenario": "forced_termination_recovery",
            "ok": (
                bool(unknown)
                and report["ledger_chain"]["ok"]
                and replay is not None
                and replay["code"] == "OK"
                and retrigger_check.ok
                and report["exit_report_claimed"] is False
            ),
            "forced_termination": True,
            "worker_reached_marker": True,
            "unknown_rebuilds": [
                {
                    "rebuild_id": row["rebuild_id"],
                    "status": "UNKNOWN",
                    "error_code": row["error_code"],
                }
                for row in unknown
            ],
            "retrigger_required": report["retrigger_required_for_unknown"],
            "exit_report_claimed": report["exit_report_claimed"],
            "ledger_chain": report["ledger_chain"],
            "replay": replay,
            "retriggered_snapshot": {
                "snapshot_id": retriggered.snapshot_id,
                "source_ledger_version": retriggered.source_ledger_version,
                "content_digest": retriggered.content_digest,
            },
            "retrigger_consistency": retrigger_check.canonical(),
            "latency": {
                "rebuild": dashboard["latency"]["rebuild"]["summary"],
                "query": dashboard["latency"]["ui_query"]["summary"],
            },
        }
        _write_scenario(output_dir, "scenario-3-crash-recovery", result)
        return result
    finally:
        recovered.close()


def _write_summary(output_dir: Path, results: list[dict[str, Any]]) -> None:
    lines = []
    for result in results:
        lines.append(
            "SCENARIO {scenario} ok={ok}".format(
                scenario=result["scenario"], ok=result["ok"]
            )
        )
        if result["scenario"] == "cold_start_ramp":
            lines.append(
                "  snapshot source={} digest={} consistency={}".format(
                    result["snapshot"]["source_ledger_version"],
                    result["snapshot"]["content_digest"],
                    result["consistency"]["code"],
                )
            )
        elif result["scenario"] == "rebuild_during_traffic":
            lines.append(
                "  snapshot source={} observed={} excluded={} hits={} routing_unchanged={}".format(
                    result["snapshot_while_running"]["source_ledger_version"],
                    result["snapshot_while_running"]["observed_ledger_version"],
                    result["snapshot_while_running"]["excluded_hits"],
                    result["snapshot_while_running"]["hits_in_snapshot"],
                    result["routing_unchanged"],
                )
            )
            lines.append(
                "  consistency_during={} consistency_final={}".format(
                    result["consistency_during"]["code"],
                    result["consistency_final"]["code"],
                )
            )
        else:
            lines.append(
                "  unknown_rebuilds={} replay={} retrigger={} exit_report_claimed={}".format(
                    len(result["unknown_rebuilds"]),
                    result["replay"]["code"] if result["replay"] else "missing",
                    result["retrigger_consistency"]["code"],
                    result["exit_report_claimed"],
                )
            )
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run all deterministic demo scenarios")
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = Path(args.spec)
    results = [
        scenario_cold_start(spec_path, output_dir),
        scenario_rebuild_during_traffic(spec_path, output_dir),
        scenario_forced_termination(spec_path, output_dir),
    ]
    _write_summary(output_dir, results)
    return 0 if all(result["ok"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

