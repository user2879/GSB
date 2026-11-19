"""End-to-end local demonstration runner with deterministic samples."""

from __future__ import annotations

import json
import socket
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .errors import ReplayError, SnapshotError
from .service import DiagnosticService
from .simulator import SimulatorProcess, reserve_free_port


ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIRS = {
    "divergence": ROOT / "samples" / "divergence",
    "missing": ROOT / "samples" / "missing",
    "string_expected": ROOT / "samples" / "string_expected",
}


def _port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _prepare_runtime_spec(source_spec: Path, runtime_dir: Path) -> dict[str, Any]:
    data = json.loads(source_spec.read_text(encoding="utf-8"))
    remapped = False
    used_ports: set[int] = set()
    for node in data["nodes"]:
        requested = int(node["port"])
        if requested not in used_ports and _port_is_available(requested):
            selected = requested
        else:
            selected = reserve_free_port()
            while selected in used_ports or not _port_is_available(selected):
                selected = reserve_free_port()
            node["port"] = selected
            remapped = True
        used_ports.add(selected)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    path = runtime_dir / "spec.json"
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return data


@contextmanager
def running_sample_simulators(
    *,
    sample: str,
    runtime_dir: Path,
    invalid_json_node: str | None = None,
    unreachable_node: str | None = None,
) -> Iterator[tuple[Path, Path, dict[str, Any]]]:
    sample_dir = SAMPLE_DIRS[sample]
    scenario_path = sample_dir / "simulator.json"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    spec_data = _prepare_runtime_spec(sample_dir / "spec.json", runtime_dir)
    spec_path = runtime_dir / "spec.json"
    processes: list[SimulatorProcess] = []
    try:
        for node in sorted(spec_data["nodes"], key=lambda item: item["id"]):
            node_id = node["id"]
            if node_id == unreachable_node:
                continue
            processes.append(
                SimulatorProcess(
                    scenario_path=scenario_path,
                    node_id=node_id,
                    port=int(node["port"]),
                    invalid_json=(node_id == invalid_json_node),
                )
            )
        yield spec_path, scenario_path, spec_data
    finally:
        for process in reversed(processes):
            process.stop()


def _read_log_summary(log_path: Path) -> dict[str, Any]:
    counts: dict[str, int] = {}
    selected: list[dict[str, Any]] = []
    interesting = {
        "E_SPEC",
        "E_OBSERVE",
        "E_PORT",
        "E_SNAPSHOT",
        "E_LATBUDGET",
        "E_REPLAY",
        "REPLAY_OK",
    }
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            code = record.get("code", "")
            counts[code] = counts.get(code, 0) + 1
            if code in interesting:
                selected.append(record)
    return {
        "counts": {code: counts[code] for code in sorted(counts)},
        "events": selected,
    }


def _tamper_current_digest(ledger_path: Path) -> None:
    connection = sqlite3.connect(ledger_path)
    try:
        with connection:
            connection.execute(
                "UPDATE meta SET value='INJECTED-DIGEST-MISMATCH' "
                "WHERE key='current_content_digest'"
            )
    finally:
        connection.close()


def run_demo_case(
    *,
    sample: str,
    inject: str,
    runtime_dir: str | Path,
) -> dict[str, Any]:
    if sample not in SAMPLE_DIRS:
        raise ValueError(f"unknown sample {sample!r}")
    if inject not in {"none", "invalid_json", "port_unreachable", "snapshot_mismatch"}:
        raise ValueError(f"unknown injection {inject!r}")
    case_dir = Path(runtime_dir) / f"{sample}-{inject}"
    case_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = case_dir / "ledger.sqlite"
    log_path = case_dir / "structured.jsonl"
    invalid_json_node = "node-a" if inject == "invalid_json" else None
    unreachable_node = "node-b" if inject == "port_unreachable" else None
    result: dict[str, Any] = {
        "sample": sample,
        "inject": inject,
        "runtime_dir": str(case_dir),
        "ledger": str(ledger_path),
        "log": str(log_path),
    }
    with running_sample_simulators(
        sample=sample,
        runtime_dir=case_dir,
        invalid_json_node=invalid_json_node,
        unreachable_node=unreachable_node,
    ) as (spec_path, _scenario_path, spec_data):
        result["ports"] = {
            node["id"]: node["port"] for node in sorted(spec_data["nodes"], key=lambda n: n["id"])
        }
        service = DiagnosticService(spec_path, ledger_path=ledger_path, log_path=log_path)
        try:
            run_report = service.run_round()
            dashboard = service.query_dashboard()
            result["round"] = {
                "round_id": run_report["round_id"],
                "parameter_version": run_report["parameter_version"],
                "content_digest": run_report["content_digest"],
                "budget_summary": run_report["budget_summary"],
                "divergence_keys": run_report["divergence_keys"],
                "conclusions": [
                    {
                        "key": decision["key"],
                        "status": decision["status"],
                        "effective_value": decision["effective_value"],
                        "winner_basis": decision["winner_basis"],
                        "attributions": decision["attributions"],
                    }
                    for decision in run_report["conclusions"]
                ],
            }
            result["dashboard"] = {
                "meta": dashboard["meta"],
                "ui_latency": dashboard["ui_latency"],
            }
            if inject == "snapshot_mismatch":
                service.close()
                _tamper_current_digest(ledger_path)
                try:
                    unexpected = DiagnosticService(
                        spec_path, ledger_path=ledger_path, log_path=log_path
                    )
                except SnapshotError as exc:
                    result["snapshot_error"] = str(exc)
                else:
                    result["snapshot_error"] = "expected E_SNAPSHOT but reopen succeeded"
                    unexpected.close()
            else:
                try:
                    replay_report = service.replay()
                    result["replay"] = {
                        "status": replay_report["status"],
                        "merge_digest": replay_report["merge_digest"],
                        "latency_comparison_source": replay_report[
                            "latency_comparison_source"
                        ],
                    }
                except ReplayError as exc:
                    result["replay"] = {
                        "status": "MISMATCH",
                        "first_diff": exc.first_diff,
                    }
                finally:
                    service.close()
        finally:
            try:
                service.close()
            except Exception:
                pass
    result["structured_log"] = _read_log_summary(log_path)
    return result


def run_all_demos(runtime_dir: str | Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for sample, inject in (
        ("divergence", "none"),
        ("missing", "none"),
        ("divergence", "invalid_json"),
        ("divergence", "port_unreachable"),
        ("divergence", "snapshot_mismatch"),
    ):
        results.append(run_demo_case(sample=sample, inject=inject, runtime_dir=runtime_dir))
    return results
