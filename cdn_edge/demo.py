from __future__ import annotations

import copy
import json
import socket
import tempfile
from pathlib import Path
from typing import Any

from .errors import PipelineError
from .fixed import FixedGrid, parse_grid_power
from .ledger import Ledger
from .orchestrator import run_session
from .replay import replay_snapshot
from .spec import load_spec, parse_spec
from .verify import verify_ledger

REPO_SAMPLE_SPEC = Path(__file__).resolve().parent.parent / "samples" / "sample_spec.json"
PACKAGE_SAMPLE_SPEC = Path(__file__).resolve().parent / "data" / "sample_spec.json"
SAMPLE_SPEC = REPO_SAMPLE_SPEC if REPO_SAMPLE_SPEC.exists() else PACKAGE_SAMPLE_SPEC


def _load_sample_raw() -> dict[str, Any]:
    return json.loads(SAMPLE_SPEC.read_text(encoding="utf-8"))


def _session_raw(
    base: dict[str, Any],
    *,
    session_id: str,
    grid_power: int,
    port_base: int,
    batch_ids: set[int] | None,
    include_updates: bool,
) -> dict[str, Any]:
    raw = copy.deepcopy(base)
    raw["session"]["session_id"] = session_id
    raw["session"]["grid_power"] = grid_power
    raw["session"]["source_port"] = port_base
    for index, node in enumerate(raw["nodes"]):
        node["port"] = port_base + index + 1
    if batch_ids is not None:
        raw["requests"] = [request for request in raw["requests"] if request.get("batch") in batch_ids]
    if not include_updates:
        raw["source_updates"] = []
    return raw


def _cycle_raw(base: dict[str, Any]) -> dict[str, Any]:
    raw = copy.deepcopy(base)
    for item in raw["content"]:
        if item["path"] == "/catalog/base.json":
            item["derivation"] = {"inputs": [{"path": "/all.js", "coefficient": 1}]}
            break
    return raw


def _port_in_use(host: str, port: int) -> socket.socket | None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        sock.listen(1)
        return sock
    except OSError:
        sock.close()
        return None


def _summarize_run(phase: str, result: dict[str, Any]) -> dict[str, Any]:
    batches = sorted(result["batches"], key=lambda item: item["batch_id"])
    final_metrics = batches[-1]["metrics"] if batches else None
    aggregate = final_metrics["aggregate"]["metrics"] if final_metrics else {}
    derived = final_metrics["derived_weights"]["paths"] if final_metrics else {}
    return {
        "phase": phase,
        "session_id": result["session_id"],
        "grid": result["grid"],
        "batch_count": result["batch_count"],
        "request_count": sum(len(batch["requests"]) for batch in batches),
        "batch_ids": [batch["batch_id"] for batch in batches],
        "snapshot_ids": [batch["snapshot_id"] for batch in batches],
        "batches": [
            {
                "batch_id": batch["batch_id"],
                "snapshot_id": batch["snapshot_id"],
                "source_counts": batch["metrics"]["source_counts"],
                "hit_rate_units": batch["metrics"]["aggregate"]["metrics"]["hit_rate"]["value_units"],
                "version_consistency_rate_units": batch["metrics"]["aggregate"]["metrics"][
                    "version_consistency_rate"
                ]["value_units"],
                "remaining_ttl_seconds_units": batch["metrics"]["aggregate"]["metrics"][
                    "remaining_ttl_seconds"
                ]["value_units"],
                "error_bounds": {
                    "hit_rate": batch["metrics"]["aggregate"]["metrics"]["hit_rate"]["error_bound_units"],
                    "version_consistency_rate": batch["metrics"]["aggregate"]["metrics"][
                        "version_consistency_rate"
                    ]["error_bound_units"],
                },
            }
            for batch in batches
        ],
        "aggregate": {
            "hit_rate_units": aggregate.get("hit_rate", {}).get("value_units", 0),
            "hit_rate_error_bound_units": aggregate.get("hit_rate", {}).get("error_bound_units", 0),
            "version_consistency_rate_units": aggregate.get("version_consistency_rate", {}).get("value_units", 0),
            "version_consistency_error_bound_units": aggregate.get("version_consistency_rate", {}).get(
                "error_bound_units", 0
            ),
            "remaining_ttl_seconds_units": aggregate.get("remaining_ttl_seconds", {}).get("value_units", 0),
            "dominant_error_sources": {
                name: aggregate[name]["dominant_error_source"] for name in sorted(aggregate)
            },
        },
        "derived_paths": {
            path: {
                "normalized_fixed_sum_units": data["normalized_fixed_sum_units"],
                "normalization_shortfall_units": data["normalization_shortfall_units"],
                "max_error_bound_units": data["max_error_bound_units"],
                "dominant_error_segments": {
                    leaf: payload["dominant_error_segment"]
                    for leaf, payload in sorted(data["leaves"].items())
                },
            }
            for path, data in sorted(derived.items())
        },
    }


def run_demo(
    ledger_path: str | Path,
    *,
    port_base: int = 28400,
    inject: str | None = None,
) -> dict[str, Any]:
    if port_base < 1024 or port_base + 23 > 65535:
        raise PipelineError(
            "E_PORT", "demo port base must leave a high-port window", path="demo", field="port_base"
        )
    base_raw = _load_sample_raw()
    with tempfile.TemporaryDirectory(prefix="cdn-edge-demo-") as temp_text:
        temp = Path(temp_text)
        if inject == "cycle":
            cycle_path = temp / "cycle_spec.json"
            cycle_path.write_text(json.dumps(_cycle_raw(base_raw), ensure_ascii=False), encoding="utf-8")
            load_spec(cycle_path)
            raise AssertionError("cycle injection was not rejected")
        if inject == "grid":
            parse_grid_power("1.5")
            raise AssertionError("grid injection was not rejected")
        if inject == "port":
            occupied = _port_in_use("127.0.0.1", port_base)
            if occupied is None:
                raise PipelineError(
                    "E_PORT", "could not reserve the requested demo port", path="demo", field="port_base"
                )
            try:
                raw = _session_raw(
                    base_raw,
                    session_id="demo-port-conflict",
                    grid_power=3,
                    port_base=port_base,
                    batch_ids={1},
                    include_updates=False,
                )
                spec = parse_spec(raw)
                run_session(spec, ledger_path, work_dir=temp / "port-work")
            finally:
                occupied.close()
            raise AssertionError("port injection was not rejected")

        ledger = Path(ledger_path)
        if ledger.exists():
            # A demo invocation owns a fresh ledger. Existing files are never
            # merged into or deleted implicitly.
            raise PipelineError(
                "E_STATE",
                "demo target ledger already exists; choose a new path",
                path=str(ledger),
                field="ledger",
            )
        specs = [
            parse_spec(
                _session_raw(
                    base_raw,
                    session_id="demo-cold-fill",
                    grid_power=3,
                    port_base=port_base,
                    batch_ids={1},
                    include_updates=False,
                )
            ),
            parse_spec(
                _session_raw(
                    base_raw,
                    session_id="demo-version-update",
                    grid_power=3,
                    port_base=port_base + 10,
                    batch_ids=None,
                    include_updates=True,
                )
            ),
            parse_spec(
                _session_raw(
                    base_raw,
                    session_id="demo-grid-perturb",
                    grid_power=2,
                    port_base=port_base + 20,
                    batch_ids=None,
                    include_updates=True,
                )
            ),
        ]
        run_results = []
        for index, spec in enumerate(specs):
            result = run_session(spec, ledger, work_dir=temp / f"session-{index + 1}")
            run_results.append(result)

        verification = verify_ledger(str(ledger))
        ledger_db = Ledger(ledger, create=True)
        try:
            update_snapshots = ledger_db.list_snapshots("demo-version-update")
            active = ledger_db.active_state()
            if active is None:
                raise PipelineError("E_STATE", "ledger has no active state", path=str(ledger), field="active_state")
            replacement = ledger_db.restore_snapshot(
                str(update_snapshots[1]["snapshot_id"]),
                current_content_digest=str(active["content_digest"]),
                current_grid=FixedGrid(int(active["grid_power"])),
                mode="replace",
            )
        finally:
            ledger_db.close()

        replay_genesis = replay_snapshot(str(ledger), str(update_snapshots[0]["snapshot_id"]))
        replay_batch_one = replay_snapshot(str(ledger), str(update_snapshots[1]["snapshot_id"]))

        if inject == "snapshot":
            ledger_db = Ledger(ledger, create=False)
            try:
                ledger_db.restore_snapshot(
                    str(update_snapshots[1]["snapshot_id"]),
                    current_content_digest=None,
                    current_grid=None,
                    expected_content_digest="0" * 64,
                    mode="replace",
                )
            finally:
                ledger_db.close()
            raise AssertionError("snapshot injection was not rejected")
        if inject == "replay":
            replay_snapshot(
                str(ledger),
                str(update_snapshots[0]["snapshot_id"]),
                grid_override=FixedGrid(4),
            )
            raise AssertionError("replay injection was not rejected")

        ledger_db = Ledger(ledger, create=False)
        try:
            snapshot_count = len(ledger_db.list_snapshots())
            session_rows = ledger_db.list_sessions()
        finally:
            ledger_db.close()
        return {
            "ok": True,
            "ledger": str(ledger.resolve()),
            "sessions": [
                _summarize_run(phase, result)
                for phase, result in zip(("cold-fill", "version-update", "grid-perturb"), run_results)
            ],
            "snapshot_count": snapshot_count,
            "ledger_sessions": len(session_rows),
            "diagnostics": {
                "ledger_self_consistency": verification,
                "whole_replacement": replacement,
                "replay_from_genesis": replay_genesis,
                "replay_from_batch_1": replay_batch_one,
            },
            "conclusion": (
                "fixed-point statistics, error bounds, whole-snapshot replacement and replay checks passed"
            ),
        }
