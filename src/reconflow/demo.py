from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from .canonical import sha256_bytes
from .errors import PortBusyError, ReceiptFieldError, SocketTimeoutFailure, SpecError
from .incremental import (
    edit_document,
    edit_shard_size,
    edit_threshold,
    edit_weight,
    initialize_fast_view,
    verify_fast_matches_authoritative,
)
from .ledger import Ledger
from .repository import (
    open_initialized_ledger,
    persist_current_view,
    process_pending_batches,
    record_fast_read,
    replay_ledger,
)
from .rpc import ComputeClient, start_worker_subprocess
from .scheduler import recompute_shards_tcp
from .synthetic import generate_workspace
from .ui import headless_diagnostic_text
from .workspace import Workspace


def _src_path() -> str:
    return str(Path(__file__).resolve().parents[1])


def _run_crash_child(root: Path, marker: Path, crash_batch: str) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = _src_path() + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "-u",
        "-m",
        "reconflow.cli",
        "--root",
        str(root),
        "process-batches",
        "--crash-batch",
        crash_batch,
        "--marker",
        str(marker),
    ]
    stdout_path = root / "logs" / "crash-child.stdout"
    stderr_path = root / "logs" / "crash-child.stderr"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        process = subprocess.Popen(command, env=env, stdout=out, stderr=err)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not marker.exists():
            if process.poll() is not None:
                raise RuntimeError(f"crash child exited early rc={process.returncode}; stderr={stderr_path.read_text()[-1000:]}")
            time.sleep(0.03)
        if not marker.exists():
            process.kill()
            raise TimeoutError("crash marker was not created")
        process.send_signal(signal.SIGKILL)
        returncode = process.wait(timeout=5)
    return {"returncode": returncode, "forced_sigkill": returncode == -9, "stdout": str(stdout_path), "stderr": str(stderr_path)}


def _shard_digests(view: dict[str, Any]) -> dict[str, str]:
    return {shard["shard_id"]: shard["shard_digest"] for shard in view["conclusion"]["shards"]}


def _edit_report(label: str, propagation: dict[str, Any], before: dict[str, str], ws: Workspace) -> dict[str, Any]:
    after = _shard_digests(ws.load_fast_view())
    actual_changed = sorted(shard_id for shard_id in set(before) | set(after) if before.get(shard_id) != after.get(shard_id))
    expected = sorted(
        propagation.get("invalidated_shards")
        or (set(propagation["affected_shards"]) | set(propagation.get("removed_shards", [])))
    )
    verification = verify_fast_matches_authoritative(ws)
    fast_bytes = ws.fast_conclusion_path.read_bytes()
    authoritative_bytes = ws.authoritative_path.read_bytes()
    preserved_violations = [
        check
        for check in propagation["preservation_checks"]
        if check["ref"].startswith("dedup:")
        and check["ref"].split(":", 1)[1] in propagation["preserved_shards"]
        and not check["byte_identical"]
    ]
    return {
        "label": label,
        "config_version": propagation["config_version"],
        "planned_affected_shards": expected,
        "actual_changed_shards": actual_changed,
        "subset_exact": expected == actual_changed,
        "affected_ratio_domains": propagation["affected_ratio_domains"],
        "preserved_shards": propagation["preserved_shards"],
        "preserved_ratio_domains": propagation["preserved_ratio_domains"],
        "preservation_holds": not preserved_violations,
        "propagation_step_count": len(propagation["steps"]),
        "byte_identical_to_authoritative": fast_bytes == authoritative_bytes,
        "fast_sha256": sha256_bytes(fast_bytes),
        "authoritative_sha256": sha256_bytes(authoritative_bytes),
        "verification_digest": verification["digest"],
    }


def rpc_failure_self_checks() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    occupied.bind(("127.0.0.1", 0))
    occupied.listen(1)
    port = occupied.getsockname()[1]
    try:
        try:
            start_worker_subprocess(port=port, ready_timeout_ms=1_000)
            checks["E_PORT"] = {"raised": False}
        except PortBusyError as exc:
            checks["E_PORT"] = {"raised": True, "code": exc.code, "port": port, "message": exc.message}
    finally:
        occupied.close()

    bound_port: dict[str, int] = {}

    def silent_server() -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        bound_port["port"] = server.getsockname()[1]
        conn, _ = server.accept()
        time.sleep(1.5)
        conn.close()
        server.close()

    thread = threading.Thread(target=silent_server, daemon=True)
    thread.start()
    for _ in range(50):
        if "port" in bound_port:
            break
        time.sleep(0.01)
    try:
        ComputeClient("127.0.0.1", bound_port["port"], timeout_ms=150).connect()
        checks["E_TIMEOUT"] = {"raised": False}
    except SocketTimeoutFailure as exc:
        checks["E_TIMEOUT"] = {"raised": True, "code": exc.code, "phase": exc.details.get("phase"), "message": exc.message}
    thread.join(timeout=2)

    try:
        with start_worker_subprocess(inject_missing_field="digest", timeout_ms=1_000) as worker:
            task = {
                "task_id": "field-self-check",
                "shard_id": "SSELF",
                "domain_id": "self",
                "points": 2,
                "similarity_threshold_bp": 8_500,
                "stage_params": {"min_shared_terms": 1},
                "docs": [
                    {"doc_id": "self:0", "doc_seq": 0, "term_counts": [3, 2], "missing_bitmap": [0, 0], "length": 5},
                    {"doc_id": "self:1", "doc_seq": 1, "term_counts": [3, 2], "missing_bitmap": [0, 0], "length": 5},
                ],
            }
            with ComputeClient("127.0.0.1", worker.port, timeout_ms=1_000) as client:
                client.submit(task)
        checks["E_FIELD"] = {"raised": False}
    except ReceiptFieldError as exc:
        checks["E_FIELD"] = {"raised": True, "code": exc.code, "field": exc.details.get("field"), "message": exc.message}
    return checks


def spec_self_check() -> dict[str, Any]:
    import tempfile

    from .specs import load_domain_records

    with tempfile.NamedTemporaryFile("w", suffix=".ndjson", delete=False, encoding="utf-8") as handle:
        handle.write(json.dumps({"domain_id": "ok", "seed": 1, "doc_count": 1, "batch_size": 1, "vocab_size": 4, "motif_count": 1, "duplicate_rate_bp": 0, "missing_rate_bp": 0}) + "\n")
        handle.write(json.dumps({"domain_id": "bad", "seed": -7, "doc_count": 1}) + "\n")
        name = handle.name
    try:
        load_domain_records(name)
        return {"E_SPEC": {"raised": False}}
    except SpecError as exc:
        return {"E_SPEC": {"raised": True, "code": exc.code, "line": exc.line, "field": exc.field, "message": exc.message}}
    finally:
        Path(name).unlink(missing_ok=True)


def run_demo(root: str | Path, *, tk_smoke: bool = False, keep_server: bool = False) -> dict[str, Any]:
    root = Path(root).resolve()
    allowed_parents = [Path.cwd().resolve(), Path(tempfile.gettempdir()).resolve()]
    if not any(root == base or base in root.parents for base in allowed_parents):
        raise ValueError(f"refusing to reset demo path outside workspace/temp: {root}")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    report: dict[str, Any] = {"success": False, "root": str(root), "steps": []}
    try:
        generate_workspace(root)
        ws = Workspace(root)
        initialize_fast_view(ws)
        ledger = open_initialized_ledger(ws)
        report["steps"].append("generate_and_initialize")

        initial_persist = persist_current_view(ws, ledger, record_read=True)
        report["initial_read_ms"] = initial_persist["read_elapsed_ms"]
        ledger.create_snapshot(ws.snapshot_dir, 1)

        marker = root / "crash.ready"
        crash = _run_crash_child(root, marker, "alpha#b0001")
        report["forced_termination"] = crash
        unknown = ledger.recover_unknown()
        report["recovery"] = {"unknown_count": len(unknown), "first_unknown_batch": unknown[0]["batch_id"] if unknown else None}
        processed = process_pending_batches(ws, ledger)
        report["recovery"]["rerun_batch_count"] = len(processed)
        report["steps"].append("kill_recover_rerun")

        edits: list[dict[str, Any]] = []
        before = _shard_digests(ws.load_fast_view())
        propagation = edit_weight(ws, "alpha", 4_000, use_tcp=True)
        edits.append(_edit_report("weight_alpha_4000", propagation, before, ws))
        persist_current_view(ws, ledger, record_read=True)

        before = _shard_digests(ws.load_fast_view())
        propagation = edit_threshold(ws, 8_800, use_tcp=True)
        edits.append(_edit_report("threshold_8800", propagation, before, ws))
        persist_current_view(ws, ledger, record_read=True)

        selected_doc = ws.load_fast_view()["conclusion"]["weighted_sampling"]["selected"][0]["doc_id"]
        before = _shard_digests(ws.load_fast_view())
        propagation = edit_document(
            ws,
            selected_doc,
            term_counts=[12, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            missing_bitmap=[0] * 12,
            use_tcp=True,
        )
        doc_report = _edit_report("document_content", propagation, before, ws)
        doc_report["doc_id"] = selected_doc
        edits.append(doc_report)
        final_persist = persist_current_view(ws, ledger, record_read=True)
        report["final_read_ms"] = final_persist["read_elapsed_ms"]

        before = _shard_digests(ws.load_fast_view())
        propagation = edit_shard_size(ws, 8, use_tcp=True)
        edits.append(_edit_report("shard_size_8", propagation, before, ws))
        shard_persist = persist_current_view(ws, ledger, record_read=True)
        report["final_read_ms"] = shard_persist["read_elapsed_ms"]
        report["edits"] = edits
        report["steps"].append("continuous_edits_and_tcp_incremental_recompute")

        scheduling = ws.load_fast_view().get("last_scheduling_summary") or {}
        report["last_tcp_scheduling"] = {
            "task_count": scheduling.get("task_count"),
            "peak_global_points": scheduling.get("peak_global_points"),
            "peak_domain_points": scheduling.get("peak_domain_points"),
            "wait_records": len(scheduling.get("wait_records", [])),
            "invariant_violations": scheduling.get("invariant_violations"),
        }
        full_remote = recompute_shards_tcp(ws.load_bundle())
        full_scheduling = full_remote["scheduling"]
        current_digests = {shard["shard_id"]: shard["shard_digest"] for shard in ws.load_fast_view()["conclusion"]["shards"]}
        report["full_tcp_scheduling"] = {
            "task_count": full_scheduling["task_count"],
            "peak_global_points": full_scheduling["peak_global_points"],
            "peak_domain_points": full_scheduling["peak_domain_points"],
            "wait_records": len(full_scheduling["wait_records"]),
            "invariant_violations": full_scheduling["invariant_violations"],
            "matches_conclusion": all(
                current_digests.get(shard["shard_id"]) == shard["shard_digest"]
                for shard in full_remote["shard_results"]
            ),
        }

        replay = replay_ledger(ws, ledger)
        report["replay"] = replay
        snapshot = ledger.create_snapshot(ws.snapshot_dir, int(ws.load_config()["config_version"]))
        report["final_snapshot"] = snapshot
        report["snapshot_count"] = len(ledger.list_snapshots())
        report["steps"].append("replay_and_snapshot")

        report["spec_self_check"] = spec_self_check()
        report["rpc_failure_checks"] = rpc_failure_self_checks()
        report["steps"].append("error_code_self_checks")

        diagnostic_text = headless_diagnostic_text(ws)
        (root / "diagnostic_view.txt").write_text(diagnostic_text, encoding="utf-8")
        report["diagnostic_text"] = str(root / "diagnostic_view.txt")
        if tk_smoke:
            from .ui import run_tk_app

            report["tk_smoke_rc"] = run_tk_app(ws, smoke_ms=600)

        report["all_edits_exact"] = all(item["subset_exact"] and item["byte_identical_to_authoritative"] and item["preservation_holds"] for item in edits)
        report["rpc_error_codes_verified"] = all(item.get("raised") for item in report["rpc_failure_checks"].values())
        report["success"] = bool(
            crash["forced_sigkill"]
            and report["recovery"]["unknown_count"] > 0
            and replay["identical"]
            and report["all_edits_exact"]
            and report["rpc_error_codes_verified"]
            and report["spec_self_check"]["E_SPEC"]["raised"]
        )
        report_path = root / "verification_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        return report
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "code": getattr(exc, "code", None), "details": getattr(exc, "details", {})}
        (root / "verification_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":  # pragma: no cover
    result = run_demo("artifacts/demo", tk_smoke="--tk-smoke" in sys.argv)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    raise SystemExit(0 if result["success"] else 1)
