"""Repeatable end-to-end acceptance demonstration over the loopback TCP service."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .client import WorkbenchClient
from .errors import BusinessError
from .protocol import ProtocolClient
from .util import (
    atomic_write_json,
    atomic_write_text,
    b64encode,
    digest_bytes,
    mono_ms,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES = REPO_ROOT / "samples"


def _sample_bytes() -> bytes:
    header = b"CHUNKDEPS-WINDOWS-DETERMINISTIC-SAMPLE/v1\n"
    body = bytes((index * 37 + 11) % 256 for index in range(8192))
    footer = b"\nEND-OF-SAMPLE\n"
    return header + body + footer


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_server(
    port: int, process: subprocess.Popen[Any], timeout: float = 15.0
) -> WorkbenchClient:
    deadline = time.monotonic() + timeout
    probe = ProtocolClient("127.0.0.1", port, timeout=1.0)
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"storage service exited early with code {process.returncode}"
            )
        try:
            probe.ping()
            return WorkbenchClient("127.0.0.1", port, timeout=120.0)
        except Exception as exc:  # noqa: BLE001 - startup polling
            last_error = exc
            time.sleep(0.05)
    raise RuntimeError(f"storage service did not become ready: {last_error}")


def _start_server(
    root: Path,
    *,
    global_concurrency: int,
    session_concurrency: int,
    env_extra: dict[str, str] | None = None,
) -> tuple[subprocess.Popen[Any], int, WorkbenchClient]:
    port = _free_port()
    env = os.environ.copy()
    env.pop("CHUNKDEPS_MERGE_PAUSE_MS", None)
    if env_extra:
        env.update(env_extra)
    command = [
        sys.executable,
        "-m",
        "chunkdeps.cli",
        "server",
        "--root",
        str(root),
        "--listen-host",
        "127.0.0.1",
        "--listen-port",
        str(port),
        "--global-concurrency",
        str(global_concurrency),
        "--session-concurrency",
        str(session_concurrency),
        "--log",
        str(root / "logs" / "service.jsonl"),
    ]
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    client = _wait_for_server(port, process)
    return process, port, client


def _stop_server(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _force_kill(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        os.kill(process.pid, signal.SIGKILL)
    process.wait(timeout=10)


def _capture(action) -> dict[str, Any]:
    try:
        return {"ok": True, "result": action()}
    except BusinessError as exc:
        return {"ok": False, "error": exc.as_dict()}


def _node_snapshot(graph: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    for node in graph["nodes"]:
        if node["node_id"] == node_id:
            return {
                "node_id": node_id,
                "entry_version": node["entry_version"],
                "cached_digest": node["cached_digest"],
                "valid": node["valid"],
            }
    return None


def _max_overlap(schedules: list[dict[str, Any]]) -> int:
    events: list[tuple[int, int]] = []
    for schedule in schedules:
        if schedule:
            events.append((schedule["start_mono_ms"], 1))
            events.append((schedule["end_mono_ms"], -1))
    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    peak = 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    return peak


def _wait_for_log_event(
    log_path: Path,
    event_name: str,
    session_id: str,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    offset = log_path.stat().st_size if log_path.is_file() else 0
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if log_path.is_file():
            with log_path.open("r", encoding="utf-8") as stream:
                stream.seek(offset)
                for line in stream:
                    offset += len(line.encode("utf-8"))
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if (
                        record.get("event") == event_name
                        and record.get("session_id") == session_id
                    ):
                        return record
        time.sleep(0.05)
    return None


def run_demo(root: Path, report_path: Path) -> dict[str, Any]:
    started = mono_ms()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    declaration = json.loads(
        (SAMPLES / "sample_declaration.json").read_text(encoding="utf-8")
    )
    sample = _sample_bytes()
    sample_path = root / "windows-sample.bin"
    sample_path.write_bytes(sample)
    if digest_bytes(sample) != declaration["sha256"]:
        raise RuntimeError("sample declaration digest does not match generated sample")

    processes: list[subprocess.Popen[Any]] = []
    process, port, client = _start_server(
        root, global_concurrency=3, session_concurrency=2
    )
    processes.append(process)
    report: dict[str, Any] = {
        "schema": "chunkdeps-validation-report/v1",
        "generated_mono_ms": started,
        "sample": {
            "path": str(sample_path),
            "size": len(sample),
            "sha256": digest_bytes(sample),
            "declaration_sha256": declaration["sha256"],
        },
        "service": {
            "host": "127.0.0.1",
            "initial_port": port,
            "global_concurrency": 3,
            "session_concurrency": 2,
        },
        "steps": {},
        "checks": {},
        "business_codes": {},
    }
    try:
        report["steps"]["initial_ping"] = client.ping()

        concurrent_sources = {
            "concurrency/a.bin": client.create_session(
                object_key="concurrency/a.bin",
                source=sample_path,
                chunk_size=512,
                tags={"case": "concurrency-a"},
            ),
            "concurrency/b.bin": client.create_session(
                object_key="concurrency/b.bin",
                source=sample_path,
                chunk_size=512,
                tags={"case": "concurrency-b"},
            ),
        }

        def upload_concurrent(object_key: str) -> dict[str, Any]:
            return client.upload_session(
                session_id=concurrent_sources[object_key]["session_id"],
                source=sample_path,
                concurrency=6,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            concurrent_uploads = list(pool.map(upload_concurrent, concurrent_sources))
        concurrent_commits = [
            client.request(
                "commit_session",
                {"session_id": concurrent_sources[key]["session_id"]},
            )
            for key in ("concurrency/a.bin", "concurrency/b.bin")
        ]
        concurrent_schedules = [
            schedule
            for upload in concurrent_uploads
            for schedule in upload.get("schedule", [])
            if schedule
        ]
        throttle = client.ping()["throttle"]
        report["steps"]["concurrency"] = {
            "sessions": {
                key: concurrent_sources[key]["session_id"] for key in concurrent_sources
            },
            "chunk_count_each": len(concurrent_uploads[0]["uploaded"]),
            "observed_peak_global": _max_overlap(concurrent_schedules),
            "observed_peak_global_server": throttle["peak_global"],
            "observed_peak_by_session_server": throttle["peak_by_session"],
            "schedule_sample": concurrent_schedules[:6],
            "content_digests": [
                commit["version"]["content_digest"] for commit in concurrent_commits
            ],
        }

        stable_object = client.upload_file(
            object_key="archive/other/stable.txt",
            source=sample_path,
            chunk_size=4096,
            tags={"case": "stable-control"},
            concurrency=3,
        )
        report["steps"]["stable_control_object"] = {
            "session_id": stable_object["created_session"]["session_id"],
            "version": stable_object["commit"]["version"],
        }

        v1 = client.upload_file(
            object_key="archive/demo/sample.bin",
            source=sample_path,
            chunk_size=1024,
            tags={"case": "v1", "team": "storage"},
            concurrency=4,
            expected_base_version=0,
        )
        report["steps"]["v1"] = {
            "session_id": v1["created_session"]["session_id"],
            "chunk_count": len(v1["upload"]["uploaded"]),
            "version": v1["commit"]["version"],
            "schedule": v1["upload"]["schedule"],
        }

        graph_before_v2 = client.request("get_graph")
        stable_metadata_node = _node_snapshot(
            graph_before_v2, "metadata:archive/other/stable.txt:v1"
        )
        stable_view_node = _node_snapshot(
            graph_before_v2, "prefix-view:archive/other"
        )

        v2_session = client.create_session(
            object_key="archive/demo/sample.bin",
            source=sample_path,
            chunk_size=2048,
            tags={"case": "v2", "team": "storage"},
            expected_base_version=1,
        )
        v2_upload = client.upload_session(
            session_id=v2_session["session_id"],
            source=sample_path,
            concurrency=5,
        )
        reupload = client.upload_session(
            session_id=v2_session["session_id"],
            source=sample_path,
            concurrency=1,
            reupload_indices=[0],
        )
        abort = client.request(
            "abort_session",
            {"session_id": v2_session["session_id"], "reason": "demo resumable abort"},
        )
        resume = client.request(
            "resume_session", {"session_id": v2_session["session_id"]}
        )
        v2_commit = client.request(
            "commit_session", {"session_id": v2_session["session_id"]}
        )
        objects_after_v2 = client.request(
            "list_objects", {"object_key": "archive/demo/sample.bin"}
        )
        graph_after_v2 = client.request("get_graph")
        stable_metadata_after = _node_snapshot(
            graph_after_v2, "metadata:archive/other/stable.txt:v1"
        )
        stable_view_after = _node_snapshot(
            graph_after_v2, "prefix-view:archive/other"
        )
        audit_v2 = client.request("audit", {})
        report["steps"]["v2"] = {
            "session_id": v2_session["session_id"],
            "uploaded": len(v2_upload["uploaded"]),
            "reupload_invalidation": reupload["uploaded"][0]["invalidation"],
            "abort": {
                "state": abort["session"]["state"],
                "chunks_retained": abort["chunks_retained"],
                "invalidation_affected": abort["invalidation"]["affected_count"],
            },
            "resume": {
                "state": resume["state"],
                "skipped_indices": resume["skipped_indices"],
                "missing_indices": resume["missing_indices"],
            },
            "commit": v2_commit,
            "versions": objects_after_v2["versions"],
            "audit": audit_v2,
            "unaffected_metadata_before": stable_metadata_node,
            "unaffected_metadata_after": stable_metadata_after,
            "unaffected_view_before": stable_view_node,
            "unaffected_view_after": stable_view_after,
        }

        snapshot = client.request("snapshot_save", {})
        deleted = client.request(
            "delete_object", {"object_key": "archive/demo/sample.bin"}
        )
        restore = client.request("snapshot_restore", {})
        audit_after_restore = client.request("audit", {})
        restored_object = client.request(
            "list_objects", {"object_key": "archive/demo/sample.bin"}
        )
        report["steps"]["snapshot"] = {
            "saved": snapshot,
            "delete": deleted,
            "restore": restore,
            "audit_after_restore": audit_after_restore,
            "restored_versions": restored_object["versions"],
        }

        fault_truncate_session = client.create_session(
            object_key="fault/chunk.bin",
            source=sample_path,
            chunk_size=2048,
            tags={"case": "fault-truncate"},
        )
        first_chunk = sample[:2048]
        truncate = _capture(
            lambda: client.request(
                "upload_chunk",
                {
                    "session_id": fault_truncate_session["session_id"],
                    "chunk_index": 0,
                    "data_b64": b64encode(first_chunk),
                    "digest": digest_bytes(first_chunk),
                    "inject": "truncate-chunk",
                },
            )
        )

        integrity_session = client.create_session(
            object_key="fault/integrity.bin",
            source=sample_path,
            chunk_size=1500,
            tags={"case": "fault-integrity"},
        )
        client.upload_session(
            session_id=integrity_session["session_id"],
            source=sample_path,
            concurrency=4,
        )
        corrupt = _capture(
            lambda: client.request(
                "commit_session",
                {
                    "session_id": integrity_session["session_id"],
                    "inject": "corrupt-metadata",
                },
            )
        )
        retained_integrity_session = client.request(
            "get_session", {"session_id": integrity_session["session_id"]}
        )
        client.upload_session(
            session_id=integrity_session["session_id"],
            source=sample_path,
            concurrency=1,
            reupload_indices=[0],
        )
        integrity_repaired = client.request(
            "commit_session", {"session_id": integrity_session["session_id"]}
        )

        version_a = client.create_session(
            object_key="fault/version.bin",
            source=sample_path,
            chunk_size=2048,
            tags={"case": "version-a"},
            expected_base_version=0,
        )
        client.upload_session(
            session_id=version_a["session_id"], source=sample_path, concurrency=4
        )
        version_b = client.create_session(
            object_key="fault/version.bin",
            source=sample_path,
            chunk_size=4096,
            tags={"case": "version-b"},
            expected_base_version=0,
        )
        client.upload_session(
            session_id=version_b["session_id"], source=sample_path, concurrency=4
        )
        competing_commit = client.request(
            "commit_session", {"session_id": version_b["session_id"]}
        )
        version_conflict = _capture(
            lambda: client.request(
                "commit_session",
                {
                    "session_id": version_a["session_id"],
                    "expected_base_version": 0,
                },
            )
        )
        report["steps"]["faults"] = {
            "truncate": truncate,
            "corrupt_metadata": corrupt,
            "retained_after_integrity": {
                "state": retained_integrity_session["state"],
                "bitmap": retained_integrity_session["bitmap"],
            },
            "integrity_repair_commit": integrity_repaired["version"],
            "version_conflict": version_conflict,
            "competing_commit": competing_commit["version"],
        }

        client.request("audit", {})
        _stop_server(process)
        processes.remove(process)

        pause_process, pause_port, pause_client = _start_server(
            root,
            global_concurrency=3,
            session_concurrency=2,
            env_extra={"CHUNKDEPS_MERGE_PAUSE_MS": "15000"},
        )
        processes.append(pause_process)
        crash_session = pause_client.create_session(
            object_key="recovery/merge.bin",
            source=sample_path,
            chunk_size=2048,
            tags={"case": "forced-termination"},
        )
        pause_client.upload_session(
            session_id=crash_session["session_id"],
            source=sample_path,
            concurrency=4,
        )
        commit_outcome: dict[str, Any] = {}

        def commit_in_dying_process() -> None:
            try:
                commit_outcome["result"] = pause_client.request(
                    "commit_session", {"session_id": crash_session["session_id"]}
                )
            except Exception as exc:  # noqa: BLE001 - expected connection loss
                commit_outcome["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }

        commit_thread = threading.Thread(target=commit_in_dying_process, daemon=True)
        commit_thread.start()
        pause_record = _wait_for_log_event(
            root / "logs" / "service.jsonl",
            "merge_pause_for_forced_termination_test",
            crash_session["session_id"],
        )
        if pause_record is None:
            raise RuntimeError(
                "commit never entered verifying state before forced termination"
            )
        observed_state = "verifying"
        time.sleep(0.8)
        _force_kill(pause_process)
        processes.remove(pause_process)
        commit_thread.join(timeout=5)

        recovery_process, recovery_port, recovery_client = _start_server(
            root, global_concurrency=3, session_concurrency=2
        )
        processes.append(recovery_process)
        recovery = recovery_client.request("recovery_status", {})
        unknown_ids = [
            item["session_id"] for item in recovery["unknown_merge_sessions"]
        ]
        crash_objects_before = recovery_client.request(
            "list_objects", {"object_key": "recovery/merge.bin"}
        )
        rerun_commit = recovery_client.request(
            "commit_session", {"session_id": crash_session["session_id"]}
        )
        crash_objects_after = recovery_client.request(
            "list_objects", {"object_key": "recovery/merge.bin"}
        )
        report["steps"]["forced_termination"] = {
            "observed_state_before_kill": observed_state,
            "dying_commit_outcome": commit_outcome,
            "recovery": recovery,
            "session_marked_unknown": crash_session["session_id"] in unknown_ids,
            "visible_versions_before_rerun": crash_objects_before["version_count"],
            "rerun_commit": rerun_commit["version"],
            "visible_versions_after_rerun": crash_objects_after["version_count"],
            "exit_report_available": recovery["exit_report_available"],
        }

        final_graph = recovery_client.request("get_graph")
        final_audit = recovery_client.request("audit", {})
        final_status = recovery_client.request("status", {})
        report["steps"]["final_graph"] = final_graph["validation"]
        report["steps"]["final_audit"] = final_audit
        report["steps"]["final_status"] = final_status

        log_path = root / "logs" / "service.jsonl"
        codes = {code: 0 for code in ("E_CHUNK", "E_INTEGRITY", "E_VERSION")}
        log_records = 0
        if log_path.is_file():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                log_records += 1
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                code = record.get("code")
                if code in codes:
                    codes[code] += 1
        report["business_codes"] = {
            "counts": codes,
            "structured_log_path": str(log_path),
            "structured_log_records": log_records,
        }

        checks = {
            "sample_digest_matches_declaration": digest_bytes(sample)
            == declaration["sha256"],
            "concurrent_content_digests_match": all(
                commit["version"]["content_digest"] == digest_bytes(sample)
                for commit in concurrent_commits
            ),
            "global_limit_respected": _max_overlap(concurrent_schedules) <= 3,
            "session_quota_respected": all(
                value <= 2 for value in throttle["peak_by_session"].values()
            ),
            "concurrency_observed": _max_overlap(concurrent_schedules) >= 2,
            "v2_content_digest_matches": v2_commit["version"]["content_digest"]
            == digest_bytes(sample),
            "v2_is_monotonic": v2_commit["version"]["version"] == 2,
            "v1_expired_after_v2": any(
                version["version"] == 1 and version["status"] == "expired"
                for version in objects_after_v2["versions"]
            ),
            "whole_replacement": v2_commit["replacement"] == "whole",
            "unaffected_metadata_node_unchanged": stable_metadata_node
            == stable_metadata_after,
            "unaffected_view_node_unchanged": stable_view_node == stable_view_after,
            "reupload_invalidated_dependencies": reupload["uploaded"][0][
                "invalidation"
            ]["affected_count"]
            >= 2,
            "abort_retained_chunks": abort["chunks_retained"] == 5,
            "resume_skipped_bitmap": len(resume["skipped_indices"]) == 5
            and not resume["missing_indices"],
            "audit_after_v2": audit_v2["passed"],
            "snapshot_restore_whole": restore["mode"] == "whole-state-replacement"
            and restore["comparison"]["metadata_version_changes"]
            == ["archive/demo/sample.bin"],
            "audit_after_snapshot_restore": audit_after_restore["passed"],
            "restored_active_version": restored_object["active_count"] == 1
            and restored_object["objects"][0]["version"] == 2,
            "fault_truncate_E_CHUNK": truncate["error"]["code"] == "E_CHUNK",
            "fault_corrupt_E_INTEGRITY": corrupt["error"]["code"] == "E_INTEGRITY",
            "integrity_retry_succeeded": integrity_repaired["version"]["version"] == 1,
            "fault_version_E_VERSION": version_conflict["error"]["code"] == "E_VERSION",
            "forced_termination_unknown": crash_session["session_id"] in unknown_ids,
            "forced_termination_no_visible_object": crash_objects_before[
                "version_count"
            ]
            == 0,
            "forced_termination_rerun_committed": rerun_commit["version"]["version"]
            == 1
            and crash_objects_after["version_count"] == 1,
            "no_exit_report_claimed": recovery["exit_report_available"] is False,
            "dependency_graph_valid": final_graph["validation"]["valid"],
            "final_audit_passed": final_audit["passed"],
            "structured_error_codes_logged": all(value > 0 for value in codes.values()),
        }
        failed = [name for name, passed in checks.items() if not passed]
        report["checks"] = checks
        report["ok"] = not failed
        report["failed_checks"] = failed
        report["elapsed_ms"] = mono_ms() - started
        if failed:
            raise RuntimeError(f"acceptance checks failed: {failed}")
    finally:
        for running_process in list(processes):
            _stop_server(running_process)
        report["elapsed_ms"] = mono_ms() - started
        report["root"] = str(root)
        report.setdefault("ok", False)
        pretty = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        atomic_write_text(report_path, pretty)
        snapshot_step = report.get("steps", {}).get("snapshot", {})
        termination_step = report.get("steps", {}).get("forced_termination", {})
        final_graph = report.get("steps", {}).get("final_graph", {})
        final_status = report.get("steps", {}).get("final_status", {})
        stable_summary = {
            "schema": "chunkdeps-validation-summary/v1",
            "ok": report.get("ok", False),
            "failed_checks": report.get("failed_checks", []),
            "checks": report.get("checks", {}),
            "sample": {
                "size": report.get("sample", {}).get("size"),
                "sha256": report.get("sample", {}).get("sha256"),
                "declaration_sha256": report.get("sample", {}).get("declaration_sha256"),
            },
            "concurrency": {
                "global_limit": report.get("service", {}).get("global_concurrency"),
                "session_limit": report.get("service", {}).get("session_concurrency"),
                "observed_peak_global": report.get("steps", {})
                .get("concurrency", {})
                .get("observed_peak_global"),
            },
            "business_code_counts": report.get("business_codes", {}).get("counts", {}),
            "snapshot_restore": {
                "mode": snapshot_step.get("restore", {}).get("mode"),
                "metadata_version_changes": snapshot_step.get("restore", {})
                .get("comparison", {})
                .get("metadata_version_changes", []),
                "chunk_set_changes": snapshot_step.get("restore", {})
                .get("comparison", {})
                .get("chunk_set_changes", []),
            },
            "forced_termination": {
                "unknown_sessions": len(
                    termination_step.get("recovery", {}).get(
                        "unknown_merge_sessions", []
                    )
                ),
                "visible_versions_before_rerun": termination_step.get(
                    "visible_versions_before_rerun"
                ),
                "visible_versions_after_rerun": termination_step.get(
                    "visible_versions_after_rerun"
                ),
                "exit_report_available": termination_step.get(
                    "exit_report_available"
                ),
            },
            "final": {
                "counts": final_status.get("counts", {}),
                "graph_validation": final_graph,
                "audit_passed": report.get("steps", {})
                .get("final_audit", {})
                .get("passed"),
            },
            "full_report": str(report_path.resolve()),
        }
        atomic_write_json(report_path.with_name("validation_summary.json"), stable_summary)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the full local acceptance demonstration"
    )
    parser.add_argument(
        "--root", default=str(REPO_ROOT / "artifacts" / "demo-workbench")
    )
    parser.add_argument(
        "--report", default=str(REPO_ROOT / "artifacts" / "validation_report.json")
    )
    args = parser.parse_args(argv)
    report = run_demo(Path(args.root), Path(args.report))
    summary = {
        "ok": report["ok"],
        "report": str(Path(args.report).resolve()),
        "root": report["root"],
        "elapsed_ms": report["elapsed_ms"],
        "failed_checks": report["failed_checks"],
        "checks": report["checks"],
        "business_codes": report["business_codes"],
        "stable_summary": str(Path(args.report).with_name("validation_summary.json").resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
