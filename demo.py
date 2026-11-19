from __future__ import annotations

import dataclasses
import json
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from meshdiag.canonical import canonical_json, pretty_json
from meshdiag.config import load_requests, load_revision_document, load_workspace
from meshdiag.engine import Engine
from meshdiag.errors import MeshError
from meshdiag.models import NormalizedRevision, normalize_revision
from meshdiag.replay import replay_snapshot


ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
VAR = ROOT / "var"
PORTS = list(range(18181, 18189))


def _port_free(port: int) -> bool:
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _wait_ports(process: subprocess.Popen, timeout: float = 6.0) -> None:
    deadline = time.monotonic() + timeout
    pending = set(PORTS)
    while pending and time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"模拟上游提前退出，code={process.returncode}")
        for port in sorted(tuple(pending)):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    pending.remove(port)
            except OSError:
                pass
        if pending:
            time.sleep(0.05)
    if pending:
        raise RuntimeError(f"模拟上游端口未就绪: {sorted(pending)}")


def _reset_upstream() -> None:
    for port in PORTS:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/__admin/reset", timeout=1.0
        ) as response:
            response.read()


def _write_active_workspace(workspace) -> Path:
    payload = {
        "workspace_id": workspace.workspace_id,
        "global_concurrency": workspace.global_concurrency,
        "revision_file": str((SAMPLES / "revision-r2-v3.json").resolve()),
        "requests_file": str((SAMPLES / "requests-hot.json").resolve()),
        "ledger_path": str(workspace.ledger_path),
        "snapshot_dir": str(workspace.snapshot_dir),
        "defaults": workspace.defaults,
        "services": [service.as_dict() for service in workspace.services],
    }
    path = VAR / "demo-workspace.json"
    path.write_text(pretty_json(payload) + "\n", encoding="utf-8")
    return path


def _print_revision_summary(label: str, revision: NormalizedRevision) -> None:
    print(f"SCENARIO {label}")
    print(f"DIGEST {revision.digest}")
    print(f"DEPENDENCY_GRAPH_VERSION {revision.dependency_graph.version}")
    for rule in revision.rules:
        print(
            f"RULE {rule.rule_id} revision={rule.revision_version} "
            f"format=v{rule.format_version} target={rule.target_service} "
            f"timeout_ms={rule.timeout_ms} retries={rule.retries} "
            f"threshold={rule.circuit_threshold} cooldown_ms={rule.cooldown_ms} "
            f"weights={json.dumps(dict(rule.instance_weights), sort_keys=True)}"
        )
    for source, target, kind in revision.dependency_graph.edges:
        print(f"EDGE {source} -[{kind}]-> {target}")


def _print_round_summary(summary) -> None:
    print(f"ROUND {summary.round_id} revision={summary.revision_version}")
    print(
        "OUTCOMES "
        f"success={summary.success} failure={summary.failure} "
        f"timeout={summary.timeout} circuit_open={summary.circuit_open} "
        f"unknown={summary.unknown}"
    )
    print(
        "CONCURRENCY "
        f"max_global={summary.max_global_active} "
        f"max_service={json.dumps(summary.max_service_active, sort_keys=True)}"
    )
    print(
        "CHANGE "
        + json.dumps(summary.change.as_dict(), sort_keys=True, ensure_ascii=False)
    )
    for result in summary.results:
        print(
            f"REQUEST {result.request_index:03d} {result.request_id} "
            f"outcome={result.outcome} reason={result.reason} "
            f"attempts={result.attempt_count} "
            f"instances={','.join(result.instance_sequence) or '-'}"
        )


def main() -> int:
    busy = [port for port in PORTS if not _port_free(port)]
    if busy:
        print(f"ERROR PORTS_BUSY {busy}", file=sys.stderr)
        return 2
    shutil.rmtree(VAR, ignore_errors=True)
    (VAR / "snapshots").mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [sys.executable, str(ROOT / "mock_upstream.py"), "--ports", ",".join(map(str, PORTS))],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_ports(process)
        base_workspace = load_workspace(SAMPLES / "workspace.json")
        active_workspace_path = _write_active_workspace(base_workspace)
        workspace = load_workspace(active_workspace_path)
        r1_doc = load_revision_document(SAMPLES / "revision-r1.json")
        r2_doc = load_revision_document(SAMPLES / "revision-r2-v3.json")
        cold_requests = load_requests(SAMPLES / "requests-cold.json")
        hot_requests = load_requests(SAMPLES / "requests-hot.json")
        r1 = normalize_revision(
            r1_doc,
            workspace.services,
            workspace_id=workspace.workspace_id,
            defaults=workspace.defaults,
        )
        r2 = normalize_revision(
            r2_doc,
            workspace.services,
            workspace_id=workspace.workspace_id,
            defaults=workspace.defaults,
        )
        _print_revision_summary("normalize-three-formats", r1)
        with Engine(workspace) as engine:
            cold = engine.run_round(
                round_id="demo-cold-r1",
                snapshot_id="snap-cold-r1",
                revision_document=r1_doc,
                requests=cold_requests,
            )
        _print_round_summary(cold)
        _reset_upstream()
        _print_revision_summary("upgrade-all-rules-to-v3", r2)
        with Engine(workspace) as engine:
            hot = engine.run_round(
                round_id="demo-hot-r2",
                snapshot_id="snap-hot-r2",
                revision_document=r2_doc,
                requests=hot_requests,
            )
            _print_round_summary(hot)
            rounds = engine.ledger.list_rounds()
            print(
                "HISTORY "
                + json.dumps(
                    [
                        {
                            "round_id": row["round_id"],
                            "revision": row["revision_version"],
                            "stale": bool(row["stale"]),
                        }
                        for row in rounds
                    ],
                    sort_keys=True,
                    ensure_ascii=False,
                )
            )
        _reset_upstream()
        print("SCENARIO failure-injection-replay")
        with Engine(workspace) as engine:
            replay = replay_snapshot(engine, "snap-hot-r2", current_revision=r2)
            print(
                "REPLAY_MATCH "
                f"snapshot={replay.snapshot_id} "
                f"requests={replay.requests_compared} "
                f"ledger_requests={replay.ledger_requests_compared} "
                f"ledger_attempts={replay.ledger_attempts_compared}"
            )
        print("SCENARIO forced-termination-recovery")
        crash = subprocess.run(
            [
                sys.executable,
                "-m",
                "meshdiag",
                "run",
                "--workspace",
                str(active_workspace_path),
                "--requests",
                str(SAMPLES / "requests-hot.json"),
                "--round-id",
                "demo-crash-r2",
                "--snapshot-id",
                "snap-crash-r2",
                "--crash-after",
                "3",
            ],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"CRASH_EXIT {crash.returncode}")
        with Engine(workspace) as engine:
            recovery = engine.recover_snapshot("snap-crash-r2")
            print(
                f"RECOVERY {recovery.consistency} completed={len(recovery.completed_request_ids)} "
                f"unknown={len(recovery.unknown_request_ids)} rerun_required={len(recovery.rerun_required)}"
            )
            for basis in recovery.basis:
                print(f"BASIS {basis}")
            restored = engine.recover_snapshot("snap-hot-r2")
            print(
                f"RESTORE_COMPLETED {restored.consistency} "
                f"unknown={len(restored.unknown_request_ids)} ledger_replaced={restored.ledger_replaced}"
            )
        _reset_upstream()
        with Engine(workspace) as engine:
            replay_after_restore = replay_snapshot(
                engine, "snap-hot-r2", current_revision=r2
            )
            print(
                "REPLAY_AFTER_RESTORE "
                f"requests={replay_after_restore.requests_compared} "
                f"ledger_attempts={replay_after_restore.ledger_attempts_compared}"
            )
            snapshots = engine.snapshots.list()
            print(
                "SNAPSHOTS "
                + json.dumps(
                    [
                        {
                            "id": row["snapshot_id"],
                            "status": row["status"],
                            "revision": row["revision_version"],
                            "completed": row["completed"],
                            "in_progress": row["in_progress"],
                        }
                        for row in snapshots
                    ],
                    sort_keys=True,
                    ensure_ascii=False,
                )
            )
        print("SCENARIO negative-control-instance-drift")
        drifted_services = []
        for service in workspace.services:
            if service.service_id == "checkout":
                instances = list(service.instances)
                instances[-1] = dataclasses.replace(instances[-1], port=18189)
                service = dataclasses.replace(service, instances=tuple(instances))
            drifted_services.append(service)
        drifted_workspace = dataclasses.replace(
            workspace, services=tuple(drifted_services)
        )
        with Engine(drifted_workspace) as engine:
            try:
                replay_snapshot(engine, "snap-hot-r2", current_revision=r2)
            except MeshError as exc:
                detail = exc.diagnostics[0].as_dict() if exc.diagnostics else {}
                print(
                    "REPLAY_REJECTED "
                    f"code={exc.code} field={detail.get('field')} "
                    f"basis={detail.get('basis')}"
                )
        print("DEMO_OK")
        return 0
    finally:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


if __name__ == "__main__":
    raise SystemExit(main())
