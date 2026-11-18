"""Reproducible three-scenario local coordination experiment."""

from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .client import request_json
from .inject import main as inject_main
from .util import JsonlLogger, atomic_write_json, canonical_json


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DemoFailure(RuntimeError):
    pass


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def free_high_port(preferred: int) -> int:
    for port in range(preferred, preferred + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise DemoFailure("no free high loopback port found")


def call(port: int, payload: dict[str, Any], timeout: float = 3.0) -> dict[str, Any]:
    response = request_json("127.0.0.1", port, payload, timeout=timeout)
    if not response.get("ok"):
        raise DemoFailure(
            f"request failed: {response.get('code')} {response.get('message')} {response.get('details')}"
        )
    return response


def wait_for(
    predicate: Callable[[], Any],
    *,
    timeout: float = 12.0,
    interval: float = 0.05,
    description: str = "condition",
) -> Any:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            value = predicate()
            if value:
                return value
        except Exception as exc:
            last_error = exc
        time.sleep(interval)
    suffix = f"; last error: {last_error}" if last_error else ""
    raise DemoFailure(f"timed out waiting for {description}{suffix}")


class Demo:
    def __init__(self, root: Path, port: int, heartbeat_ms: int = 300) -> None:
        self.root = root
        self.port = port
        self.heartbeat_ms = heartbeat_ms
        self.logger = JsonlLogger(root / "logs" / "demo.jsonl")
        self.server: subprocess.Popen[bytes] | None = None
        self.server_log_handle: Any = None
        self.nodes: dict[str, subprocess.Popen[bytes]] = {}
        self.node_log_handles: dict[str, Any] = {}
        self.node_sessions: dict[str, int] = {}

    def start_server(self) -> None:
        log_path = self.root / "logs" / "server_process.log"
        self.server_log_handle = log_path.open("ab")
        command = [
            sys.executable,
            "-m",
            "coord_mw.server",
            "--root",
            str(self.root / "server"),
            "--port",
            str(self.port),
            "--members",
            str(PROJECT_ROOT / "sample_members.json"),
        ]
        self.server = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=self.server_log_handle,
            stderr=subprocess.STDOUT,
        )

        def ping() -> bool:
            if self.server and self.server.poll() is not None:
                text = log_path.read_text(encoding="utf-8", errors="replace")
                raise DemoFailure(f"server exited early: {text.strip()}")
            response = request_json("127.0.0.1", self.port, {"op": "ping"}, timeout=0.4)
            return bool(response.get("ok"))

        wait_for(ping, timeout=10, description="server readiness")
        self.logger.write(
            "server_started", port=self.port, root=str(self.root / "server")
        )

    def start_node(self, member: str) -> subprocess.Popen[bytes]:
        session = self.node_sessions.get(member, 0) + 1
        self.node_sessions[member] = session
        log_path = self.root / "logs" / "nodes" / f"node_{member}_session_{session}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("ab")
        command = [
            sys.executable,
            "-m",
            "coord_mw.node",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--member",
            member,
            "--heartbeat-ms",
            str(self.heartbeat_ms),
        ]
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        self.nodes[member] = process
        self.node_log_handles[member] = handle
        self.logger.write(
            "node_started",
            member=member,
            session=session,
            recovery_source="tcp_persisted_state",
        )
        return process

    def state(self) -> dict[str, Any]:
        return call(self.port, {"op": "query_state"})

    def leader(self) -> dict[str, Any]:
        return call(self.port, {"op": "query_leader"})

    def stop(self) -> None:
        for member, process in list(self.nodes.items()):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            handle = self.node_log_handles.pop(member, None)
            if handle is not None:
                handle.close()
        self.nodes.clear()
        if self.server and self.server.poll() is None:
            try:
                call(self.port, {"op": "shutdown"}, timeout=2.0)
            except Exception:
                self.server.terminate()
            try:
                self.server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server.kill()
                self.server.wait(timeout=5)
        if self.server_log_handle is not None:
            self.server_log_handle.close()
            self.server_log_handle = None


def run_demo(root: Path, port: int, heartbeat_ms: int) -> dict[str, Any]:
    reset_directory(root)
    demo = Demo(root, port, heartbeat_ms)
    scenarios: list[dict[str, Any]] = []
    try:
        demo.start_server()
        for member in ["A", "B", "C", "D", "E"]:
            demo.start_node(member)

        def all_initial_members_active() -> dict[str, Any] | None:
            state = demo.state()
            active = {
                member_id
                for member_id, member in state["members"].items()
                if member["status"] == "active"
                and member["lease_valid"]
                and member["candidate_eligible"]
            }
            if active == set(state["members"]):
                if state["election"].get("leader") == "A" and not state[
                    "election"
                ].get("requires_rerun"):
                    return state
            return None

        initial_state = wait_for(
            all_initial_members_active,
            timeout=15,
            description="A-E active and A elected",
        )
        scenario_one = {
            "name": "01_first_election",
            "term": initial_state["term"],
            "leader": initial_state["election"]["leader"],
            "eligible": initial_state["election"]["eligible"],
            "vote_counts": initial_state["election"]["vote_counts"],
            "members": sorted(initial_state["members"]),
        }
        demo.logger.write("scenario_result", **scenario_one)
        scenarios.append(scenario_one)

        before_change = demo.state()
        unaffected_before = {
            member_id: {
                "term": before_change["members"][member_id]["term"],
                "candidate_eligible": before_change["members"][member_id][
                    "candidate_eligible"
                ],
                "invalidation_version": before_change["members"][member_id][
                    "invalidation_version"
                ],
            }
            for member_id in ["D", "E"]
        }
        change = call(
            demo.port,
            {
                "op": "apply_change",
                "changes": {
                    "remove": ["A"],
                },
            },
        )
        after_change = demo.state()
        if change["affected_order"] != ["B", "C"]:
            raise DemoFailure(f"unexpected affected order: {change['affected_order']}")
        for member_id in ["B", "C"]:
            member = after_change["members"][member_id]
            if member["candidate_eligible"] or member["lease_valid"]:
                raise DemoFailure(f"downstream member {member_id} was not invalidated")
            if member["invalidation_version"] != 1:
                raise DemoFailure(f"downstream member {member_id} has wrong invalidation version")
        for member_id, before in unaffected_before.items():
            member = after_change["members"][member_id]
            after = {
                "term": member["term"],
                "candidate_eligible": member["candidate_eligible"],
                "invalidation_version": member["invalidation_version"],
            }
            if after != before:
                raise DemoFailure(
                    f"unaffected member {member_id} changed: {before} -> {after}"
                )
        if after_change["election"]["leader"] != "D":
            raise DemoFailure("leader did not move to D after removing A")
        snapshot_two = call(
            demo.port,
            {"op": "snapshot", "path": "snapshots/after_invalidation.json"},
        )
        scenario_two = {
            "name": "02_incremental_dependency_invalidation",
            "change_set": {"remove": ["A"]},
            "affected_order": change["affected_order"],
            "invalidation_versions": {
                member_id: after_change["members"][member_id]["invalidation_version"]
                for member_id in ["B", "C"]
            },
            "unaffected": [
                {
                    "id": member_id,
                    **unaffected_before[member_id],
                }
                for member_id in ["D", "E"]
            ],
            "leader": after_change["election"]["leader"],
            "trace": change["traces"][0],
            "snapshot": {
                "path": snapshot_two["path"],
                "snapshot_version": snapshot_two["snapshot_version"],
            },
        }
        demo.logger.write("scenario_result", **scenario_two)
        scenarios.append(scenario_two)

        killed = demo.nodes.pop("D")
        killed.kill()
        killed_exit = killed.wait(timeout=5)
        handle = demo.node_log_handles.pop("D", None)
        if handle is not None:
            handle.close()
        demo.logger.write(
            "node_forced_kill",
            member="D",
            exit_code=killed_exit,
            exit_report="unavailable",
        )

        def d_reaped_and_e_leader() -> dict[str, Any] | None:
            state = demo.state()
            member = state["members"].get("D")
            if member is None:
                return None
            if (
                not member["lease_valid"]
                and not member["candidate_eligible"]
                and state["election"].get("leader") == "E"
            ):
                return state
            return None

        reaped_state = wait_for(
            d_reaped_and_e_leader,
            timeout=10,
            description="D lease reclaimed and E elected",
        )
        demo.start_node("D")

        def d_recovered() -> dict[str, Any] | None:
            state = demo.state()
            member = state["members"].get("D")
            if member and member["status"] == "active" and member["lease_valid"]:
                if state["election"].get("leader") == "D":
                    return state
            return None

        recovered_state = wait_for(d_recovered, timeout=10, description="D restart recovery")

        final_snapshot = call(
            demo.port,
            {"op": "snapshot", "path": "snapshots/after_recovery.json"},
        )
        call(
            demo.port,
            {
                "op": "apply_change",
                "changes": {"lease": [{"id": "E", "lease_ms": 2600}]},
            },
        )
        restored = call(
            demo.port,
            {
                "op": "restore_snapshot",
                "path": final_snapshot["path"],
            },
        )
        if not restored.get("replaced") or restored.get("replacement_mode") != "whole_state":
            raise DemoFailure("snapshot was not restored by whole-state replacement")
        if not restored.get("replay", {}).get("ok"):
            raise DemoFailure("snapshot trace replay did not pass")

        def restored_state_continues() -> dict[str, Any] | None:
            state = demo.state()
            if "E" in state["members"] and state["members"]["E"]["lease_ms"] == 2200:
                if state["election"].get("leader") is not None:
                    return state
            return None

        continued_state = wait_for(
            restored_state_continues,
            timeout=10,
            description="requests continue after snapshot replacement",
        )
        scenario_three = {
            "name": "03_forced_kill_lease_reclaim_and_restart",
            "killed_member": "D",
            "killed_exit_code": killed_exit,
            "exit_report": "unavailable",
            "reaped_leader": reaped_state["election"]["leader"],
            "reaped_member": {
                "id": "D",
                "status": reaped_state["members"]["D"]["status"],
                "lease_valid": reaped_state["members"]["D"]["lease_valid"],
                "invalidation_version": reaped_state["members"]["D"][
                    "invalidation_version"
                ],
            },
            "restarted_member": {
                "id": "D",
                "status": recovered_state["members"]["D"]["status"],
                "lease_valid": recovered_state["members"]["D"]["lease_valid"],
                "recovered_from": "tcp_persisted_state",
            },
            "recovered_leader": recovered_state["election"]["leader"],
            "snapshot_reopen": {
                "snapshot_version": restored["snapshot_version"],
                "previous_state_version": restored["previous_state_version"],
                "version_mismatch": restored["version_mismatch"],
                "replaced": restored["replaced"],
                "replacement_mode": restored["replacement_mode"],
                "replay": restored["replay"],
                "continued_leader": continued_state["election"]["leader"],
            },
        }
        demo.logger.write("scenario_result", **scenario_three)
        scenarios.append(scenario_three)

        summary = {
            "ok": True,
            "status": "passed",
            "port": demo.port,
            "root": str(root),
            "scenarios": scenarios,
            "checks": {
                "first_election_stable": True,
                "incremental_invalidations_stable": True,
                "unaffected_members_unchanged": True,
                "forced_kill_reclaimed": True,
                "restart_recovered_from_persisted_state": True,
                "snapshot_whole_replacement": True,
                "trace_replay_passed": True,
            },
            "artifacts": {
                "engine_log": str(root / "server" / "logs" / "engine.jsonl"),
                "server_log": str(root / "server" / "logs" / "server.jsonl"),
                "demo_log": str(root / "logs" / "demo.jsonl"),
                "snapshots": str(root / "server" / "snapshots"),
            },
        }
        atomic_write_json(root / "summary.json", summary)
        demo.logger.write("demo_summary", **summary)
        return summary
    finally:
        demo.stop()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local coordination experiment")
    parser.add_argument("--root", default="demo_runtime")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--heartbeat-ms", type=int, default=300)
    parser.add_argument(
        "--inject",
        choices=["cycle", "snapshot", "port", "all"],
        default=None,
        help="run deterministic fault injection instead of the main demo",
    )
    args = parser.parse_args(argv)
    if args.inject:
        inject_args = ["--case", args.inject, "--base", str(Path(args.root) / "injections")]
        return inject_main(inject_args)
    if args.heartbeat_ms <= 0:
        print(
            canonical_json(
                {"ok": False, "error": "E_PROTO", "message": "heartbeat must be positive"}
            ),
            flush=True,
        )
        return 2
    port = args.port if args.port is not None else free_high_port(28765)
    if not (1024 <= port <= 65535):
        print(
            canonical_json(
                {"ok": False, "error": "E_PORT", "message": "port must be high"}
            ),
            flush=True,
        )
        return 2
    try:
        summary = run_demo(Path(args.root), port, args.heartbeat_ms)
    except Exception as exc:
        failure = {
            "ok": False,
            "status": "failed",
            "error": type(exc).__name__,
            "message": str(exc),
        }
        print(canonical_json(failure), flush=True)
        return 1
    print(canonical_json(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

