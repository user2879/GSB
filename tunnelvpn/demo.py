"""Repeatable local demonstration orchestration."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .codec import encode_message, payload_digest_hex
from .config import Route, TunnelConfig
from .engine import TunnelEngine, send_inbound_payload
from .errors import ErrorCode, TunnelError
from .peer import read_json_line, send_frame
from .replay import ReplayVerifier
from .snapshot import SnapshotStore, restore_snapshot

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DemoHarness:
    def __init__(self, config: TunnelConfig, state_dir: str | Path) -> None:
        self.config = config
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.state_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.pid_file = self.state_dir / "demo-pids.json"
        self.processes: list[subprocess.Popen[bytes]] = []
        self.log_handles: list[Any] = []

    def cleanup_recorded_processes(self) -> list[int]:
        terminated: list[int] = []
        if not self.pid_file.exists():
            return terminated
        try:
            raw = json.loads(self.pid_file.read_text(encoding="utf-8"))
            entries = raw if isinstance(raw, list) else [raw]
        except (OSError, json.JSONDecodeError):
            return terminated
        for entry in entries:
            pid = entry.get("pid") if isinstance(entry, dict) else entry
            if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
                continue
            if _terminate_pid(pid):
                terminated.append(pid)
        try:
            self.pid_file.unlink(missing_ok=True)
        except OSError:
            pass
        return terminated

    def start_peers(self) -> list[dict[str, Any]]:
        started: list[dict[str, Any]] = []
        reachable: list[Any] = []
        for peer in self.config.peers:
            if peer.behavior == "unreachable":
                started.append(
                    {
                        "peer_id": peer.id,
                        "behavior": peer.behavior,
                        "pid": None,
                        "port": peer.endpoint.port,
                        "reachable": False,
                    }
                )
                continue
            log_path = self.log_dir / f"peer-{peer.id}.jsonl"
            stderr_path = self.log_dir / f"peer-{peer.id}.stderr.log"
            stdout_handle = open(stderr_path, "ab", buffering=0)
            self.log_handles.append(stdout_handle)
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "peer_server.py"),
                "--config",
                str(self.config_path()),
                "--peer-id",
                peer.id,
                "--log",
                str(log_path),
            ]
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=stdout_handle,
                stderr=subprocess.STDOUT,
            )
            self.processes.append(process)
            started.append(
                {
                    "peer_id": peer.id,
                    "behavior": peer.behavior,
                    "pid": process.pid,
                    "port": peer.endpoint.port,
                    "reachable": True,
                }
            )
            self._write_pid_file(started)
            reachable.append((peer, process))
        self._write_pid_file(started)
        for peer, process in reachable:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise TunnelError(
                        ErrorCode.E_CONFIG,
                        "对端子进程提前退出",
                        field=f"peers[{peer.id}]",
                        details={
                            "pid": process.pid,
                            "returncode": process.returncode,
                            "log": str(self.log_dir / f"peer-{peer.id}.stderr.log"),
                        },
                    )
                if _port_open(peer.endpoint.host, peer.endpoint.port):
                    break
                time.sleep(0.03)
            else:
                raise TunnelError(
                    ErrorCode.E_PEER,
                    "等待对端监听超时",
                    field=f"peers[{peer.id}].port",
                    details={"port": peer.endpoint.port},
                )
        return started

    def config_path(self) -> Path:
        return self.state_dir / "active-routes.json"

    def materialize_config(self) -> Path:
        path = self.config_path()
        path.write_text(
            json.dumps(self.config.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def _write_pid_file(self, entries: list[dict[str, Any]]) -> None:
        active = [entry for entry in entries if entry.get("pid") is not None]
        self.pid_file.write_text(
            json.dumps(active, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def stop(self) -> None:
        for process in reversed(self.processes):
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 3.0
        for process in reversed(self.processes):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.kill()
        for handle in self.log_handles:
            try:
                handle.close()
            except OSError:
                pass
        self.log_handles.clear()
        try:
            self.pid_file.unlink(missing_ok=True)
        except OSError:
            pass

    def __enter__(self) -> "DemoHarness":
        self.materialize_config()
        self.cleanup_recorded_processes()
        try:
            self.start_peers()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()


def _terminate_pid(pid: int) -> bool:
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
                check=False,
            )
            return True
        except (OSError, subprocess.SubprocessError):
            return False
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    try:
        cmdline = cmdline_path.read_bytes()
    except OSError:
        return False
    if b"peer_server.py" not in cmdline and b"run_demo.py" not in cmdline:
        return False
    try:
        os.kill(pid, 15)
    except ProcessLookupError:
        return False
    except OSError:
        return False
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        time.sleep(0.05)
    try:
        os.kill(pid, 9)
        return True
    except OSError:
        return False


def _port_open(host: str, port: int, timeout: float = 0.1) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _route(config: TunnelConfig, route_id: str) -> Route:
    route = config.route_map().get(route_id)
    if route is None:
        raise AssertionError(f"demo config missing route {route_id}")
    return route


def run_demo(
    config: TunnelConfig,
    *,
    state_dir: str | Path = "artifacts/demo-state",
    scenario: str = "all",
) -> dict[str, Any]:
    if scenario not in {"all", "normal", "queue", "fragment"}:
        raise ValueError("scenario must be all|normal|queue|fragment")
    state_path = Path(state_dir)
    state_path.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {"scenario": scenario, "steps": {}, "code": "OK"}
    fixture_checks: list[dict[str, Any]] = []
    engine: TunnelEngine | None = None
    with DemoHarness(config, state_path) as harness:
        engine = TunnelEngine(config, log_path=harness.log_dir / "engine.jsonl")
        engine.start()
        try:
            if scenario in {"all", "normal"}:
                results["steps"]["normal"] = _scenario_normal(config, engine)
            if scenario in {"all", "queue"}:
                results["steps"]["queue"] = _scenario_queue(config, engine)
            if scenario in {"all", "fragment"}:
                results["steps"]["fragment"] = _scenario_fragment(config, engine)
            if scenario == "all":
                results["steps"]["business_codes"] = _scenario_business_codes(
                    config, engine
                )
                results["steps"]["fragment_injection"] = _inject_missing_fragment(config)
            fixture_checks.append(
                _single_write_check(engine)
            )
            results["engine"] = engine.diagnostics()

            snapshot_store = SnapshotStore(state_path / "snapshots")
            snapshot = snapshot_store.save_engine(engine)
            results["snapshot"] = {
                "path": str(snapshot.path),
                "payload_sha256": snapshot.payload_digest,
                "route_version": snapshot.route_version,
                "sessions": len(snapshot.raw_sessions()),
                "snapshots": [entry.to_dict() for entry in snapshot_store.list_snapshots()],
            }
            restored = restore_snapshot(snapshot, config)
            results["restore"] = restored.report.to_dict()
            replay = ReplayVerifier(snapshot, current_config=config).verify()
            results["replay"] = replay.to_dict()
        finally:
            engine.stop()

        if "snapshot" in results:
            # Prove the restored state can be opened as a fresh engine after the
            # original process has stopped.  It is intentionally not merged
            # with the current process' ledger.
            restored = restore_snapshot(
                SnapshotStore(state_path / "snapshots").load(results["snapshot"]["path"]),
                config,
            )
            restarted = TunnelEngine(
                restored.config,
                ledger=restored.ledger,
                log_path=harness.log_dir / "engine-restarted.jsonl",
                sequence_watermarks=restored.sequence_watermarks,
                budget_usage=restored.budget_usage,
                restored_sessions=restored.sessions,
            )
            restarted.start()
            try:
                results["restart_check"] = {
                    "code": "OK",
                    "ledger_digest": restarted.ledger.digest(),
                    "session_count": len(restarted.sessions_snapshot()),
                    "unknown_sessions": [
                        item["session_id"]
                        for item in restarted.sessions_snapshot()
                        if item["requires_rerun"]
                    ],
                }
            finally:
                restarted.stop()

    results["fixture_checks"] = fixture_checks
    expected_codes = _collect_expected_codes(results["steps"])
    actual_codes = _collect_actual_codes(results["steps"])
    results["codes"] = {"expected": expected_codes, "actual": actual_codes}
    if scenario == "all":
        if results.get("replay", {}).get("code") != "OK":
            results["code"] = "E_REPLAY"
        elif not results["steps"]["fragment"]["fragment_roundtrip"]:
            results["code"] = "E_FRAG"
        elif results["steps"]["business_codes"].get("codes") != {
            "corrupt": "E_DIGEST",
            "dead": "E_PEER",
            "slow": "E_TIMEOUT",
        }:
            results["code"] = "E_PEER"
    else:
        step_name = {"normal": "normal", "queue": "queue", "fragment": "fragment"}[scenario]
        if not results["steps"].get(step_name, {}).get("ok", False):
            results["code"] = "E_REPLAY"

    output = state_path / "demo_result.json"
    results["result_path"] = str(output)
    output.write_text(
        json.dumps(results, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return results


def _scenario_normal(config: TunnelConfig, engine: TunnelEngine) -> dict[str, Any]:
    payload = (PROJECT_ROOT / "data" / "samples" / "payload-normal.txt").read_bytes()
    response = send_inbound_payload(_route(config, "alpha"), payload, timeout=20)
    return {
        "expected_code": "OK",
        "actual_code": response["code"],
        "payload_bytes": response["payload_bytes"],
        "fragment_count": response["fragment_count"],
        "payload_digest": response["payload_digest"],
        "ok": response["code"] == "OK" and response["fragment_count"] == 1,
    }


def _scenario_queue(config: TunnelConfig, engine: TunnelEngine) -> dict[str, Any]:
    pattern = (PROJECT_ROOT / "data" / "samples" / "payload-fragment.txt").read_bytes()
    payload = (pattern * 8)[:8192]
    route = _route(config, "alpha")
    observed = {"global": 0, "route": 0}
    monitoring = threading.Event()

    def monitor() -> None:
        while not monitoring.is_set():
            snapshot = engine.budget.snapshot()
            observed["global"] = max(observed["global"], snapshot.global_active)
            observed["route"] = max(
                observed["route"], snapshot.routes[route.id]["active_sessions"]
            )
            time.sleep(0.005)

    monitor_thread = threading.Thread(target=monitor, name="demo-budget-monitor", daemon=True)
    monitor_thread.start()
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(send_inbound_payload, route, payload, timeout=30) for _ in range(3)]
            responses = [future.result() for future in futures]
    finally:
        monitoring.set()
        monitor_thread.join(timeout=1.0)
    queued = [item for item in responses if item["queue_count"] > 0]
    records = [
        engine.ledger.get(item["session_id"]) for item in responses
    ]
    records = [item for item in records if item is not None]
    span_start = min(item.start_ms for item in records)
    span_end = max(item.end_ms for item in records)
    span_ms = max(1, span_end - span_start)
    wire_bytes = sum(item.wire_bytes for item in records)
    route_capacity = route.bandwidth_kbps * 1024
    global_capacity = config.global_limits.bandwidth_kbps * 1024
    route_allowed = route_capacity + route.bandwidth_kbps * 1024 * span_ms // 1000
    global_allowed = (
        global_capacity
        + config.global_limits.bandwidth_kbps * 1024 * span_ms // 1000
    )
    return {
        "expected_code": "OK",
        "actual_codes": [item["code"] for item in responses],
        "sessions": len(responses),
        "queued_sessions": len(queued),
        "queue_counts": [item["queue_count"] for item in responses],
        "total_queue_ms": [item["total_queue_ms"] for item in responses],
        "fragment_counts": [item["fragment_count"] for item in responses],
        "queue_observed": bool(queued),
        "per_tunnel_limit": route.max_sessions,
        "global_session_limit": config.global_limits.max_sessions,
        "max_observed_global_sessions": observed["global"],
        "max_observed_route_sessions": observed["route"],
        "wire_bytes": wire_bytes,
        "window_ms": span_ms,
        "route_bandwidth_bound_bytes": route_allowed,
        "global_bandwidth_bound_bytes": global_allowed,
        "concurrency_bounds_ok": (
            observed["global"] <= config.global_limits.max_sessions
            and observed["route"] <= route.max_sessions
        ),
        "bandwidth_bounds_ok": wire_bytes <= route_allowed and wire_bytes <= global_allowed,
        "ok": (
            all(item["code"] == "OK" for item in responses)
            and bool(queued)
            and observed["global"] <= config.global_limits.max_sessions
            and observed["route"] <= route.max_sessions
            and wire_bytes <= route_allowed
            and wire_bytes <= global_allowed
        ),
    }


def _scenario_fragment(config: TunnelConfig, engine: TunnelEngine) -> dict[str, Any]:
    payload = (PROJECT_ROOT / "data" / "samples" / "payload-fragment.txt").read_bytes()
    route = _route(config, "alpha")
    response = send_inbound_payload(route, payload, timeout=20)
    encoded = encode_message(
        tunnel_id=route.id,
        sequence=response["sequence"],
        cipher=route.cipher,
        mtu=route.mtu,
        payload=payload,
    )
    return {
        "expected_code": "OK",
        "actual_code": response["code"],
        "payload_bytes": len(payload),
        "fragment_count": response["fragment_count"],
        "expected_min_fragments": (len(payload) + (route.mtu - 84) - 1) // (route.mtu - 84),
        "payload_digest": payload_digest_hex(payload),
        "response_digest": response["payload_digest"],
        "frame_digests": list(encoded.frame_digests),
        "fragment_roundtrip": (
            response["code"] == "OK"
            and response["payload_digest"] == payload_digest_hex(payload)
            and response["fragment_count"] > 1
        ),
    }


def _scenario_business_codes(config: TunnelConfig, engine: TunnelEngine) -> dict[str, Any]:
    results: dict[str, Any] = {"codes": {}, "details": {}}
    for route_id in ("corrupt", "dead", "slow"):
        response = send_inbound_payload(_route(config, route_id), b"business-code", timeout=20)
        results["codes"][route_id] = response["code"]
        results["details"][route_id] = response
    results["ok"] = results["codes"] == {
        "corrupt": "E_DIGEST",
        "dead": "E_PEER",
        "slow": "E_TIMEOUT",
    }
    return results


def _inject_missing_fragment(config: TunnelConfig) -> dict[str, Any]:
    route = _route(config, "alpha")
    payload = b"fragment-injection-" * 80
    encoded = encode_message(
        tunnel_id=route.id,
        sequence=999,
        cipher=route.cipher,
        mtu=route.mtu,
        payload=payload,
    )
    if len(encoded.frames) < 3:
        raise AssertionError("fragment injection needs at least three frames")
    omitted_index = 1
    omitted_offset = omitted_index * (route.mtu - 84)
    with socket.create_connection(
        (route.target.host, route.target.port), timeout=2.0
    ) as sock:
        sock.settimeout(5.0)
        for index, frame in enumerate(encoded.frames):
            if index == omitted_index:
                continue
            send_frame(sock, frame)
        sock.shutdown(socket.SHUT_WR)
        reply = read_json_line(sock)
    missing = reply.get("missing_offsets", [])
    return {
        "expected_code": "E_FRAG",
        "actual_code": reply.get("code"),
        "omitted_offset": omitted_offset,
        "missing_offsets": missing,
        "ok": reply.get("code") == "E_FRAG" and omitted_offset in missing,
    }


def _collect_expected_codes(steps: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    for step in steps.values():
        if "expected_code" in step:
            codes.append(step["expected_code"])
        if "codes" in step and isinstance(step["codes"], dict):
            codes.extend(step["codes"].values())
    return codes


def _collect_actual_codes(steps: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    for step in steps.values():
        if "actual_code" in step:
            codes.append(step["actual_code"])
        if "actual_codes" in step:
            codes.extend(step["actual_codes"])
        if "codes" in step and isinstance(step["codes"], dict):
            codes.extend(step["codes"].values())
    return codes


def _single_write_check(engine: TunnelEngine) -> dict[str, Any]:
    records = engine.ledger.records()
    if not records:
        return {"name": "ledger_single_write", "ok": True, "reason": "no records"}
    try:
        engine.ledger.record(records[0])
    except TunnelError as exc:
        return {
            "name": "ledger_single_write",
            "ok": exc.code == "E_STATE",
            "code": exc.code,
            "ledger_digest": engine.ledger.digest(),
        }
    return {"name": "ledger_single_write", "ok": False, "reason": "duplicate accepted"}
