"""Deterministic loopback observation server used by samples and tests."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


MAX_RESPONSE_BYTES = 1_000_000


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


class Scenario:
    """Validated observation scenario keyed by node id."""

    def __init__(self, data: dict[str, Any], *, source: str = "<memory>"):
        if not isinstance(data, dict):
            raise ValueError(f"{source}: scenario must be an object keyed by node id")
        self.nodes: dict[str, dict[str, Any]] = {}
        for node_id in sorted(data):
            raw = data[node_id]
            if not isinstance(raw, dict):
                raise ValueError(f"{source}: scenario[{node_id!r}] must be an object")
            observations = raw.get("observations", {})
            if not isinstance(observations, dict):
                raise ValueError(f"{source}: {node_id}.observations must be an object")
            normalized: dict[str, list[dict[str, Any]]] = {}
            for key in sorted(observations):
                records = observations[key]
                if not isinstance(records, list):
                    raise ValueError(
                        f"{source}: {node_id}.observations[{key!r}] must be an array"
                    )
                normalized_records: list[dict[str, Any]] = []
                for index, record in enumerate(records):
                    if not isinstance(record, dict):
                        raise ValueError(
                            f"{source}: {node_id}.{key}[{index}] must be an object"
                        )
                    value = record.get("value")
                    if not (
                        (isinstance(value, int) and not isinstance(value, bool))
                        or isinstance(value, str)
                    ):
                        raise ValueError(
                            f"{source}: {node_id}.{key}[{index}].value must be int/string"
                        )
                    seq = record.get("observation_seq")
                    ts = record.get("timestamp_ms")
                    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
                        raise ValueError(
                            f"{source}: {node_id}.{key}[{index}].observation_seq invalid"
                        )
                    if not isinstance(ts, int) or isinstance(ts, bool) or ts < 0:
                        raise ValueError(
                            f"{source}: {node_id}.{key}[{index}].timestamp_ms invalid"
                        )
                    normalized_records.append(
                        {
                            "value": value,
                            "observation_seq": seq,
                            "timestamp_ms": ts,
                        }
                    )
                normalized[key] = normalized_records

            missing = raw.get("missing_keys", [])
            if not isinstance(missing, list) or not all(
                isinstance(item, str) for item in missing
            ):
                raise ValueError(f"{source}: {node_id}.missing_keys must be string array")
            duplicate_counts = raw.get("duplicate_counts", {})
            if not isinstance(duplicate_counts, dict):
                raise ValueError(f"{source}: {node_id}.duplicate_counts must be object")
            for key, count in duplicate_counts.items():
                if not isinstance(key, str) or not isinstance(count, int) or count < 0:
                    raise ValueError(
                        f"{source}: {node_id}.duplicate_counts values must be int >= 0"
                    )
            delays = raw.get("response_delays_ms", {})
            if not isinstance(delays, dict):
                raise ValueError(f"{source}: {node_id}.response_delays_ms must be object")
            for key, delay in delays.items():
                if (
                    not isinstance(key, str)
                    or not isinstance(delay, int)
                    or isinstance(delay, bool)
                    or delay < 0
                ):
                    raise ValueError(
                        f"{source}: {node_id}.response_delays_ms values must be int >= 0"
                    )
            default_delay = raw.get("default_response_delay_ms", 0)
            if not isinstance(default_delay, int) or isinstance(default_delay, bool) or default_delay < 0:
                raise ValueError(
                    f"{source}: {node_id}.default_response_delay_ms must be int >= 0"
                )
            invalid_json = bool(raw.get("invalid_json", False))
            self.nodes[node_id] = {
                "observations": normalized,
                "missing_keys": sorted(set(missing)),
                "duplicate_counts": duplicate_counts,
                "response_delays_ms": delays,
                "default_response_delay_ms": default_delay,
                "invalid_json": invalid_json,
            }

    @classmethod
    def load(cls, path: str | Path) -> "Scenario":
        scenario_path = Path(path)
        return cls(json.loads(scenario_path.read_text(encoding="utf-8")), source=str(scenario_path))

    def node(self, node_id: str) -> dict[str, Any]:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise ValueError(f"scenario has no node {node_id!r}") from exc

    def observations_for(
        self, node_id: str, key: str
    ) -> tuple[list[dict[str, Any]], int, bool]:
        node = self.node(node_id)
        delay = int(node["response_delays_ms"].get(key, node["default_response_delay_ms"]))
        if key in node["missing_keys"]:
            return [], delay, True
        records = node["observations"].get(key)
        if records is None:
            return [], delay, True
        copies = int(node["duplicate_counts"].get(key, 1))
        if copies < 1:
            copies = 1
        expanded: list[dict[str, Any]] = []
        for _ in range(copies):
            for record in records:
                expanded.append(dict(record))
        return expanded, delay, False


class _SimulatorHandler(BaseHTTPRequestHandler):
    server_version = "LocalConfigObservation/1"
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    @property
    def simulator_server(self) -> "SimulatorHTTPServer":
        return self.server  # type: ignore[return-value]

    def _send_json(self, status: HTTPStatus, payload: Any) -> None:
        body = _canonical_json(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "node_id": self.simulator_server.node_id,
                    "invalid_json": self.simulator_server.invalid_json,
                },
            )
            return
        if parsed.path != "/observe":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        params = parse_qs(parsed.query, keep_blank_values=True)
        key = params.get("key", [""])[0]
        window = params.get("window", [""])[0]
        parameter_version = params.get("parameter_version", [""])[0]
        if not key:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "missing_key_parameter"})
            return

        records, delay_ms, missing = self.simulator_server.scenario.observations_for(
            self.simulator_server.node_id, key
        )
        if delay_ms:
            time.sleep(delay_ms / 1000.0)
        if self.simulator_server.invalid_json:
            body = b'{"observations": [invalid-json'
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)
            return
        if missing:
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {
                    "error": "missing_key",
                    "node_id": self.simulator_server.node_id,
                    "key": key,
                    "window_id": window,
                    "parameter_version": parameter_version,
                },
            )
            return

        observations = [
            {
                "node_id": self.simulator_server.node_id,
                "key": key,
                **record,
            }
            for record in records
        ]
        self._send_json(
            HTTPStatus.OK,
            {
                "node_id": self.simulator_server.node_id,
                "key": key,
                "window_id": window,
                "parameter_version": parameter_version,
                "observations": observations,
            },
        )


class SimulatorHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        scenario: Scenario,
        node_id: str,
        *,
        invalid_json: bool = False,
    ):
        self.scenario = scenario
        self.node_id = node_id
        self.invalid_json = invalid_json
        super().__init__(address, _SimulatorHandler)


def start_server_in_thread(
    scenario: Scenario,
    *,
    node_id: str,
    port: int,
    invalid_json: bool = False,
) -> tuple[SimulatorHTTPServer, threading.Thread]:
    server = SimulatorHTTPServer(
        ("127.0.0.1", port),
        scenario,
        node_id,
        invalid_json=invalid_json,
    )
    thread = threading.Thread(target=server.serve_forever, name=f"sim-{node_id}", daemon=True)
    thread.start()
    return server, thread


def reserve_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_health(port: int, *, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2) as sock:
                request = (
                    "GET /health HTTP/1.1\r\n"
                    "Host: 127.0.0.1\r\n"
                    "Connection: close\r\n\r\n"
                )
                sock.sendall(request.encode("ascii"))
                data = sock.recv(4096)
                if b"200" in data and b'"ok"' in data:
                    return
        except OSError as exc:
            last_error = exc
        time.sleep(0.03)
    raise RuntimeError(f"simulator on port {port} did not become ready: {last_error}")


class SimulatorProcess:
    """Subprocess wrapper used by the end-to-end demo runner."""

    def __init__(
        self,
        *,
        scenario_path: str | Path,
        node_id: str,
        port: int,
        invalid_json: bool = False,
    ):
        self.node_id = node_id
        self.port = port
        package_root = Path(__file__).resolve().parent.parent
        scenario_abs = Path(scenario_path).resolve()
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            str(package_root)
            if not existing_pythonpath
            else str(package_root) + os.pathsep + existing_pythonpath
        )
        command = [
            sys.executable,
            "-m",
            "config_ledger.simulator",
            "--scenario",
            str(scenario_abs),
            "--node-id",
            node_id,
            "--port",
            str(port),
        ]
        if invalid_json:
            command.append("--invalid-json")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(package_root),
            env=environment,
        )
        try:
            wait_for_health(port)
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)

    def __enter__(self) -> "SimulatorProcess":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic loopback observation simulator")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--invalid-json", action="store_true")
    parser.add_argument(
        "--delay-ms",
        type=int,
        help="override the default response delay for every key on this node",
    )
    parser.add_argument(
        "--duplicate-count",
        type=int,
        help="override duplicate_counts for every observed key on this node",
    )
    parser.add_argument(
        "--missing-key",
        action="append",
        default=[],
        help="force a key to return HTTP 404 missing_key (repeatable)",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.delay_ms is not None and args.delay_ms < 0:
        parser.error("--delay-ms must be >= 0")
    if args.duplicate_count is not None and args.duplicate_count < 1:
        parser.error("--duplicate-count must be >= 1")
    scenario = Scenario.load(args.scenario)
    node_config = scenario.node(args.node_id)
    if args.delay_ms is not None:
        node_config["default_response_delay_ms"] = args.delay_ms
    if args.duplicate_count is not None:
        for key in sorted(node_config["observations"]):
            node_config["duplicate_counts"][key] = args.duplicate_count
    if args.missing_key:
        node_config["missing_keys"] = sorted(
            set(node_config["missing_keys"]) | set(args.missing_key)
        )
    server = SimulatorHTTPServer(
        ("127.0.0.1", args.port),
        scenario,
        args.node_id,
        invalid_json=args.invalid_json,
    )
    print(
        _canonical_json(
            {
                "event": "simulator_started",
                "host": "127.0.0.1",
                "node_id": args.node_id,
                "port": args.port,
                "invalid_json": args.invalid_json,
                "delay_ms_override": args.delay_ms,
                "duplicate_count_override": args.duplicate_count,
                "missing_key_overrides": sorted(set(args.missing_key)),
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
