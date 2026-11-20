from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .errors import UpstreamError
from .util import load_json_strict, require_int, require_list, require_mapping, require_string


@dataclass(frozen=True)
class UpstreamSpec:
    name: str
    port: int
    mode: str
    delay_ms: int
    failure_rate_percent: int


def load_upstream_config(path: str | Path) -> tuple[UpstreamSpec, ...]:
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise UpstreamError(
            "unable to read upstream configuration",
            "$",
            {"source": str(path), "reason": str(exc)},
        ) from exc
    try:
        document = load_json_strict(raw, str(path))
    except Exception as exc:
        raise UpstreamError(
            "invalid upstream configuration",
            "$",
            {"reason": str(exc)},
        ) from exc
    try:
        return _parse_upstream_document(document)
    except UpstreamError:
        raise
    except Exception as exc:
        raise UpstreamError(
            "invalid upstream configuration",
            "$",
            {"reason": str(exc)},
        ) from exc


def _parse_upstream_document(document: Any) -> tuple[UpstreamSpec, ...]:
    root = require_mapping(document, "$")
    raw_specs = require_list(root.get("instances"), "$.instances")
    specs: list[UpstreamSpec] = []
    seen_ports: set[int] = set()
    for index, item in enumerate(raw_specs):
        path_prefix = f"$.instances[{index}]"
        obj = require_mapping(item, path_prefix)
        name = require_string(obj.get("name"), f"{path_prefix}.name")
        port = require_int(
            obj.get("port"), f"{path_prefix}.port", minimum=1, maximum=65535
        )
        if port in seen_ports:
            raise UpstreamError(
                "duplicate simulated upstream port",
                f"{path_prefix}.port",
                {"port": port},
            )
        seen_ports.add(port)
        mode = require_string(obj.get("mode", "ok"), f"{path_prefix}.mode")
        if mode not in {"ok", "fail", "flaky", "slow"}:
            raise UpstreamError(
                "mode must be one of ok, fail, flaky, slow",
                f"{path_prefix}.mode",
            )
        delay_ms = require_int(
            obj.get("delay_ms", 0),
            f"{path_prefix}.delay_ms",
            minimum=0,
            maximum=60_000,
        )
        failure_rate_percent = require_int(
            obj.get("failure_rate_percent", 100 if mode == "fail" else 0),
            f"{path_prefix}.failure_rate_percent",
            minimum=0,
            maximum=100,
        )
        specs.append(
            UpstreamSpec(
                name=name,
                port=port,
                mode=mode,
                delay_ms=delay_ms,
                failure_rate_percent=failure_rate_percent,
            )
        )
    if not specs:
        raise UpstreamError("at least one upstream instance is required", "$.instances")
    return tuple(specs)


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    # POSIX servers need SO_REUSEADDR to rebind after TIME_WAIT. On Windows
    # SO_REUSEADDR can let two listeners share a port, so keep exclusive bind
    # semantics there and surface an occupied port as E_UPSTREAM.
    allow_reuse_address = sys.platform != "win32"


def _deterministic_failure(request_id: str, attempt: str, percent: int) -> bool:
    if percent <= 0:
        return False
    if percent >= 100:
        return True
    digest = hashlib.sha256(f"{request_id}|{attempt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 100 < percent


def _handler_factory(spec: UpstreamSpec) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

        def _write_json(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/__health":
                self._write_json(200, {"ok": True, "instance": spec.name, "port": spec.port})
                return
            query = parse_qs(parsed.query)
            request_id = (query.get("mesh_request_id") or ["unknown"])[0]
            attempt = (query.get("mesh_attempt") or ["0"])[0]
            if spec.delay_ms:
                time.sleep(spec.delay_ms / 1000.0)
            should_fail = spec.mode == "fail" or (
                spec.mode == "flaky"
                and _deterministic_failure(
                    request_id, attempt, spec.failure_rate_percent
                )
            )
            payload = {
                "ok": not should_fail,
                "instance": spec.name,
                "port": spec.port,
                "request_id": request_id,
                "attempt": attempt,
                "mode": spec.mode,
                "server_mono_ms": round(time.monotonic_ns() / 1_000_000.0, 3),
            }
            if should_fail:
                self._write_json(503, payload)
            else:
                self._write_json(200, payload)

    return Handler


class UpstreamRuntime:
    def __init__(self, servers: list[tuple[UpstreamSpec, _ReusableThreadingHTTPServer]], threads: list[threading.Thread]) -> None:
        self.servers = servers
        self.threads = threads

    @property
    def ports(self) -> list[int]:
        return sorted(spec.port for spec, _ in self.servers)

    def stop(self) -> None:
        for _, server in self.servers:
            server.shutdown()
            server.server_close()
        for thread in self.threads:
            thread.join(timeout=2)

    def wait(self) -> None:
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop()


def start_upstreams(path: str | Path) -> UpstreamRuntime:
    specs = load_upstream_config(path)
    servers: list[tuple[UpstreamSpec, _ReusableThreadingHTTPServer]] = []
    threads: list[threading.Thread] = []
    try:
        for index, spec in enumerate(specs):
            try:
                server = _ReusableThreadingHTTPServer(
                    ("127.0.0.1", spec.port), _handler_factory(spec)
                )
            except OSError as exc:
                raise UpstreamError(
                    "unable to bind deterministic upstream port",
                    f"$.instances[{index}].port",
                    {
                        "port": spec.port,
                        "instance": spec.name,
                        "reason": str(exc),
                    },
                ) from exc
            thread = threading.Thread(
                target=server.serve_forever,
                name=f"mesh-upstream-{spec.port}",
                daemon=True,
            )
            thread.start()
            servers.append((spec, server))
            threads.append(thread)
    except Exception:
        for _, server in servers:
            server.shutdown()
            server.server_close()
        raise
    return UpstreamRuntime(servers, threads)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic loopback upstreams")
    parser.add_argument("--config", default="samples/upstreams.json")
    args = parser.parse_args(argv)
    runtime = start_upstreams(args.config)
    print(json.dumps({"status": "running", "ports": runtime.ports}))
    runtime.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
