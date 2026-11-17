from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from typing import Any

from .determinism import compute_aggregate_summary
from .errors import DiagError, E_PORT, E_SPEC, StructuredLogger
from .spec import SystemSpec, load_spec
from .util import canonical_json, digest_json, sha256_text


def _bind_server(host: str, port: int) -> socket.socket:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        if os.name != "nt":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        elif hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        if family == socket.AF_INET6:
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        sock.bind((host, port))
        sock.listen(16)
        sock.settimeout(0.5)
        return sock
    except OSError:
        sock.close()
        raise


def _response_line(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _handle_command(
    command: dict[str, Any],
    *,
    spec: SystemSpec,
    cluster_id: str,
    session: str,
) -> tuple[dict[str, Any], bool]:
    kind = command.get("command")
    request_id = command.get("request_id")
    if kind == "ping":
        return {
            "type": "pong",
            "service": "cluster-sim",
            "cluster_id": cluster_id,
            "session": session,
            "request_id": request_id,
        }, False
    if kind == "shutdown":
        return {
            "type": "shutdown_ack",
            "cluster_id": cluster_id,
            "session": session,
            "request_id": request_id,
        }, True
    if kind != "distribute":
        return {
            "type": "error",
            "code": E_SPEC,
            "message": "unsupported command",
            "request_id": request_id,
        }, False

    revision = command.get("revision")
    task_id = command.get("task_id")
    if not isinstance(revision, int) or isinstance(revision, bool) or not isinstance(task_id, str):
        return {
            "type": "error",
            "code": E_SPEC,
            "message": "revision must be an integer and task_id must be a string",
            "field_path": "$.request",
            "request_id": request_id,
        }, False

    started_ns = time.monotonic_ns()
    try:
        summary = compute_aggregate_summary(
            spec,
            cluster_id=cluster_id,
            revision_version=revision,
            task_id=task_id,
        )
    except DiagError as exc:
        return {
            "type": "error",
            "code": exc.code,
            "message": exc.message,
            "field_path": exc.field_path,
            "details": exc.details,
            "request_id": request_id,
        }, False
    ended_ns = time.monotonic_ns()
    compute_ms = max(0, (ended_ns - started_ns) // 1_000_000)
    summary_canonical = canonical_json(summary)
    return {
        "type": "result",
        "service": "cluster-sim",
        "session": session,
        "request_id": request_id,
        "cluster_id": cluster_id,
        "revision": revision,
        "task_id": task_id,
        "aggregate_summary": summary,
        "aggregate_summary_canonical": summary_canonical,
        "aggregate_summary_digest": sha256_text(summary_canonical),
        "summary_object_digest": digest_json(summary),
        "compute_ms": int(compute_ms),
    }, False


def serve(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic local cluster simulator")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--cluster", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args(argv)

    logger = StructuredLogger(args.log)
    try:
        spec = load_spec(args.spec)
        cluster = spec.clusters_by_id.get(args.cluster)
        if cluster is None:
            raise DiagError(
                E_SPEC,
                f"unknown cluster {args.cluster!r}",
                field_path="$.clusters",
            )
    except DiagError as exc:
        logger.error(exc)
        return 2

    try:
        server = _bind_server(cluster.host, cluster.port)
    except OSError as exc:
        error = DiagError(
            E_PORT,
            f"cannot bind simulator port {cluster.host}:{cluster.port}: {exc}",
            field_path=f"$.clusters[cluster_id={cluster.id}].port",
            details={"host": cluster.host, "port": cluster.port, "errno": getattr(exc, "errno", None)},
        )
        logger.error(error)
        return 3

    logger.info(
        "SERVICE_READY",
        "deterministic simulator is listening",
        details={"cluster_id": cluster.id, "host": cluster.host, "port": cluster.port},
    )
    try:
        while True:
            try:
                connection, _address = server.accept()
            except socket.timeout:
                continue
            with connection:
                connection.settimeout(10.0)
                reader = connection.makefile("rb")
                writer = connection.makefile("wb")
                shutdown = False
                try:
                    for raw_line in reader:
                        if not raw_line.strip():
                            continue
                        try:
                            command = json.loads(raw_line.decode("utf-8"))
                            if not isinstance(command, dict):
                                raise ValueError("request must be a JSON object")
                            response, shutdown = _handle_command(
                                command,
                                spec=spec,
                                cluster_id=cluster.id,
                                session=args.session,
                            )
                        except Exception as exc:
                            response = {
                                "type": "error",
                                "code": E_SPEC,
                                "message": f"invalid request: {exc}",
                            }
                        writer.write(_response_line(response))
                        writer.flush()
                        if shutdown:
                            return 0
                except (BrokenPipeError, ConnectionResetError, socket.timeout):
                    continue
    finally:
        server.close()
        logger.info("SERVICE_STOPPED", "deterministic simulator stopped", details={"cluster_id": cluster.id})


def main() -> int:
    return serve(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
