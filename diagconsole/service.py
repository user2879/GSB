"""Deterministic cluster simulation service (one process per cluster)."""

from __future__ import annotations

import argparse
import json
import socket
import sys

from .aggregate import summarize_request
from .errors import DiagError

REQUIRED_REPLY_FIELDS = ("type", "cluster_id", "task_id", "version",
                         "aggregate", "digest")


def _send_line(handle, payload) -> None:
    handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    handle.flush()


def serve_cluster(cluster_id: str, host: str, port: int) -> None:
    """Run a single-threaded JSON-lines TCP service until the socket closes."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((host, port))
    except OSError:
        print(json.dumps({"type": "fatal", "code": "E_PORT",
                          "cluster_id": cluster_id, "port": port}))
        sys.stdout.flush()
        sys.exit(2)
    server.listen(8)
    print(json.dumps({"type": "ready", "cluster_id": cluster_id,
                      "host": host, "port": port}))
    sys.stdout.flush()
    while True:
        try:
            conn, _ = server.accept()
        except OSError:
            return
        with conn:
            stream = conn.makefile("rw", encoding="utf-8")
            for line in stream:
                try:
                    request = json.loads(line)
                    if request.get("type") == "shutdown":
                        _send_line(stream, {"type": "bye",
                                            "cluster_id": cluster_id})
                        stream.flush()
                        conn.shutdown(socket.SHUT_WR)
                        server.close()
                        return
                    summary = summarize_request(request)
                    reply = {
                        "type": "result",
                        "cluster_id": request["cluster_id"],
                        "task_id": request["task_id"],
                        "version": request["version"],
                        "aggregate": summary,
                    }
                    reply["digest"] = _reply_digest(reply)
                    _send_line(stream, reply)
                except (KeyError, ValueError, json.JSONDecodeError) as exc:
                    _send_line(stream, {"type": "error",
                                        "cluster_id": cluster_id,
                                        "message": str(exc)})


def _reply_digest(reply: dict) -> str:
    import hashlib

    from .errors import canonical_json

    body = {key: reply[key] for key in ("cluster_id", "task_id", "version",
                                        "aggregate")}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Cluster simulation service")
    parser.add_argument("--cluster-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        serve_cluster(args.cluster_id, args.host, args.port)
    except DiagError as exc:  # pragma: no cover - defensive
        print(json.dumps(exc.as_dict()))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
