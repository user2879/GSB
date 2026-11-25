"""Deterministic inference backend run as a gateway-managed subprocess.

Wire protocol (one JSON object per line, UTF-8, loopback only):
  client -> {"id": str, "vector": [floats], "param_summary": str,
             "model_dim": int, "hidden_dim": int}
  server -> {"id": str, "ok": true, "values": [floats], "summary": str}
             or {"id": str, "ok": false, "error": str}

The process prints ``READY <port>`` on stdout once listening so the parent can
wait deterministically without sleeps.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys

from .constants import LOOPBACK_HOST
from .inference import deterministic_output, output_summary_from_values


def _readline(conn: socket.socket) -> bytes:
    chunks = []
    while True:
        piece = conn.recv(65536)
        if not piece:
            break
        chunks.append(piece)
        if b"\n" in piece:
            break
    return b"".join(chunks)


def serve(host: str, port: int) -> None:
    if host != LOOPBACK_HOST:
        raise SystemExit("backend must bind to 127.0.0.1")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(16)
    print(f"READY {port}", flush=True)
    try:
        while True:
            conn, _addr = sock.accept()
            with conn:
                data = _readline(conn)
                if not data:
                    continue
                line = data.split(b"\n", 1)[0]
                try:
                    req = json.loads(line.decode("utf-8"))
                    values = deterministic_output(
                        req["vector"], req["model_dim"],
                        req["hidden_dim"], req["param_summary"])
                    resp = {"id": req["id"], "ok": True, "values": values,
                            "summary": output_summary_from_values(values)}
                except Exception as exc:  # protocol-level, never crash the proc
                    resp = {"id": req.get("id", "?"), "ok": False,
                            "error": str(exc)}
                conn.sendall(
                    (json.dumps(resp, ensure_ascii=False) + "\n")
                    .encode("utf-8"))
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=LOOPBACK_HOST)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args(argv)
    serve(args.host, args.port)


if __name__ == "__main__":
    main(sys.argv[1:])
