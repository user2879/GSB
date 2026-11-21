"""Deterministic simulated service instance (runs as a child process).

Usage:
  python -m quota_workbench.instance_server --instance-id svc-a --port 18301 \
      --behavior normal|refused|delay|error [--delay-ms 5000]

Behaviors:
  normal  - bind, accept, reply "OK <instance_id> <seq>" per probe line
  refused - never bind; the port stays closed so connects are refused
  delay   - bind, accept, sleep delay_ms before replying (probe times out)
  error   - bind, accept, reply "ERR injected" (a well-formed bad reply)

The process prints "READY <port>" on stdout once it is listening (or
"NOLISTEN" in refused mode) so the orchestrator can synchronize.
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import time


def _handle_normal(conn: socket.socket, instance_id: str, counter: list) -> None:
    with conn:
        try:
            data = conn.recv(1024)
            if not data:
                return
            counter[0] += 1
            conn.sendall(f"OK {instance_id} {counter[0]}\n".encode("ascii"))
        except OSError:
            return


def _handle_delay(conn: socket.socket, instance_id: str, delay_ms: int) -> None:
    with conn:
        try:
            data = conn.recv(1024)
            if not data:
                return
            time.sleep(delay_ms / 1000.0)
            conn.sendall(f"OK {instance_id} delayed\n".encode("ascii"))
        except OSError:
            return


def _handle_error(conn: socket.socket) -> None:
    with conn:
        try:
            data = conn.recv(1024)
            if not data:
                return
            conn.sendall(b"ERR injected\n")
        except OSError:
            return


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--behavior", choices=["normal", "refused", "delay", "error"], required=True)
    parser.add_argument("--delay-ms", type=int, default=0)
    args = parser.parse_args(argv)

    if args.behavior == "refused":
        # Never bind: connecting to the port yields ECONNREFUSED for real.
        print("NOLISTEN", flush=True)
        while True:
            time.sleep(3600)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind((args.host, args.port))
    except OSError as exc:
        # Real bind failure (e.g. port already occupied) -> non-zero exit.
        print(f"BINDFAIL {exc}", file=sys.stderr, flush=True)
        return 3
    srv.listen(16)
    print(f"READY {args.port}", flush=True)

    counter = [0]
    while True:
        try:
            conn, _addr = srv.accept()
        except OSError:
            continue
        if args.behavior == "normal":
            target = _handle_normal
            targs = (conn, args.instance_id, counter)
        elif args.behavior == "delay":
            target = _handle_delay
            targs = (conn, args.instance_id, args.delay_ms)
        else:
            target = _handle_error
            targs = (conn,)
        threading.Thread(target=target, args=targs, daemon=True).start()


if __name__ == "__main__":
    sys.exit(main())
