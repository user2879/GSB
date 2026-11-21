"""Deterministic simulated backend instance.

Run as a subprocess: ``python -m lbhc.backend --port 18601 --behavior ok``.

Behaviours:
  ok        - answer immediately with the expected body
  mismatch  - answer with a body whose digest differs from the expectation
  delay     - hold the response longer than the probe timeout, then answer
  refused   - accepts and immediately RST-closes the connection via
              SO_LINGER(1,0), so the failure is a real connection error
              produced by the socket stack

The same socket serves both probe protocols: if the client sends an HTTP
request first we answer HTTP; otherwise we push a TCP banner immediately.
"""

from __future__ import annotations

import argparse
import os
import socket
import struct
import sys
import threading
import time

PEEK_TIMEOUT_S = 0.05


def _http_response(body: bytes) -> bytes:
    return (b"HTTP/1.0 200 OK\r\nContent-Type: text/plain\r\n"
            b"Content-Length: " + str(len(body)).encode()
            + b"\r\nConnection: close\r\n\r\n" + body)


def _handle(conn: socket.socket, behavior: str, body_ok: bytes,
            body_bad: bytes, delay_s: float):
    try:
        if behavior == "refused":
            conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                            struct.pack("ii", 1, 0))
            conn.close()
            return
        conn.settimeout(PEEK_TIMEOUT_S)
        is_http = False
        try:
            peeked = conn.recv(16, socket.MSG_PEEK)
            is_http = peeked[:3].upper() == b"GET"
        except socket.timeout:
            is_http = False
        if behavior == "delay":
            time.sleep(delay_s)
        body = body_bad if behavior == "mismatch" else body_ok
        if is_http:
            conn.settimeout(2.0)
            try:
                conn.recv(4096)  # drain the request line/headers
            except socket.timeout:
                pass
            conn.sendall(_http_response(body))
        else:
            conn.sendall(body)
    except OSError:
        pass
    finally:
        try:
            conn.close()
        except OSError:
            pass


def serve(port: int, behavior: str, body_ok: str, body_bad: str,
          delay_s: float):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(64)
    while True:
        try:
            conn, _ = srv.accept()
        except OSError:
            return
        threading.Thread(
            target=_handle,
            args=(conn, behavior, body_ok.encode(), body_bad.encode(),
                  delay_s),
            daemon=True,
        ).start()


def _stdin_watchdog():
    """Exit when the parent engine dies (its stdin pipe write-end closes).

    Works after a hard kill (kill -9 / TerminateProcess) on both POSIX and
    Windows, so simulated backends never outlive the workbench.
    """
    try:
        sys.stdin.buffer.read()
    except Exception:
        pass
    os._exit(0)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lbhc.backend")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--behavior",
                    choices=("ok", "mismatch", "delay", "refused"),
                    required=True)
    ap.add_argument("--body-ok", default="OK")
    ap.add_argument("--body-bad", default="BAD")
    ap.add_argument("--delay-s", type=float, default=1.0)
    args = ap.parse_args(argv)
    threading.Thread(target=_stdin_watchdog, daemon=True).start()
    serve(args.port, args.behavior, args.body_ok, args.body_bad,
          args.delay_s)


if __name__ == "__main__":
    main()
