"""Check that a range of high 127.0.0.1 loopback TCP ports is available.

Usage: python scripts/port_check.py 39000 39005
Exit code 0 when every port can be bound, 2 (with details) otherwise.
"""
from __future__ import annotations

import socket
import sys


def check(host: str, port: int) -> tuple[bool, str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    try:
        sock.bind((host, port))
    except OSError as exc:
        return False, str(exc)
    else:
        return True, "free"
    finally:
        sock.close()


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: python scripts/port_check.py <start_port> <end_port>")
        return 64
    start, end = int(argv[1]), int(argv[2])
    host = "127.0.0.1"
    ok_all = True
    for port in range(start, end + 1):
        ok, detail = check(host, port)
        ok_all &= ok
        print(f"{host}:{port} -> {'FREE' if ok else 'BUSY'} ({detail})")
    return 0 if ok_all else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
