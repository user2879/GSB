"""Standalone loopback-only mock device process."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from pathlib import Path

from .transport import LoopbackServer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Loopback haptic device simulator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--port-file", required=True)
    parser.add_argument("--drop-seq", type=int, action="append", default=[])
    parser.add_argument("--downgrade-protocol", type=int, choices=(1, 2), default=None)
    parser.add_argument("--delay-ms", type=float, default=0.0)
    args = parser.parse_args(argv)

    server = LoopbackServer(
        host=args.host,
        port=args.port,
        drop_sequences=args.drop_seq,
        downgrade_protocol=args.downgrade_protocol,
        delay_ms=args.delay_ms,
    )
    port_file = Path(args.port_file)
    port_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = port_file.with_suffix(port_file.suffix + ".tmp")
    temporary.write_text(f"{server.host}:{server.port}\n", encoding="ascii")
    os.replace(temporary, port_file)

    stopping = False

    def stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True
        threading.Thread(target=server.stop, name="device-server-stop", daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop)
    try:
        server.server.serve_forever()
    finally:
        if not stopping:
            server.server.server_close()
        try:
            port_file.unlink()
        except FileNotFoundError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
