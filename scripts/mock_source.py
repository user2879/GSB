#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leaseaudit.mock_service import serve_source  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic loopback load source")
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--mode", choices=("success", "fixed_delay", "refuse"), required=True)
    parser.add_argument("--processing-ms", type=int, default=0)
    args = parser.parse_args()
    if args.port < 1024 or args.port > 65535:
        parser.error("port must be in the high-port range 1024..65535")
    print(
        f"READY source_id={args.source_id} address=127.0.0.1 port={args.port} mode={args.mode}",
        flush=True,
    )
    serve_source(args.source_id, args.port, args.mode, args.processing_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
