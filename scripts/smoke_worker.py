#!/usr/bin/env python3
"""Call one running loopback worker and print its returned receipt rows."""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / ".deps"):
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))

from gsb_ledger.config import load_config  # noqa: E402
from gsb_ledger.util import canonical_json  # noqa: E402


def main() -> int:
    project = Path(sys.argv[1] if len(sys.argv) > 1 else "sample").resolve()
    worker_id = sys.argv[2] if len(sys.argv) > 2 else "w-alpha"
    batch_id = sys.argv[3] if len(sys.argv) > 3 else "b-smoke-001"
    config = load_config(project / "config.json")
    worker = config.worker_by_id.get(worker_id)
    if worker is None:
        raise SystemExit(f"unknown worker {worker_id!r}")
    request = {
        "batch_id": batch_id,
        "request_ids": ["r-001", "r-004"],
        "receipt_version": "r3",
        "duration_ms": 6,
        "covered_refs": {},
    }
    with socket.create_connection(("127.0.0.1", worker.port), timeout=5) as connection:
        connection.sendall((canonical_json(request) + "\n").encode("utf-8"))
        buffer = bytearray()
        while b"\n" not in buffer:
            chunk = connection.recv(65536)
            if not chunk:
                break
            buffer.extend(chunk)
    response = json.loads(bytes(buffer).split(b"\n", 1)[0].decode("utf-8"))
    print(json.dumps(response, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if response.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

