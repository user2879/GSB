from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .canonical import content_digest


def _emit(payload: dict[str, object]) -> int:
    print(json.dumps(payload, sort_keys=True), flush=True)
    status = payload.get("status")
    if status == "pass":
        return 0
    if status == "corrupt":
        return 20
    return 30


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m blockledger.worker")
    parser.add_argument("--block-file", required=True)
    parser.add_argument("--volume-id", required=True)
    parser.add_argument("--block-index", required=True, type=int)
    parser.add_argument("--baseline-version", required=True, type=int)
    parser.add_argument("--expected-digest", required=True)
    parser.add_argument("--mode", choices=("hash", "slow"), default="hash")
    parser.add_argument("--duration-ms", type=int, default=30_000)
    parser.add_argument("--corrupt-in-memory", action="store_true")
    args = parser.parse_args(argv)

    if args.mode == "slow":
        deadline = time.monotonic() + max(1, args.duration_ms) / 1000.0
        while time.monotonic() < deadline:
            # Busy loop intentionally exercises an unresponsive worker. The
            # parent must terminate the whole process group with SIGKILL.
            pass

    try:
        content = Path(args.block_file).read_bytes()
        if args.corrupt_in_memory and content:
            flipped = content[0] ^ 0xFF
            content = bytes([flipped]) + content[1:]
        actual = content_digest(
            args.volume_id, args.block_index, args.baseline_version, content
        )
    except Exception as exc:  # noqa: BLE001 - child reports any worker failure
        return _emit({"status": "worker_error", "error": type(exc).__name__, "message": str(exc)})

    status = "pass" if actual == args.expected_digest else "corrupt"
    return _emit(
        {
            "status": status,
            "volume_id": args.volume_id,
            "block_index": args.block_index,
            "expected_digest": args.expected_digest,
            "actual_digest": actual,
            "pid": __import__("os").getpid(),
        }
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
