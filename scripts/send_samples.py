from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gray_merge_lab.gateway import GatewayClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send an NDJSON sample epoch to the local JSONL gateway")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("samples", help="NDJSON sample file")
    args = parser.parse_args(argv)
    path = Path(args.samples)
    accepted = 0
    try:
        with GatewayClient(args.host, args.port) as client:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    sample = json.loads(line)
                    response = client.accept(sample)
                    if not response.get("ok"):
                        print(json.dumps(response, ensure_ascii=False), file=sys.stderr)
                        return 2
                    accepted += 1
            summary = client.flush()
        print(json.dumps({"accepted": accepted, **summary}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"E_RUNTIME: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
