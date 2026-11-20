"""Find a contiguous high loopback port range for demos or tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edgecache.cli import find_free_port_base  # noqa: E402
from edgecache.errors import EdgeError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--start", type=int, default=44000)
    args = parser.parse_args()
    try:
        base = find_free_port_base(args.count, args.start)
    except EdgeError as exc:
        print(json.dumps(exc.envelope(), ensure_ascii=False, sort_keys=True, indent=2))
        return 2
    print(json.dumps({"ok": True, "port_base": base, "ports": list(range(base, base + args.count))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
