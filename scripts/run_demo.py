#!/usr/bin/env python3
"""Run the repeatable normal, queue, and fragment demonstration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tunnelvpn.config import load_config  # noqa: E402
from tunnelvpn.demo import run_demo  # noqa: E402
from tunnelvpn.errors import TunnelError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Loopback tunnel repeatable demo")
    parser.add_argument("--config", default="data/demo_routes.json")
    parser.add_argument("--state-dir", default="artifacts/demo-state")
    parser.add_argument("--scenario", choices=["all", "normal", "queue", "fragment"], default="all")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        result = run_demo(config, state_dir=args.state_dir, scenario=args.scenario)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0 if result["code"] == "OK" else 1
    except TunnelError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, sort_keys=True, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
