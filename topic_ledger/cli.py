from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .app import load_bundle
from .canonical import canonical_json
from .demo import run_demo
from .samplegen import write_sample_bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="topic-ledger", description="Deterministic topic ledger demo")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("generate", help="generate deterministic config and samples")
    sub.add_parser("validate", help="validate config, sample, and command stream")
    demo_parser = sub.add_parser("demo", help="run subprocess, lease, ledger, and replay verification")
    demo_parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)

    if args.command == "generate":
        paths = write_sample_bundle(Path.cwd())
        for path in paths:
            print(path)
        return 0
    if args.command == "validate":
        bundle = load_bundle(Path.cwd())
        print(
            canonical_json(
                {
                    "ok": True,
                    "shards": len(bundle["topology"]["shards"]),
                    "nodes": len(bundle["topology"]["nodes"]),
                    "documents": len(bundle["documents"]),
                    "commands": len(bundle["commands"]),
                }
            )
        )
        return 0
    if args.command == "demo":
        result = run_demo(args.root)
        print(canonical_json(result))
        return 0 if result["overall_equal"] else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
