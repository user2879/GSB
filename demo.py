"""End-to-end deterministic demo for the GPU scheduling component.

Running ``python demo.py`` executes, inside the local workspace only:

1. a clean deterministic run with ledger and snapshot persistence;
2. the four fault injections and their business codes;
3. a forced termination followed by ledger-based recovery and replay check;
4. a version-drift snapshot replacement;
5. a console rendering of the diagnostic view.
"""
from __future__ import annotations

import os
import shutil

from gpu_scheduler.cli import build_parser


WORKSPACE = os.path.join("workdata", "demo")


def _invoke(argv):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def _make_writable(path: str) -> None:
    # Snapshot backups are deliberately read-only; restore write bits before
    # wiping the local demo workspace so the demo can be rerun repeatedly.
    for root, dirs, files in os.walk(path):
        try:
            os.chmod(root, 0o755)
        except OSError:
            pass
        for name in files:
            try:
                os.chmod(os.path.join(root, name), 0o644)
            except OSError:
                pass


def main() -> int:
    if os.path.isdir(WORKSPACE):
        _make_writable(WORKSPACE)
        shutil.rmtree(WORKSPACE)
    common = ["--workspace", WORKSPACE]

    print("### 1. clean deterministic run")
    _invoke(common + ["run"])

    print("\n### 2. fault injections")
    for kind in ("memory", "quota", "cycle", "ledger"):
        print(f"-- injection: {kind}")
        _invoke(common + ["inject", kind])

    print("\n### 3. forced termination then recovery + replay")
    _invoke(common + ["crash", "--after-round", "3"])
    _invoke(common + ["recover"])

    print("\n### 4. snapshot version drift -> whole-state replacement")
    _invoke(common + ["replace"])

    print("\n### 5. diagnostic console view")
    _invoke(common + ["stats"])
    print("\nDemo finished. Structured log: "
          f"{os.path.join(WORKSPACE, 'logs', 'events.jsonl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
