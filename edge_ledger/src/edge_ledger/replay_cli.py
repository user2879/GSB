from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import EdgeLedgerError
from .replay import verify_ledger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Independently replay and compare an edge ledger")
    parser.add_argument("--spec-dir", required=True)
    parser.add_argument("--ledger", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = verify_ledger(Path(args.spec_dir), Path(args.ledger))
    except EdgeLedgerError as exc:
        payload = {
            "ok": False,
            "code": exc.code,
            "message": exc.message,
            "field": exc.field,
            "line": exc.line,
            "shard_id": exc.shard_id,
            "basis": exc.basis,
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 3 if exc.code == "E_REPLAY" else 2
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
