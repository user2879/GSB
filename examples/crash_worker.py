"""Open an uncommitted batch and terminate without a normal cleanup path."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from srebudget.loader import load_chain, load_trace
from srebudget.storage import SessionStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--chain", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()

    chain = load_chain(args.chain)
    trace = load_trace(args.trace, chain)
    store = SessionStore(args.db)
    store.start_batch_attempt(
        args.session_id,
        start_call=args.start,
        end_call_exclusive=args.end,
    )
    # The batch attempt is committed, but no conclusion/state/cursor batch is.
    os._exit(86)


if __name__ == "__main__":
    raise SystemExit(main())
