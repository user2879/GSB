"""Test/demo helper that reserves a lease and then dies without completion."""

from __future__ import annotations

import argparse
import os
import time

from .ledger import Ledger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--session-version", type=int, default=1)
    parser.add_argument("--lease-id", default="crash-lease")
    parser.add_argument("--shard-id", default="crash-shard")
    parser.add_argument("--node-id", default="crash-node")
    parser.add_argument("--units", type=int, default=7)
    args = parser.parse_args(argv)
    ledger = Ledger(args.ledger, session_version=args.session_version)
    now = time.monotonic_ns()
    ledger.reserve_lease(
        lease_id=args.lease_id,
        shard_id=args.shard_id,
        node_id=args.node_id,
        units=args.units,
        issued_ns=now,
        deadline_ns=now + 60_000_000_000,
    )
    # Deliberately do not close, release, or write an exit report.
    os._exit(70)


if __name__ == "__main__":
    raise SystemExit(main())

