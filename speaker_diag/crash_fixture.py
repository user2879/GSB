"""Commit a prefix of batches and die without graceful cleanup.

Used by the recovery demonstration/test to prove that only committed rows are
trusted after an abrupt process exit.
"""

from __future__ import annotations

import argparse
import json
import os

from .ledger import Ledger
from .pipeline import normalize_local_results
from .spec import load_workload
from .synthetic import synthesize


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Crash after committing a batch prefix")
    parser.add_argument("--db", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--commit-count", type=int, required=True)
    parser.add_argument("--exit-code", type=int, default=37)
    args = parser.parse_args(argv)
    spec = load_workload(args.spec)
    bundle = synthesize(spec)
    ledger = Ledger(
        args.db,
        session_version=spec.session_version,
        spec_digest=spec.spec_digest,
        session_snapshot=spec.to_dict(),
        reset=True,
    )
    _, normalized = normalize_local_results(spec, bundle)
    truth_by_window = {window.window_id: window for window in bundle.windows}
    committed = []
    for request in spec.requests[: args.commit_count]:
        result = ledger.commit_batch(
            normalized[request.batch_id],
            truth_by_window,
            ordinal=request.request_index,
        )
        committed.append(result["batch_id"])
    print(json.dumps({"committed": committed, "exit_code": args.exit_code}), flush=True)
    os._exit(args.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
