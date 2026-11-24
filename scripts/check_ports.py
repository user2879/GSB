"""Pre-flight loopback port check (requirement: ports must be free and distinct)."""

from __future__ import annotations

import argparse
import json
import sys

from batch_diagnostic.errors import DiagnosticError
from batch_diagnostic.network import check_port_bindable
from batch_diagnostic.spec import load_spec


def main() -> int:
    parser = argparse.ArgumentParser(description="Check loopback ports are bindable")
    parser.add_argument("--requests", required=True)
    parser.add_argument("--workers", required=True)
    parser.add_argument("--budget", required=True)
    args = parser.parse_args()
    spec = load_spec(args.requests, args.workers, args.budget)
    seen: set[int] = set()
    report = []
    for worker in spec.workers:
        if worker.port in seen:
            print(json.dumps({"code": "E_SPEC", "message": "duplicate port", "port": worker.port}))
            return 1
        check_port_bindable(worker.host, worker.port)
        seen.add(worker.port)
        report.append({"worker_id": worker.worker_id, "endpoint": f"{worker.host}:{worker.port}"})
    print(json.dumps({"code": "OK", "bindable": report}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except DiagnosticError as exc:
        print(json.dumps(exc.to_dict(), indent=2, ensure_ascii=False))
        sys.exit(1)
