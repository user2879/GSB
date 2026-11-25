#!/usr/bin/env python
"""Standalone loopback port probe used before worker tests/demos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from speaker_diag.canonical import canonical_json
from speaker_diag.errors import DiagnosticError
from speaker_diag.orchestration import probe_workload_ports
from speaker_diag.spec import load_workload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "samples" / "workload.json"))
    args = parser.parse_args()
    try:
        result = probe_workload_ports(load_workload(args.config))
    except DiagnosticError as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

