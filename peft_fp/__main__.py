"""CLI: python -m peft_fp <command> ...

Commands:
  run --samples DIR --store DIR [--report PATH]   run one pipeline pass
  upgrade --samples DIR --step N                  apply upgrade plan step N
  gen-samples DIR                                 generate the sample directory
  gen-faults BASE_DIR OUT_DIR                     generate fault scenarios
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .pipeline import apply_upgrade_step, run_pipeline
from .store import canonical_json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="peft_fp")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--samples", required=True)
    p_run.add_argument("--store", required=True)
    p_run.add_argument("--report", default=None)

    p_up = sub.add_parser("upgrade")
    p_up.add_argument("--samples", required=True)
    p_up.add_argument("--step", type=int, required=True)

    p_gen = sub.add_parser("gen-samples")
    p_gen.add_argument("out_dir")

    p_fault = sub.add_parser("gen-faults")
    p_fault.add_argument("base_dir")
    p_fault.add_argument("out_dir")

    args = parser.parse_args(argv)
    if args.cmd == "run":
        report = run_pipeline(args.samples, args.store, args.report)
        print(canonical_json(report))
        return 0 if not report["failures"] else 1
    if args.cmd == "upgrade":
        applied = apply_upgrade_step(args.samples, args.step)
        print(json.dumps({"applied": applied}))
        return 0
    if args.cmd == "gen-samples":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        import gen_samples

        import unittest.mock as mock

        with mock.patch("sys.argv", ["gen_samples.py", args.out_dir]):
            gen_samples.main()
        return 0
    if args.cmd == "gen-faults":
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        import unittest.mock as mock

        import gen_faults

        with mock.patch("sys.argv", ["gen_faults.py", args.base_dir, args.out_dir]):
            gen_faults.main()
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
