"""Independent command-line verifier (E_VERIFY).

Reads bucket files and the query definition, recomputes partition assignment
and joins from source tables, and compares against on-disk results/ledger.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dhj import verify as verify_mod
from dhj.errors import DHJError
from tools.experiment_lib import load_experiment_spec, stats_calculator


def main(argv=None):
    parser = argparse.ArgumentParser(description="Independent DHJ verifier")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--query", default=None)
    parser.add_argument("--bucket-dir", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--partitions", type=int, required=True)
    parser.add_argument("--fraction-bits", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        spec = load_experiment_spec(args.sample, args.query)
        calc = stats_calculator(spec, args.fraction_bits)
        result = verify_mod.verify_all(
            spec,
            args.bucket_dir,
            args.result_dir,
            args.partitions,
            args.fraction_bits,
            calc,
        )
    except DHJError as exc:
        print(json.dumps(exc.to_dict(), indent=2, ensure_ascii=False))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
