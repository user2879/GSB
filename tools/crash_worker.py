"""Subprocess worker that self-terminates after a fixed number of bucket ops.

Used to demonstrate real forced-termination recovery (SIGKILL semantics on
Linux/macOS; hard ``os._exit`` on Windows where SIGKILL does not exist).
"""

import argparse
import os
import signal
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dhj import engine
from tools.experiment_lib import load_experiment_spec


def _hard_exit():
    if hasattr(signal, "SIGKILL"):
        os.kill(os.getpid(), signal.SIGKILL)
    os._exit(137)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--query", default=None)
    parser.add_argument("--bucket-dir", required=True)
    parser.add_argument("--partitions", type=int, required=True)
    parser.add_argument("--fraction-bits", type=int, required=True)
    parser.add_argument("--kill-after", type=int, default=0)
    args = parser.parse_args(argv)

    spec = load_experiment_spec(args.sample, args.query)
    calls = {"n": 0}

    def callback(side, partition):
        calls["n"] += 1
        if args.kill_after and calls["n"] >= args.kill_after:
            _hard_exit()

    engine.generate_version(
        args.bucket_dir,
        spec,
        3,
        args.partitions,
        args.fraction_bits,
        progress_callback=callback,
    )
    print("worker completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
