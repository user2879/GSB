"""Helper used by signal tests: send SIGTERM to ourselves after a short delay."""
import argparse
import os
import signal
import sys
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mvrec.cli import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    def send_signal():
        time.sleep(0.25)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=send_signal, daemon=True).start()
    _, pipeline = run_pipeline(
        args.sample, args.config, None, args.snapshot, args.report
    )
    return 0 if pipeline.shutdown_requested.is_set() else 0


if __name__ == "__main__":
    raise SystemExit(main())
