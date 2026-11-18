"""Full (from-scratch) pipeline run: validate -> fingerprint -> gather."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from embedgather.config import load_config
from embedgather.errors import BusinessError, GracefulExit
from embedgather.injection import Injector
from embedgather.pipeline import Pipeline, load_requests, write_report
from embedgather.signals import ExitFlag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--inject", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    injector = Injector.load(args.inject) if args.inject else None
    exit_flag = ExitFlag().install()
    pipeline = Pipeline(config, injector=injector, exit_flag=exit_flag)
    requests = load_requests(config.requests_path)
    report_path = args.report or config.report_path
    try:
        report = pipeline.full_run(requests)
    except GracefulExit as exc:
        print(f"graceful exit, progress={exc.progress}")
        return 0
    except BusinessError as exc:
        print(f"{exc.code}: {exc.detail}")
        return 2
    write_report(report, report_path)
    print(f"full run ok: snapshot_version={report['snapshot_version']} "
          f"report={report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
