"""Incremental edit-command session runner."""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from embedgather.config import load_config
from embedgather.errors import BusinessError, GracefulExit
from embedgather.injection import Injector
from embedgather.pipeline import Pipeline, load_requests, write_report
from embedgather.signals import ExitFlag


def write_final_state(pipeline, final_dir):
    final_dir = os.path.abspath(final_dir)
    os.makedirs(final_dir, exist_ok=True)
    table_path = os.path.join(final_dir, "table.jsonl")
    with open(table_path, "w", encoding="utf-8") as fh:
        for rec in pipeline.table.to_jsonl_records():
            fh.write(json.dumps(rec) + "\n")
    cfg = pipeline.config.to_runtime_dict()
    cfg["table_path"] = table_path
    cfg["snapshot_dir"] = os.path.join(final_dir, "snapshots")
    cfg["report_path"] = os.path.join(final_dir, "report.json")
    config_path = os.path.join(final_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    return config_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--edits", required=True)
    parser.add_argument("--inject", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--delay", type=float, default=0.0,
                        help="sleep seconds between commands (signal tests)")
    parser.add_argument("--final-state", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    injector = Injector.load(args.inject) if args.inject else None
    exit_flag = ExitFlag().install()
    pipeline = Pipeline(config, injector=injector, exit_flag=exit_flag)
    requests = load_requests(config.requests_path)
    with open(args.edits, "r", encoding="utf-8") as fh:
        commands = json.load(fh)["commands"]
    report_path = args.report or config.report_path

    def pause(_step):
        if args.delay:
            time.sleep(args.delay)

    try:
        report = pipeline.incremental_session(requests, commands,
                                              on_step=pause)
    except GracefulExit as exc:
        print(f"graceful exit, progress={exc.progress}")
        return 0
    except BusinessError as exc:
        print(f"{exc.code}: {exc.detail}")
        return 2
    write_report(report, report_path)
    if args.final_state:
        cfg = write_final_state(pipeline, args.final_state)
        print(f"final state written: {cfg}")
    print(f"incremental run ok: snapshot_version="
          f"{report['snapshot_version']} report={report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
