"""Command line interface for one incremental/full run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import run_full_rebuild, run_incremental


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deterministic incremental tile segmentation")
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--mode", choices=("incremental", "full"), default="incremental")
    parser.add_argument("--injections", default=None)
    parser.add_argument("--patches-dir", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)

    kwargs = {}
    if args.injections:
        kwargs["injections_path"] = args.injections
    if args.patches_dir:
        kwargs["patches_dir"] = args.patches_dir
    if args.mode == "full":
        _inputs, report = run_full_rebuild(args.sample_dir, args.work_dir, **kwargs)
    else:
        _inputs, report = run_incremental(args.sample_dir, args.work_dir, **kwargs)

    payload = report.to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
