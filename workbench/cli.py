"""Command line interface for the workbench."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import samplegen
from .edits import apply_edit
from .pipeline import Pipeline
from .verify import fusion_artifacts_equal, reports_equal


def main(argv=None):
    parser = argparse.ArgumentParser(prog="workbench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("gen-samples")
    p.add_argument("--out", default="samples")

    p = sub.add_parser("run")
    p.add_argument("--samples", default="samples")
    p.add_argument("--config", default=None)
    p.add_argument("--work", default="work")
    p.add_argument("--inject", default=None)
    p.add_argument("--full", action="store_true")
    p.add_argument("--report", default=None)
    p.add_argument("--stop-after-tiles", type=int, default=None)
    p.add_argument("--stop-after-fusions", type=int, default=None)

    p = sub.add_parser("run-edits",
                       help="apply edit stream with incremental runs, then "
                            "verify against a from-scratch full rebuild")
    p.add_argument("--samples", default="samples")
    p.add_argument("--config", default=None)
    p.add_argument("--work", default="work")
    p.add_argument("--edits", default=None)

    args = parser.parse_args(argv)

    if args.cmd == "gen-samples":
        samplegen.generate_all(args.out)
        print(f"samples generated at {args.out}")
        return 0

    if args.cmd == "run":
        stop_after = {}
        if args.stop_after_tiles is not None:
            stop_after["tiles"] = args.stop_after_tiles
        if args.stop_after_fusions is not None:
            stop_after["fusions"] = args.stop_after_fusions
        pipe = Pipeline(args.samples, args.work, config_path=args.config,
                        inject_path=args.inject, full=args.full,
                        stop_after=stop_after)
        report = pipe.run()
        if args.report:
            Path(args.report).write_text(json.dumps(report, indent=2))
        print(json.dumps({"global_version": report["global_version"],
                          "failed": len(report["failed"]),
                          "pending": report["pending_tiles"],
                          "retained": report["retained_fusions"]}, indent=2))
        return 0

    if args.cmd == "run-edits":
        samples = Path(args.samples)
        config = args.config or str(samples / "config.json")
        edits = json.loads(Path(args.edits or samples / "edits.json").read_text())
        work = Path(args.work)
        if work.exists():
            shutil.rmtree(work)
        for idx, edit in enumerate(edits):
            apply_edit(samples, config, edit)
            report = Pipeline(samples, work / "inc", config_path=config).run()
            print(f"edit {idx} ({edit['op']}): global_version="
                  f"{report['global_version'][:12]} failed={len(report['failed'])}")
        full_report = Pipeline(samples, work / "full", config_path=config,
                               full=True).run()
        ok, diffs = reports_equal(report, full_report)
        ok2, diffs2 = fusion_artifacts_equal(work / "inc", work / "full")
        for d in diffs + diffs2:
            print("DIFF:", d)
        print("INCREMENTAL==FULL:", ok and ok2)
        return 0 if (ok and ok2) else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
