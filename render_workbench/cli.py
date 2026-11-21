"""Command line entry point for the deterministic rendering workbench."""

import argparse
import json
import os
import sys

# Single-threaded BLAS keeps serial and thread-pool runs bit-identical.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

from .runner import run_workflow  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser(
        prog="render-workbench",
        description="Local deterministic text-to-image / img2img workbench")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--config", required=True, help="render config YAML")
        sp.add_argument("--prompts", required=True, help="prompt JSONL")
        sp.add_argument("--plan", help="batch edit plan JSON")
        sp.add_argument("--workdir", required=True,
                        help="persistent artifact/work directory")

    sp = sub.add_parser("render", help="incremental render over edit batches")
    add_common(sp)

    sp = sub.add_parser("full",
                        help="clean-slate one-shot full render of final inputs")
    add_common(sp)

    sp = sub.add_parser("baseline",
                        help="forced single-thread serial execution baseline")
    add_common(sp)

    sp = sub.add_parser("inspect", help="print a workdir manifest summary")
    sp.add_argument("--workdir", required=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "inspect":
        manifest_path = os.path.join(args.workdir, "manifest.json")
        if not os.path.exists(manifest_path):
            print(json.dumps({"error": "no manifest in workdir"}, indent=2))
            return 1
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        summary = {
            "version": manifest.get("version"),
            "config_fp": manifest.get("config_fp"),
            "node_count": len(manifest.get("nodes", {})),
            "cursors": manifest.get("cursors", {}),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    mode = {"render": "incremental", "full": "full",
            "baseline": "baseline"}[args.command]
    report, exit_code = run_workflow(
        args.config, args.prompts, args.plan, args.workdir, mode=mode)
    print(json.dumps({
        "status": report["status"],
        "mode": report["mode"],
        "total_recomputed_nodes": report["total_recomputed_nodes"],
        "resume_version": report["resume_version"],
        "failures": report["failures"],
        "report": os.path.join(os.path.abspath(args.workdir), "report.json"),
    }, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
