"""Command line entry point for the multispectral tile workbench."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .injection import load_injections, run_injection_suite
from .loader import load_manifest
from .paths import to_path
from .pipeline import Pipeline
from .report import write_report
from .samplegen import generate_sample


def _grid(manifest_path: Path, root: Path):
    manifest = load_manifest(manifest_path, root)
    rows = max(t.row for t in manifest.tiles) + 1
    cols = max(t.col for t in manifest.tiles) + 1
    return (rows, cols), manifest.tile_edge, manifest.overlap


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mstc", description="multispectral tile classification workbench")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate deterministic local sample")
    gen.add_argument("--root", default=".")
    gen.add_argument("--rows", type=int, default=3)
    gen.add_argument("--cols", type=int, default=3)
    gen.add_argument("--edge", type=int, default=16)
    gen.add_argument("--overlap", type=int, default=4)

    run_p = sub.add_parser("run", help="run the full pipeline")
    run_p.add_argument("--root", default=".")
    run_p.add_argument("--config", required=True)
    run_p.add_argument("--version", type=int, default=1)
    run_p.add_argument("--report", default=None)

    inj = sub.add_parser("inject", help="run fault injection suite")
    inj.add_argument("--root", default=".")
    inj.add_argument("--config", required=True)
    inj.add_argument("--injections", required=True)
    inj.add_argument("--report", required=True)

    args = parser.parse_args(argv)

    if args.command == "generate":
        path, _ = generate_sample(args.root, args.rows, args.cols, args.edge, args.overlap)
        print(f"generated manifest -> {path}")
        return 0

    root = Path(args.root)
    if args.command == "run":
        config = load_config(to_path(args.config, root))
        manifest_native = to_path(config.manifest_path, root)
        grid, edge, overlap = _grid(manifest_native, root)
        pipeline = Pipeline(config, root, global_version=args.version)
        run = pipeline.run(grid, edge, overlap)
        report_path = Path(args.report) if args.report else to_path(config.report_dir, root) / "report.json"
        write_report(run, report_path)
        print(json.dumps({
            "report": str(report_path),
            "recomputed_tiles": run.recomputed_tiles,
            "cached_tiles": run.cached_tiles,
            "failures": run.failures.codes(),
            "boundary_records": len(run.boundary_records),
        }, indent=2))
        return 0

    if args.command == "inject":
        config = load_config(to_path(args.config, root))
        manifest_native = to_path(config.manifest_path, root)
        grid, edge, overlap = _grid(manifest_native, root)
        injections = load_injections(to_path(args.injections, root))
        doc = run_injection_suite(
            root, args.config, config.manifest_path, injections, grid, edge, overlap,
            Path(args.report),
        )
        print(json.dumps(doc["summary"], indent=2))
        return 0 if doc["summary"]["hit"] == doc["summary"]["total"] else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
