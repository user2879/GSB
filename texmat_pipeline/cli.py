"""Command line entry points for the texture/material pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .loader import scan_textures
from .models import load_generation_config, load_materials
from .pipeline import TexturePipeline


def _load_inputs(input_dir: Path):
    config, cfg_failures = load_generation_config(input_dir / "generation_config.json")
    materials, mat_failures = load_materials(input_dir / "materials.json")
    failures = cfg_failures + mat_failures
    if config is None or not materials:
        return None, None, failures
    return config, materials, failures


def cmd_run(args) -> int:
    input_dir = Path(args.input)
    work_dir = Path(args.work)
    config, materials, load_failures = _load_inputs(input_dir)
    if config is None:
        print(json.dumps({"failures": [f.to_dict() for f in load_failures]}, indent=2))
        return 2
    pipeline = TexturePipeline(input_dir, work_dir, config, materials)
    state = pipeline.snapshots.inspect(config.config_hash, config.fixed_point.frac_bits)
    resume_note = {"resumed": not state.stale and state.global_version > 0,
                   "global_version": state.global_version, "stale": state.stale}
    if state.stale and state.failure is not None:
        load_failures.append(state.failure)
    resolution = args.resolution or config.resolution_tiers[-1]
    result = pipeline.run(resolution, use_threads=not args.serial, force_rebuild=args.fresh)
    summary = {
        "resolution": result.resolution,
        "config_hash": result.config_hash,
        "resume": resume_note,
        "recompute_keys": len(result.recompute_keys),
        "reuse_keys": len(result.reuse_keys),
        "affected_scope": {k: len(v) for k, v in result.affected_scope.items()},
        "scheduler": result.scheduler,
        "failures": result.failure_dicts() + [f.to_dict() for f in load_failures],
        "report_path": str(Path(args.work) / "report.json"),
        "snapshot": result.snapshots[0],
    }
    print(json.dumps(summary, indent=2))
    return 0


def cmd_inspect(args) -> int:
    input_dir = Path(args.input)
    config, materials, failures = _load_inputs(input_dir)
    if config is None:
        print(json.dumps({"failures": [f.to_dict() for f in failures]}, indent=2))
        return 2
    out = {
        "config_hash": config.config_hash,
        "resolution_tiers": list(config.resolution_tiers),
        "layer_count": len(config.layers),
        "blend_mode": config.blend_mode,
        "fixed_point": config.fixed_point.__dict__,
        "materials": [m.material_id for m in materials],
        "failures": [f.to_dict() for f in failures],
    }
    print(json.dumps(out, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic fixed-point texture/material pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run (or incrementally resume) the pipeline")
    run_p.add_argument("--input", required=True)
    run_p.add_argument("--work", required=True)
    run_p.add_argument("--resolution", type=int, default=None)
    run_p.add_argument("--serial", action="store_true", help="force single-threaded scheduling")
    run_p.add_argument("--fresh", action="store_true", help="ignore cache and fully rebuild")
    run_p.set_defaults(func=cmd_run)

    inspect_p = sub.add_parser("inspect", help="validate and print parsed inputs")
    inspect_p.add_argument("--input", required=True)
    inspect_p.set_defaults(func=cmd_inspect)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
