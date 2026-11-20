from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config_io import default_config_path, load_config
from .report import write_report
from .runner import run_experiment
from .samples import generate_samples, save_sample_npz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic GEMM/BMM precision observability experiments."
    )
    parser.add_argument(
        "--config",
        default=str(default_config_path()),
        help="Path to the experiment JSON configuration.",
    )
    parser.add_argument("--output", help="Override the report path from configuration.")
    parser.add_argument(
        "--samples-dir",
        help="Optional directory to materialize deterministic local .npz samples.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run the experiment but do not write JSON/Markdown reports.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.output:
        config = replace(config, report_path=args.output)
    if args.samples_dir:
        samples = generate_samples(config.samples)
        output_dir = Path(args.samples_dir)
        for sample in samples.values():
            save_sample_npz(sample, output_dir / f"{sample.spec.name}.npz")
    report = run_experiment(config, write=False)
    if not args.no_write:
        write_report(report, config.report_path)
    summary = report["summary"]
    print("GEMM/BMM precision observability run complete")
    print(f"  samples:              {summary['sample_count']}")
    print(f"  combinations:         {summary['combination_count']}")
    print(f"  injections hit:       {summary['injections_hit']}/{summary['injection_count']}")
    print(f"  failures:             {summary['failure_count']}")
    print(f"  concurrency equal:    {summary['all_concurrency_checks_equal']}")
    print(f"  layer exceedances:    {summary['layer_exceedance_count']}")
    print(f"  report digest:        {report['report_digest']}")
    if not args.no_write:
        print(f"  JSON report:          {config.report_path}")
        print(f"  Markdown audit:       {Path(config.report_path).with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
