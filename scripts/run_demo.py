"""End-to-end, fully local, offline diagnostic runner.

Examples
--------
# cold full rebuild, serial
python scripts/run_demo.py --samples-dir samples --mode serial

# thread-pool execution (worker count comes from the config)
python scripts/run_demo.py --samples-dir samples --mode parallel

# incremental run reusing the previous snapshot
python scripts/run_demo.py --samples-dir samples --mode serial
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pe_diag.common import canonical_json  # noqa: E402
from pe_diag.config import (  # noqa: E402
    Config,
    load_injections,
    load_json,
    load_samples,
)
from pe_diag.pipeline import DiagnosticPipeline  # noqa: E402


def run(samples_dir: str, mode: str, report_name: str | None = None,
        snapshot_subdir: str | None = None,
        limit_per_tier: int | None = None) -> dict:
    samples_dir = Path(samples_dir)
    config = Config.load(samples_dir / "position_encoding_config.json")
    samples = load_samples(samples_dir / "attention_samples.jsonl")
    if limit_per_tier is not None:
        counts: dict = {}
        limited = []
        for smp in samples:
            used = counts.get(smp.tier_id, 0)
            if used < limit_per_tier:
                limited.append(smp)
                counts[smp.tier_id] = used + 1
        samples = limited
    scale_doc = load_json(samples_dir / "scale_descriptor.json")
    injections = load_injections(samples_dir / "injection_manifest.json")

    snap_root = Path(config.paths.get("snapshot_dir", "snapshots"))
    if snapshot_subdir:
        snap_root = snap_root / snapshot_subdir
    report_dir = Path(config.paths.get("report_dir", "reports"))
    report_dir.mkdir(parents=True, exist_ok=True)

    pipeline = DiagnosticPipeline(
        config=config, samples=samples, injections=injections,
        scale_doc=scale_doc, snapshot_root=str(snap_root))
    parallel = mode == "parallel"
    report = pipeline.run(parallel=parallel)

    name = report_name or "report_{}.json".format(mode)
    out_path = report_dir / name
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(canonical_json(report))
    print("wrote {} (snapshot status: {}, failures: {}, codes: {})".format(
        out_path, report["snapshot"]["status"],
        len(report["failures"]), report["failure_codes"]))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", default="samples")
    parser.add_argument("--mode", choices=["serial", "parallel"],
                        default="serial")
    parser.add_argument("--report-name", default=None)
    parser.add_argument("--snapshot-subdir", default=None)
    parser.add_argument("--limit-per-tier", type=int, default=None)
    args = parser.parse_args()
    run(args.samples_dir, args.mode, args.report_name,
        args.snapshot_subdir, args.limit_per_tier)


if __name__ == "__main__":
    main()
