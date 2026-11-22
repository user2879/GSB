"""High-level pipeline facade and command line entry point."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

from .config import parse_config
from .engine import run_engine, EngineResult
from .errors import FailureRecord
from .hashing import hash_iterable
from .injection import (
    Injection,
    load_injections,
    apply_file_injections,
    config_overrides,
)
from .loader import load_samples
from .report import build_report
from .snapshot import resume_or_reset


def global_relation_hash(samples) -> str:
    parts = sorted(
        f"{s.sample_id}:{s.relation_hash}" for s in samples
    )
    return hash_iterable(parts, prefix="global-relations")


def _prepare_sample_root(
    sample_root: Path, injections: list[Injection], work_dir: Path
) -> tuple[Path, bool]:
    file_injections = [
        inj
        for inj in injections
        if inj.injection_type
        not in ("nonfinite_fusion_param", "threshold_out_of_range", "crash_after_stage")
    ]
    if not file_injections:
        return sample_root, False
    isolated = work_dir / "injected_samples"
    if isolated.exists():
        shutil.rmtree(isolated)
    shutil.copytree(sample_root, isolated)
    apply_file_injections(injections, isolated)
    return isolated, True


def execute_pipeline(
    sample_root: str | Path,
    config_path: str | Path,
    work_dir: str | Path,
    *,
    injection_path: Optional[str | Path] = None,
    config_overrides_extra: Optional[dict[str, Any]] = None,
    crash_env: Optional[dict[str, str]] = None,
    force_reset: bool = False,
) -> dict[str, Any]:
    sample_root = Path(sample_root)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    injections = load_injections(injection_path) if injection_path else []
    overrides = config_overrides(injections)
    if config_overrides_extra:
        overrides.update(config_overrides_extra)

    # Parameter-level injections must surface as E_CONFIG_INVALID before any
    # snapshot is reused.
    config = parse_config(config_path, overrides=overrides or None)

    prepared_root, isolated = _prepare_sample_root(sample_root, injections, work_dir)

    snapshot, stale_error = resume_or_reset(
        work_dir / "snapshot",
        config_hash=config.fingerprint(),
        relation_hash="pending",
        reset=force_reset,
    )
    start_version = snapshot.version

    failures: list[FailureRecord] = list()
    if stale_error is not None:
        failures.append(
            FailureRecord(
                sample=None,
                region=None,
                stage="load",
                code=stale_error.code,
                message=stale_error.message,
            )
        )

    samples = load_samples(prepared_root, config, failures=failures)
    relation_hash = global_relation_hash(samples)

    # Keep the manifest relation hash truthful even on the first promotion.
    result: EngineResult = run_engine(
        samples,
        config,
        snapshot,
        failures=failures,
        crash_env=crash_env,
        start_version=start_version,
        stale_error=stale_error,
        relation_hash=relation_hash,
    )
    report = build_report(
        result,
        config,
        injections=injections,
        extra={"isolated_sample_root": str(prepared_root) if isolated else None},
    )
    return report


def write_report(report: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".report-", suffix=".json", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(report, fh, sort_keys=True, ensure_ascii=False, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Multimodal evidence pipeline")
    parser.add_argument("--samples", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--injections")
    parser.add_argument("--report", required=True)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args(argv)

    report = execute_pipeline(
        args.samples,
        args.config,
        args.work_dir,
        injection_path=args.injections,
        force_reset=args.reset,
    )
    write_report(report, args.report)
    print(
        json.dumps(
            {
                "report": args.report,
                "final_version": report["meta"]["final_version"],
                "failures": len(report["failures"]),
                "regions": len(report["region_results"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
