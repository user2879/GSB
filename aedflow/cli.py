"""Command line entry point for generation, pipeline runs and recovery drills."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from .config import load_config
from .generator import generate
from .pipeline import GracefulExit, Pipeline
from .snapshot import SnapshotStore


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AED shard boundary reconstruction")
    parser.add_argument("command", choices=("generate", "run", "drill", "stale-drill"))
    parser.add_argument("--workspace", default=".")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--injection", default="config/injection.yaml")
    args = parser.parse_args(argv)

    workspace = Path(args.workspace).resolve()
    config = load_config(workspace / args.config)
    injection = _load_yaml(workspace / args.injection)

    if args.command == "generate":
        stats = generate(workspace, workspace / args.config)
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return 0

    if args.command == "run":
        pipeline = Pipeline(workspace, config, injection)
        try:
            context = pipeline.run()
        except GracefulExit as exc:
            print(f"graceful exit after stage {exc}; snapshot committed", file=sys.stderr)
            return 0
        report_path = pipeline.write_report()
        summary = {
            "run_id": context["run_id"],
            "report": str(report_path.relative_to(workspace)),
            "trajectories": len(context["trajectories"]),
            "matched": context["evaluation"]["matched_trajectory_count"],
            "failures": len(context["failures"]),
            "exit": context["exit"],
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    if args.command == "drill":
        return _recovery_drill(workspace, config)

    return _stale_drill(workspace, config)


def _recovery_drill(workspace: Path, config: dict[str, Any]) -> int:
    snapshot_dir = workspace / config["paths"]["snapshot_dir"]
    if snapshot_dir.exists():
        import shutil

        shutil.rmtree(snapshot_dir)

    first = Pipeline(workspace, config, {"exit_after_stage": "boundary_decision"})
    try:
        first.run()
    except GracefulExit:
        pass
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        partial_manifest = json.load(handle)
    print("partial snapshot stage:", partial_manifest["stage"])

    resumed = Pipeline(workspace, config, {})
    resumed_context = resumed.run()
    resumed_report = resumed.write_report()

    # A second uninterrupted run must produce a field-identical report.
    import shutil

    shutil.rmtree(snapshot_dir)
    fresh = Pipeline(workspace, config, {})
    fresh_context = fresh.run()
    fresh_report = fresh.write_report()

    same = resumed_report.read_text(encoding="utf-8") == fresh_report.read_text(encoding="utf-8")
    result = {
        "partial_stage": partial_manifest["stage"],
        "resumed_report": str(resumed_report.relative_to(workspace)),
        "fresh_report": str(fresh_report.relative_to(workspace)),
        "field_identical": same,
        "resumed_run_id": resumed_context["run_id"],
        "fresh_run_id": fresh_context["run_id"],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if same else 2


def _stale_drill(workspace: Path, config: dict[str, Any]) -> int:
    import shutil

    from .errors import AedError
    from .loader import load_shards

    snapshot_dir = workspace / config["paths"]["snapshot_dir"]
    shutil.rmtree(snapshot_dir, ignore_errors=True)
    pipeline = Pipeline(workspace, config, {})
    pipeline.run()
    pipeline.write_report()

    manifest_path = workspace / config["paths"]["manifest"]
    shard_dir = workspace / config["paths"]["shard_dir"]
    target = shard_dir / "shard_01.json"
    manifest_backup = manifest_path.with_suffix(".yaml.bak")
    shard_backup = target.with_suffix(".json.bak")
    shutil.copy2(manifest_path, manifest_backup)
    shutil.copy2(target, shard_backup)
    try:
        # Mutate the manifest-declared header AND the package header so the
        # header version set a new run derives genuinely differs. codebook_v3
        # uses fresh code points but the same semantic class names, so the
        # shard stays valid while the snapshot identity changes.
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle)
        for spec in manifest["shards"]:
            if spec["shard_id"] == "shard_01":
                spec["header"]["codebook_version"] = "codebook_v3"
        with manifest_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(manifest, handle, sort_keys=False, allow_unicode=True)

        with target.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        v2 = config["classes"]["codebooks"]["codebook_v2"]
        v3 = config["classes"]["codebooks"]["codebook_v3"]
        name_to_old_col = {
            v2[str(code)]: idx for idx, code in enumerate(payload["class_codes"])
            if str(code) in v2 and v2[str(code)] in ("alarm", "music", "noise", "speech")
        }
        ordered_codes = sorted(v3, key=lambda code: int(code))
        new_columns = [
            name_to_old_col[v3[code]] for code in ordered_codes
        ]
        payload["class_codes"] = ordered_codes
        payload["frame_probabilities"] = [
            [row[col] for col in new_columns] for row in payload["frame_probabilities"]
        ]
        payload["header"]["codebook_version"] = "codebook_v3"
        with target.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

        stale = Pipeline(workspace, config, {})
        loaded_manifest, _, _ = stale._load_inputs()
        loaded = load_shards(shard_dir, loaded_manifest, config)
        identity = {
            "config_fingerprint": stale.fingerprint,
            "header_version_set": loaded["header_version_set"],
            "global_version": 1,
        }
        store = SnapshotStore(workspace / config["paths"]["snapshot_dir"], identity)
        try:
            store.load_manifest()
            observed = None
        except AedError as exc:
            observed = exc.code

        # The pipeline itself must refuse to resume from the stale snapshot.
        pipeline_code = None
        try:
            stale.run()
        except AedError as exc:
            pipeline_code = exc.code
        result = {
            "expected_code": "E_SNAPSHOT_STALE",
            "store_observed_code": observed,
            "pipeline_observed_code": pipeline_code,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        ok = observed == "E_SNAPSHOT_STALE" and pipeline_code == "E_SNAPSHOT_STALE"
        return 0 if ok else 2
    finally:
        manifest_backup.replace(manifest_path)
        shard_backup.replace(target)
