"""End-to-end diagnostic service and deterministic run report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .canonical import (
    atomic_write_bytes,
    canonical_hash,
    canonical_json_bytes,
    file_records,
    json_safe,
    pretty_json_bytes,
    sha256_file,
)
from .config import Config, load_config
from .consistency import assess_consistency
from .evaluation import evaluate_tier
from .projection import project_tier
from .render import render_committed_snapshot
from .snapshot import SnapshotStore
from .strategy import select_and_fit
from .synth import generate_tier


SERVICE_VERSION = "spatial-diagnostic-service/1.0.0"


def _direction_error_lines(tiers: list[dict[str, Any]]) -> bytes:
    lines: list[bytes] = []
    for tier in tiers:
        name = tier["decision"]["tier"]
        fields = tier["fields"]
        for index, (azimuth, elevation, reference, reconstructed, relative, low, over) in enumerate(
            zip(
                fields["azimuth_deg"],
                fields["elevation_deg"],
                fields["reference_normalized"],
                fields["reconstructed_normalized"],
                fields["relative_error"],
                fields["low_confidence_grid"],
                fields["over_tolerance"],
            )
        ):
            record = {
                "tier": name,
                "direction_index": index,
                "azimuth_deg": azimuth,
                "elevation_deg": elevation,
                "reference_energy": reference,
                "reconstructed_energy": reconstructed,
                "relative_error": relative,
                "low_confidence": bool(low),
                "over_tolerance": bool(over),
            }
            lines.append(canonical_json_bytes(record))
    return b"\n".join(lines) + b"\n"


def _write_jsonl(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, payload)


def run_workbench(
    workspace: str | Path,
    config_path: str | Path | None = None,
    *,
    render_view: bool = True,
) -> dict[str, Any]:
    config: Config = load_config(config_path)
    store = SnapshotStore(Path(workspace))
    recovery = store.recover()
    staging = store.start_staging()
    samples_root = staging / "samples"
    reports_root = staging / "reports"
    steps: dict[str, Any] = {"config_sha256": config.config_hash}

    tier_results: list[dict[str, Any]] = []
    source_artifacts: list[Path] = []
    for tier in config.tiers:
        tier_name = str(tier["name"])
        tier_dir = samples_root / tier_name
        generated = generate_tier(config, tier, tier_dir)
        manifest = generated["manifest"]
        manifest_path = Path(generated["manifest_path"])
        manifest_hash = sha256_file(manifest_path)
        if manifest_hash != sha256_file(tier_dir / "manifest.json"):
            raise RuntimeError("manifest hash changed during generation")
        projection = project_tier(config, tier, manifest, tier_dir)
        coefficients, decision = select_and_fit(config, tier, projection)
        evaluated = evaluate_tier(config, tier, manifest, projection, coefficients, decision)
        coefficient_list = json_safe(np.asarray(coefficients, dtype=np.float64).tolist())
        tier_payload = {
            "schema": "spatial-diagnostic-tier/1",
            "tier": tier_name,
            "source_positions": int(tier["source_positions"]),
            "sphere_points": int(tier["sphere_points"]),
            "manifest": {
                "path": manifest_path.relative_to(staging).as_posix(),
                "sha256": manifest_hash,
                "content_hash": manifest["content_hash"],
                "path_count": int(manifest["path_count"]),
            },
            "projection": projection,
            "coefficients": coefficient_list,
            "coefficient_hash": canonical_hash(coefficient_list),
            "decision": decision,
            "validation": evaluated["validation"],
            "fields": evaluated["fields"],
        }
        tier_results.append(tier_payload)
        source_artifacts.extend(generated["sample_paths"])
        source_artifacts.append(manifest_path)
        steps[f"generator_{tier_name}"] = generated["step_hash"]
        steps[f"wav_set_{tier_name}"] = generated["wav_hash_set"]
        steps[f"projection_{tier_name}"] = projection["content_hash"]
        steps[f"strategy_{tier_name}"] = decision["decision_hash"]
        steps[f"validation_{tier_name}"] = evaluated["validation"]["content_hash"]
        steps[f"coefficients_{tier_name}"] = tier_payload["coefficient_hash"]

    consistency = assess_consistency(config, tier_results)
    steps["consistency"] = consistency["content_hash"]

    direction_error_path = reports_root / "direction-errors.jsonl"
    _write_jsonl(direction_error_path, _direction_error_lines(tier_results))
    steps["direction_error_manifest"] = sha256_file(direction_error_path)
    step_hash_path = reports_root / "step-hashes.json"
    atomic_write_bytes(step_hash_path, pretty_json_bytes(steps))
    steps["step_hash_manifest"] = sha256_file(step_hash_path)

    all_artifacts = sorted(
        [path for path in staging.rglob("*") if path.is_file()],
        key=lambda path: path.relative_to(staging).as_posix(),
    )
    artifact_records = file_records(all_artifacts, staging)
    steps["artifact_set"] = canonical_hash(artifact_records)

    payload: dict[str, Any] = {
        "schema": config.data["snapshot"]["schema"],
        "service_version": SERVICE_VERSION,
        "config": config.public,
        "config_sha256": config.config_hash,
        "tiers": {tier["tier"]: tier for tier in tier_results},
        "consistency": consistency,
        "steps": steps,
        "artifacts": artifact_records,
        "runtime_policy": {
            "deterministic_seed": config.seed,
            "timestamps_excluded": True,
            "snapshot_identity": "SHA-256(content payload) -> version prefix + 16 hex characters",
            "recovery": "CURRENT.json is committed last; staging and non-current runs are recycled",
        },
    }
    complete_snapshot = store.commit(staging, payload, str(config.data["snapshot"]["version_prefix"]))
    view_result = None
    if render_view:
        view_result = render_committed_snapshot(store)

    report: dict[str, Any] = {
        "schema": "spatial-diagnostic-run-report/1",
        "version": complete_snapshot["version"],
        "content_sha256": complete_snapshot["integrity"]["content_sha256"],
        "config_sha256": config.config_hash,
        "tiers": {
            tier["tier"]: {
                "sources": tier["source_positions"],
                "sphere_points": tier["sphere_points"],
                "strategy": tier["decision"]["selected_strategy"],
                "order": tier["decision"]["selected_order"],
                "peak_memory_bytes": tier["decision"]["peak_memory_bytes"],
                "coefficient_hash": tier["coefficient_hash"],
                "error_interval": tier["validation"]["selected"]["error_interval"],
                "over_tolerance_direction_ratio": tier["validation"]["selected"][
                    "over_tolerance_direction_ratio"
                ],
                "worst_direction": tier["validation"]["selected"]["worst_direction"],
            }
            for tier in tier_results
        },
        "consistency": {
            "consistent": consistency["consistent"],
            "pairs": [
                {
                    "left_tier": pair["left_tier"],
                    "right_tier": pair["right_tier"],
                    "consistent": pair["consistent"],
                    "divergence_ratio": pair["divergence_ratio"],
                    "first_direction_divergence": pair["first_direction_divergence"],
                }
                for pair in consistency["pairs"]
            ],
        },
        "steps": steps,
        "artifacts": artifact_records,
        "view": {
            "version": view_result["version"],
            "files": view_result["files"],
        }
        if view_result is not None
        else None,
    }
    store.write_latest_report(report)
    report["paths"] = {
        "workspace": str(store.root),
        "snapshot": str(store.run_dir(complete_snapshot) / "snapshot.json"),
        "views": str(store.views / complete_snapshot["version"])
        if view_result is not None
        else None,
        "direction_errors": str(store.run_dir(complete_snapshot) / "reports" / "direction-errors.jsonl"),
        "latest_report": str(store.latest_report_path),
    }
    report["recovered_orphan_count"] = len(recovery["recovered_orphans"])
    return json_safe(report)

