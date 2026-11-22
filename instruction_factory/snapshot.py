from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import SCHEMA_VERSION, STAGES, E_SNAPSHOT_STALE
from .errors import Failure
from .loader import LoadedInputs
from .plan import ConstructionState, state_from_manifest
from .pipeline import PipelineResult
from .util import atomic_write_json, fingerprint_object


@dataclass
class SnapshotCheck:
    snapshot_exists: bool
    stale: bool
    rebuild_from: str | None
    reason: str | None
    manifest: dict[str, Any] | None
    completed_batches: int
    state: ConstructionState | None

    def stale_failure(self) -> Failure | None:
        if not self.stale:
            return None
        return Failure(
            code=E_SNAPSHOT_STALE,
            message=self.reason or "snapshot is stale",
            stage="snapshot",
            field="snapshot",
            fatal=False,
        )


def snapshot_root(output_dir: str | Path) -> Path:
    return Path(output_dir).resolve() / "snapshot"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def check_snapshot(output_dir: str | Path, inputs: LoadedInputs) -> SnapshotCheck:
    root = snapshot_root(output_dir)
    manifest_path = root / "manifest.json"
    if not root.exists():
        return SnapshotCheck(False, False, None, None, None, 0, None)
    if not manifest_path.exists():
        return SnapshotCheck(
            True,
            True,
            "sampling",
            "snapshot directory exists but manifest.json is missing",
            None,
            0,
            None,
        )
    try:
        manifest = _read_json(manifest_path)
    except (OSError, ValueError, TypeError) as exc:
        return SnapshotCheck(True, True, "sampling", f"snapshot manifest cannot be parsed: {exc}", None, 0, None)
    if not isinstance(manifest, dict):
        return SnapshotCheck(True, True, "sampling", "snapshot manifest must be an object", None, 0, None)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        return SnapshotCheck(True, True, "sampling", "snapshot schema version changed", manifest, 0, None)

    comparisons = (
        ("seed_file_hash", inputs.fingerprints["seed_file"], "sampling", "seed file hash changed"),
        ("lexicon_hash", inputs.fingerprints["lexicon"], "sampling", "lexicon/tokenizer hash changed"),
        ("base_config_hash", inputs.fingerprints["config"], "sampling", "base config hash changed"),
        ("plan_hash", inputs.fingerprints["plan"], "sampling", "batch plan hash changed"),
        ("scoring_file_hash", inputs.fingerprints["scoring"], "scoring", "scoring parameter file hash changed"),
    )
    for key, expected, rebuild_from, reason in comparisons:
        if manifest.get(key) != expected:
            return SnapshotCheck(True, True, rebuild_from, f"{reason}; rebuild from {rebuild_from}", manifest, 0, None)

    phase_versions = manifest.get("phase_versions")
    if not isinstance(phase_versions, dict):
        return SnapshotCheck(True, True, "sampling", "snapshot phase versions are missing", manifest, 0, None)
    phase_files = {
        "sampling": root / "sampling" / "artifacts.json",
        "scoring": root / "scoring" / "artifacts.json",
        "dedupe": root / "dedupe" / "artifacts.json",
        "aggregate": root / "aggregate" / "result.json",
    }
    for stage in STAGES:
        path = phase_files[stage]
        if not path.exists():
            return SnapshotCheck(True, True, stage, f"snapshot phase {stage} is missing", manifest, 0, None)
        try:
            payload = _read_json(path)
        except (OSError, ValueError, TypeError) as exc:
            return SnapshotCheck(True, True, stage, f"snapshot phase {stage} cannot be read: {exc}", manifest, 0, None)
        if not isinstance(payload, dict) or payload.get("version") != phase_versions.get(stage):
            return SnapshotCheck(True, True, stage, f"snapshot phase {stage} version mismatch", manifest, 0, None)

    effective_path = root / "aggregate" / "effective_config.json"
    try:
        effective_payload = _read_json(effective_path)
        state = state_from_manifest(effective_payload["state"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return SnapshotCheck(True, True, "aggregate", f"effective config snapshot is invalid: {exc}", manifest, 0, None)
    if manifest.get("effective_config_hash") != state.fingerprint():
        return SnapshotCheck(True, True, "aggregate", "effective config hash does not match the snapshot", manifest, 0, None)
    if manifest.get("effective_scoring_hash") != fingerprint_object(state.scoring):
        return SnapshotCheck(True, True, "aggregate", "effective scoring hash does not match the snapshot", manifest, 0, None)

    completed_batches = manifest.get("completed_batches", 0)
    if not isinstance(completed_batches, int) or isinstance(completed_batches, bool) or completed_batches < 0:
        return SnapshotCheck(True, True, "aggregate", "completed_batches is invalid", manifest, 0, None)
    total_batches = len(inputs.plan["batches"])
    if completed_batches > total_batches:
        return SnapshotCheck(True, True, "sampling", "completed batch count exceeds the current plan", manifest, 0, None)
    expected_global_version = fingerprint_object(
        {
            "schema_version": SCHEMA_VERSION,
            "phase_versions": phase_versions,
            "seed_file_hash": inputs.fingerprints["seed_file"],
            "lexicon_hash": inputs.fingerprints["lexicon"],
            "base_config_hash": inputs.fingerprints["config"],
            "scoring_file_hash": inputs.fingerprints["scoring"],
            "plan_hash": inputs.fingerprints["plan"],
            "effective_config_hash": state.fingerprint(),
            "completed_batches": completed_batches,
        }
    )
    if manifest.get("global_version") != expected_global_version:
        return SnapshotCheck(True, True, "aggregate", "snapshot global version does not match its manifest hashes", manifest, 0, None)
    return SnapshotCheck(True, False, None, None, manifest, completed_batches, state)


def invalidate_snapshot(output_dir: str | Path, from_stage: str) -> None:
    root = snapshot_root(output_dir)
    if not root.exists():
        return
    start = STAGES.index(from_stage)
    for stage in STAGES[start:]:
        target = root / stage
        if target.exists():
            shutil.rmtree(target)
    manifest = root / "manifest.json"
    if manifest.exists():
        manifest.unlink()


def _phase_payload(stage: str, version: str, items: Any) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "stage": stage, "version": version, "items": items}


def _budget_consumed(report: dict[str, Any]) -> dict[str, Any]:
    scheduler = report["scheduler"]
    initial = scheduler["initial_budget"]
    remaining = scheduler["remaining_budget"]
    return {
        "global_score": initial["global_score"] - remaining["global_score"],
        "task_quota": {
            task: initial["task_quota"][task] - remaining["task_quota"].get(task, 0)
            for task in sorted(initial["task_quota"])
        },
        "per_batch_generation_points": initial["per_batch_generation_points"] - remaining["per_batch_generation_points"],
    }


def commit_snapshot(
    *,
    output_dir: str | Path,
    state: ConstructionState,
    inputs: LoadedInputs,
    result: PipelineResult,
    report: dict[str, Any],
    completed_batches: int,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = snapshot_root(output)
    previous_manifest: dict[str, Any] | None = None
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        try:
            previous_manifest = _read_json(manifest_path)
        except (OSError, ValueError):
            previous_manifest = None
    previous_versions = previous_manifest.get("phase_versions", {}) if isinstance(previous_manifest, dict) else {}
    changed_phases = [stage for stage in STAGES if previous_versions.get(stage) != result.phase_versions.get(stage)]
    reused_phases = [stage for stage in STAGES if stage not in changed_phases]
    global_version = fingerprint_object(
        {
            "schema_version": SCHEMA_VERSION,
            "phase_versions": result.phase_versions,
            "seed_file_hash": inputs.fingerprints["seed_file"],
            "lexicon_hash": inputs.fingerprints["lexicon"],
            "base_config_hash": inputs.fingerprints["config"],
            "scoring_file_hash": inputs.fingerprints["scoring"],
            "plan_hash": inputs.fingerprints["plan"],
            "effective_config_hash": state.fingerprint(),
            "completed_batches": completed_batches,
        }
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "global_version": global_version,
        "seed_file_hash": inputs.fingerprints["seed_file"],
        "lexicon_hash": inputs.fingerprints["lexicon"],
        "base_config_hash": inputs.fingerprints["config"],
        "scoring_file_hash": inputs.fingerprints["scoring"],
        "plan_hash": inputs.fingerprints["plan"],
        "effective_config_hash": state.fingerprint(),
        "effective_scoring_hash": fingerprint_object(state.scoring),
        "consumed_budget": _budget_consumed(report),
        "completed_batches": completed_batches,
        "phase_versions": result.phase_versions,
        "changed_phases": changed_phases,
        "reused_phases": reused_phases,
    }

    staging = output / ".snapshot-staging"
    backup = output / ".snapshot-backup"
    if staging.exists():
        shutil.rmtree(staging)
    if backup.exists():
        shutil.rmtree(backup)
    staging.mkdir(parents=True)
    for stage in ("sampling", "scoring", "dedupe"):
        atomic_write_json(
            staging / stage / "artifacts.json",
            _phase_payload(stage, result.phase_versions[stage], result.stage_artifacts[stage]),
        )
    aggregate_payload = {
        **report,
        "snapshot": {
            "global_version": global_version,
            "phase_versions": result.phase_versions,
            "changed_phases": changed_phases,
            "reused_phases": reused_phases,
            "completed_batches": completed_batches,
        },
    }
    atomic_write_json(
        staging / "aggregate" / "result.json",
        _phase_payload("aggregate", result.phase_versions["aggregate"], aggregate_payload),
    )
    atomic_write_json(
        staging / "aggregate" / "effective_config.json",
        {"schema_version": SCHEMA_VERSION, "state": state.to_manifest_dict()},
    )
    atomic_write_json(staging / "manifest.json", manifest)

    if root.exists():
        root.rename(backup)
    staging.rename(root)
    if backup.exists():
        shutil.rmtree(backup)
    return {
        **manifest,
        "snapshot_action": "replaced" if changed_phases or previous_manifest is None else "manifest_refreshed",
    }
