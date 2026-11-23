"""Generation-based, atomically published stage snapshots."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .constants import E_SNAPSHOT_INVALID, SCHEMA_VERSION, STAGE_ORDER
from .loader import config_fingerprint, corpus_fingerprint, vocabulary_fingerprint
from .models import (
    CorpusSample,
    Failure,
    InteractionConfig,
    ModelParams,
    StageArtifact,
    VocabEntry,
)
from .util import (
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    digest,
    parse_json_text,
)


def _encode_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return {"$decimal": str(value)}
    if isinstance(value, dict):
        return {str(key): _encode_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode_value(item) for item in value]
    return value


def _decode_value(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"$decimal"}:
            return Decimal(str(value["$decimal"]))
        if set(value) == {"__decimal__"}:
            return Decimal(str(value["__decimal__"]))
        if set(value) == {"__float__"}:
            return float(value["__float__"])
        if set(value) == {"__fraction__"}:
            from fractions import Fraction

            numerator, denominator = value["__fraction__"]
            return Fraction(int(numerator), int(denominator))
        return {key: _decode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    return value



_TRANSIENT_KEYS = {
    "elapsed_ms",
    "response_elapsed_ms",
    "settle_elapsed_ms",
}


def _sanitize_transient(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize_transient(item)
            for key, item in value.items()
            if key not in _TRANSIENT_KEYS
        }
    if isinstance(value, list):
        return [_sanitize_transient(item) for item in value]
    return value

def _model_state(model: ModelParams) -> dict[str, Any]:
    return {
        "version": model.version,
        "feature_dim": model.feature_dim,
        "hash_buckets": model.hash_buckets,
        "labels": list(model.labels),
        "char_embeddings": {
            key: list(vector) for key, vector in model.char_embeddings.items()
        },
        "emission_weights": model.emission_weights,
        "transition_weights": model.transition_weights,
        "start_weights": model.start_weights,
        "default_entity_type": model.default_entity_type,
        "vocabulary": {
            key: entry.as_dict() for key, entry in model.vocabulary.items()
        },
    }


def _config_state(config: InteractionConfig) -> dict[str, Any]:
    return {
        "latency_budget_ms": config.latency_budget_ms,
        "fixed_decimals": config.fixed_decimals,
        "quantization_level": config.quantization_level,
        "quantization_levels": list(config.quantization_levels),
        "thread_pool_workers": config.thread_pool_workers,
        "snapshot_commit_interval_commands": config.snapshot_commit_interval_commands,
        "boundary_score_tolerance": config.boundary_score_tolerance,
        "temp_dir": config.temp_dir,
        "default_entity_type": config.default_entity_type,
    }


def _restore_model(value: dict[str, Any]) -> ModelParams:
    vocabulary = {
        entry_id: VocabEntry(
            entry_id=str(entry["id"]),
            text=str(entry["text"]),
            type_code=str(entry["type"]),
        )
        for entry_id, entry in value.get("vocabulary", {}).items()
    }
    return ModelParams(
        version=str(value["version"]),
        feature_dim=int(value["feature_dim"]),
        hash_buckets=int(value["hash_buckets"]),
        labels=tuple(value["labels"]),
        char_embeddings={
            key: [Decimal(str(item)) for item in vector]
            for key, vector in value["char_embeddings"].items()
        },
        emission_weights=[
            [Decimal(str(item)) for item in row] for row in value["emission_weights"]
        ],
        transition_weights=[
            [Decimal(str(item)) for item in row] for row in value["transition_weights"]
        ],
        start_weights=[Decimal(str(item)) for item in value["start_weights"]],
        default_entity_type=str(value.get("default_entity_type", "ORG")),
        vocabulary=vocabulary,
    )


def _restore_config(value: dict[str, Any]) -> InteractionConfig:
    return InteractionConfig(
        latency_budget_ms=float(value["latency_budget_ms"]),
        fixed_decimals=int(value["fixed_decimals"]),
        quantization_level=int(value["quantization_level"]),
        quantization_levels=tuple(int(item) for item in value["quantization_levels"]),
        thread_pool_workers=int(value["thread_pool_workers"]),
        snapshot_commit_interval_commands=int(
            value["snapshot_commit_interval_commands"]
        ),
        boundary_score_tolerance=Decimal(str(value["boundary_score_tolerance"])),
        temp_dir=str(value.get("temp_dir", "artifacts/tmp")),
        default_entity_type=str(value.get("default_entity_type", "ORG")),
    )


def _failure_state(failure: Failure) -> dict[str, Any]:
    return failure.as_dict()


def _restore_failure(value: dict[str, Any]) -> Failure:
    return Failure(
        code=str(value["code"]),
        stage=str(value["stage"]),
        message=str(value["message"]),
        sample_id=value.get("sample_id"),
        command_position=value.get("command_position"),
        locator=value.get("locator"),
        details=dict(value.get("details", {})),
    )


@dataclass
class RestoredSnapshot:
    samples: dict[str, CorpusSample]
    model: ModelParams
    config: InteractionConfig
    engine_version: int
    last_command_position: int
    processed_commands: int
    load_failures: list[Failure]
    command_failures: list[Failure]
    runtime_failures: list[Failure]
    command_responses: list[dict[str, Any]]
    last_ready_summary: dict[str, Any] | None
    artifacts: list[StageArtifact]
    manifest: dict[str, Any]
    generation_path: str


class SnapshotManager:
    """Write stages into a fresh generation and atomically switch CURRENT."""

    def __init__(self, workspace: Path | str, run_id: str = "default"):
        self.workspace = Path(workspace)
        self.run_id = "".join(
            char if char.isalnum() or char in {"-", "_", "."} else "_"
            for char in run_id
        ) or "default"
        base = self.workspace / "artifacts" / "snapshots"
        self.root = base / self.run_id

    @property
    def current_pointer(self) -> Path:
        return self.root / "CURRENT"

    def _read_current_generation(self) -> Path | None:
        if not self.current_pointer.exists():
            return None
        name = self.current_pointer.read_text(encoding="utf-8").strip()
        if not name:
            return None
        candidate = self.root / name
        if candidate.is_dir():
            return candidate
        return None

    def load(self) -> tuple[RestoredSnapshot | None, Failure | None]:
        generation = self._read_current_generation()
        if generation is None:
            return None, Failure(
                code=E_SNAPSHOT_INVALID,
                stage="load",
                message="no readable CURRENT snapshot generation",
                locator=str(self.current_pointer),
            )
        try:
            manifest = parse_json_text((generation / "manifest.json").read_text(encoding="utf-8"))
            raw_state = parse_json_text((generation / "state.json").read_text(encoding="utf-8"))
            if not isinstance(manifest, dict) or not isinstance(raw_state, dict):
                raise ValueError("manifest/state must be objects")
            if manifest.get("schema") != SCHEMA_VERSION:
                raise ValueError("snapshot schema mismatch")
            if manifest.get("state_hash") != digest(raw_state):
                raise ValueError("snapshot state hash mismatch")
            state = _decode_value(raw_state)

            samples: dict[str, CorpusSample] = {}
            for record in state.get("samples", []):
                sample = CorpusSample(
                    sample_id=str(record["id"]),
                    chars=tuple(record["chars"]),
                    labels=tuple(record["labels"]),
                    types=tuple(record["types"]),
                )
                samples[sample.sample_id] = sample
            model = _restore_model(state["model"])
            config = _restore_config(state["config"])
            if manifest.get("model_version") != model.version:
                raise ValueError("snapshot model version mismatch")
            if manifest.get("config_hash") != config_fingerprint(config):
                raise ValueError("snapshot config hash mismatch")
            if manifest.get("corpus_hash") != corpus_fingerprint(samples):
                raise ValueError("snapshot corpus hash mismatch")
            if manifest.get("vocabulary_hash") != vocabulary_fingerprint(
                model.vocabulary
            ):
                raise ValueError("snapshot vocabulary hash mismatch")
            if manifest.get("completed_samples") != sorted(samples):
                raise ValueError("snapshot completed sample list mismatch")

            artifacts: list[StageArtifact] = []
            version_map = manifest.get("stage_versions", {})
            for stage in STAGE_ORDER:
                stage_dir = generation / "stages" / stage
                if not stage_dir.exists():
                    continue
                for path in sorted(stage_dir.glob("*.json")):
                    raw = _decode_value(parse_json_text(path.read_text(encoding="utf-8")))
                    artifact = StageArtifact.from_dict(raw)
                    expected = version_map.get(stage, {}).get(artifact.sample_id)
                    if expected != artifact.output_fingerprint:
                        raise ValueError(
                            f"artifact version mismatch for {stage}/{artifact.sample_id}"
                        )
                    artifacts.append(artifact)
            return (
                RestoredSnapshot(
                    samples=samples,
                    model=model,
                    config=config,
                    engine_version=int(state.get("engine_version", 0)),
                    last_command_position=int(state.get("last_command_position", 0)),
                    processed_commands=int(state.get("processed_commands", 0)),
                    load_failures=[
                        _restore_failure(item) for item in state.get("load_failures", [])
                    ],
                    command_failures=[
                        _restore_failure(item) for item in state.get("command_failures", [])
                    ],
                    runtime_failures=[
                        _restore_failure(item) for item in state.get("runtime_failures", [])
                    ],
                    command_responses=list(state.get("command_responses", [])),
                    last_ready_summary=state.get("last_ready_summary"),
                    artifacts=artifacts,
                    manifest=manifest,
                    generation_path=str(generation),
                ),
                None,
            )
        except Exception as exc:
            return None, Failure(
                code=E_SNAPSHOT_INVALID,
                stage="load",
                message=f"cannot restore snapshot: {exc}",
                locator=str(generation),
            )

    def _previous_manifest(self) -> tuple[Path | None, dict[str, Any] | None]:
        generation = self._read_current_generation()
        if generation is None:
            return None, None
        try:
            manifest = parse_json_text((generation / "manifest.json").read_text(encoding="utf-8"))
            return generation, manifest if isinstance(manifest, dict) else None
        except Exception:
            return generation, None

    def _stage_artifact_paths(self, engine: Any) -> dict[tuple[str, str], StageArtifact]:
        result: dict[tuple[str, str], StageArtifact] = {}
        for sample_id in sorted(engine.samples):
            view = engine._sample_stage_view(engine.samples[sample_id])
            for stage in STAGE_ORDER:
                artifact = view["stages"].get(stage, {}).get("artifact")
                if artifact is not None:
                    result[(stage, sample_id)] = artifact
        return result

    def commit(self, engine: Any, *, reason: str) -> dict[str, Any]:
        engine.settle()
        artifacts = self._stage_artifact_paths(engine)
        stage_versions: dict[str, dict[str, str]] = {
            stage: {} for stage in STAGE_ORDER
        }
        for (stage, sample_id), artifact in artifacts.items():
            stage_versions[stage][sample_id] = artifact.output_fingerprint

        state = _sanitize_transient({
            "schema": SCHEMA_VERSION,
            "engine_version": engine.version,
            "last_command_position": engine.last_command_position,
            "processed_commands": engine.processed_commands,
            "samples": [
                engine.samples[sample_id].as_dict() for sample_id in sorted(engine.samples)
            ],
            "model": _encode_value(_model_state(engine.model)),
            "config": _encode_value(_config_state(engine.config)),
            "load_failures": [_failure_state(item) for item in engine.load_failures],
            "command_failures": [
                _failure_state(item) for item in engine.command_failures
            ],
            "runtime_failures": [
                _failure_state(item) for item in engine.current_failures()
            ],
            "command_responses": engine.command_responses,
            "last_ready_summary": engine._last_ready_summary,
        })
        state_hash = digest(state)
        previous_generation, previous_manifest = self._previous_manifest()
        changed_stages: list[str] = []
        if previous_manifest is not None:
            previous_versions = previous_manifest.get("stage_versions", {})
            for stage in STAGE_ORDER:
                old = previous_versions.get(stage, {})
                new = stage_versions.get(stage, {})
                for sample_id in sorted(set(old) | set(new)):
                    if old.get(sample_id) != new.get(sample_id):
                        changed_stages.append(f"{stage}/{sample_id}")
        else:
            for stage in STAGE_ORDER:
                for sample_id in sorted(stage_versions.get(stage, {})):
                    changed_stages.append(f"{stage}/{sample_id}")
        if (
            previous_manifest is not None
            and previous_manifest.get("state_hash") == state_hash
            and not changed_stages
        ):
            return {
                "status": "skipped",
                "reason": "no version change",
                "generation": previous_manifest.get("generation"),
                "manifest": previous_manifest,
            }

        self.root.mkdir(parents=True, exist_ok=True)
        generation_number = int(previous_manifest.get("generation", 0)) + 1 if previous_manifest else 1
        generation_name = f"gen-{generation_number:06d}"
        staging_parent = self.root / ".tmp"
        configured_temp = Path(engine.config.temp_dir)
        try:
            configured_temp.resolve().relative_to(self.workspace.resolve())
            staging_parent = configured_temp
        except (OSError, ValueError):
            pass
        staging_parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f"ner-stage-{generation_name}-",
                dir=str(staging_parent),
            )
        )
        try:
            for stage in STAGE_ORDER:
                (staging / "stages" / stage).mkdir(parents=True, exist_ok=True)
            old_artifact_paths: dict[tuple[str, str], Path] = {}
            if previous_generation is not None:
                for stage in STAGE_ORDER:
                    directory = previous_generation / "stages" / stage
                    if not directory.exists():
                        continue
                    for path in directory.glob("*.json"):
                        try:
                            raw = parse_json_text(path.read_text(encoding="utf-8"))
                            sample_id = str(raw["sample_id"])
                        except Exception:
                            continue
                        old_artifact_paths[(stage, sample_id)] = path

            for (stage, sample_id), artifact in sorted(artifacts.items()):
                filename = quote(sample_id, safe="") + ".json"
                destination = staging / "stages" / stage / filename
                old_path = old_artifact_paths.get((stage, sample_id))
                old_version = (
                    previous_manifest.get("stage_versions", {})
                    .get(stage, {})
                    .get(sample_id)
                    if previous_manifest
                    else None
                )
                if old_path is not None and old_version == artifact.output_fingerprint:
                    try:
                        os.link(old_path, destination)
                    except OSError:
                        shutil.copy2(old_path, destination)
                else:
                    atomic_write_json(destination, artifact.as_dict())

            manifest = {
                "schema": SCHEMA_VERSION,
                "generation": generation_number,
                "run_id": self.run_id,
                "global_version": engine.version,
                "model_version": engine.model.version,
                "config_hash": config_fingerprint(engine.config),
                "corpus_hash": corpus_fingerprint(engine.samples),
                "vocabulary_hash": vocabulary_fingerprint(engine.model.vocabulary),
                "completed_samples": sorted(engine.samples),
                "stage_versions": stage_versions,
                "stage_cache_keys": {
                    sample_id: {
                        stage: view["stages"].get(stage, {}).get("key")
                        for stage in STAGE_ORDER
                    }
                    for sample_id, view in (
                        (
                            sample_id,
                            engine._sample_stage_view(engine.samples[sample_id]),
                        )
                        for sample_id in sorted(engine.samples)
                    )
                },
                "changed_stages": sorted(changed_stages),
                "state_hash": state_hash,
                "command_position": engine.last_command_position,
                "processed_commands": engine.processed_commands,
                "reason": reason,
            }
            atomic_write_json(staging / "state.json", state)
            atomic_write_json(staging / "manifest.json", manifest)
            final_generation = self.root / generation_name
            if final_generation.exists():
                shutil.rmtree(final_generation)
            os.replace(staging, final_generation)
            pointer_temp = self.root / f".CURRENT.{os.getpid()}.tmp"
            atomic_write_text(pointer_temp, generation_name + "\n")
            os.replace(pointer_temp, self.current_pointer)
            self._cleanup_generations(generation_number)
            return {
                "status": "committed",
                "generation": generation_number,
                "generation_path": str(final_generation),
                "manifest": manifest,
                "changed_stages": sorted(changed_stages),
            }
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise

    def _cleanup_generations(self, current_number: int) -> None:
        keep = {f"gen-{number:06d}" for number in range(max(1, current_number - 2), current_number + 1)}
        try:
            entries = list(self.root.iterdir())
        except OSError:
            return
        for entry in entries:
            if not entry.is_dir() or not entry.name.startswith("gen-"):
                continue
            if entry.name in keep:
                continue
            shutil.rmtree(entry, ignore_errors=True)
