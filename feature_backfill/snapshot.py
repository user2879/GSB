"""Generation-based atomic snapshot store with Windows-safe replacement."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from threading import RLock
from typing import Any

from .errors import AppError, E_SNAPSHOT_MISSING, E_SNAPSHOT_STALE
from .fingerprints import sha256_json, sha256_text


FORMAT_VERSION = 1
REQUIRED_STAGES = ("load_validation", "window_features", "entity_summary")


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_text(text)


def _replace_file(source: Path, target: Path) -> None:
    os.replace(source, target)


class SnapshotStore:
    """Publish immutable generation directories via one atomic pointer file.

    Directory renames alone are not reliably replace-existing operations on
    Windows. A generation directory is therefore fully materialised first and
    a tiny CURRENT pointer is atomically replaced. Unpointed temporary
    generations are ignored by recovery.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.generations_dir = self.root / "generations"
        self.current_file = self.root / "CURRENT"
        self._lock = RLock()
        self.generations_dir.mkdir(parents=True, exist_ok=True)

    def _current_payload(self) -> dict[str, Any] | None:
        if not self.current_file.exists():
            return None
        try:
            return json.loads(self.current_file.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise AppError(
                E_SNAPSHOT_STALE,
                f"cannot parse CURRENT pointer: {exc}",
                stage="snapshot",
                details={"path": str(self.current_file)},
            ) from exc

    def current_generation(self) -> int | None:
        payload = self._current_payload()
        if payload is None:
            return None
        generation = payload.get("generation")
        if isinstance(generation, bool) or not isinstance(generation, int):
            raise AppError(
                E_SNAPSHOT_STALE,
                "CURRENT pointer has invalid generation",
                stage="snapshot",
                details={"payload": payload},
            )
        return generation

    def _next_generation(self) -> int:
        generations: list[int] = []
        for child in self.generations_dir.iterdir():
            if not child.is_dir() or not child.name.startswith("gen-"):
                continue
            suffix = child.name[4:]
            if suffix.isdigit():
                generations.append(int(suffix))
        return max(generations, default=0) + 1

    def _write_generation(
        self,
        directory: Path,
        *,
        generation: int,
        state_payload: dict[str, Any],
        manifest_fields: dict[str, Any],
        incomplete_after_stage: str | None = None,
    ) -> dict[str, Any]:
        directory.mkdir(parents=True, exist_ok=False)
        stage_dir = directory / "stages"
        stage_dir.mkdir(parents=True, exist_ok=False)
        state_hash = _write_json(directory / "state.json", state_payload)
        load_validation_payload = {
            "status": "committed",
            "entities": state_payload.get("entities", []),
            "failure_count": len(state_payload.get("failures", [])),
            "dataset_fingerprint": state_payload.get("dataset_fingerprint"),
        }
        load_validation_hash = _write_json(
            stage_dir / "load_validation.json", load_validation_payload
        )
        window_payload = {
            "status": "committed",
            "result_count": len(state_payload.get("results", [])),
            "skip_count": len(state_payload.get("skips", [])),
            "completed_entities": state_payload.get("completed_entities", []),
        }
        stage_hashes: dict[str, str] = {"load_validation": load_validation_hash}
        if incomplete_after_stage == "load_validation":
            return {
                "generation": generation,
                "incomplete": True,
                "stage_hashes": stage_hashes,
            }
        window_hash = _write_json(stage_dir / "window_features.json", window_payload)
        stage_hashes["window_features"] = window_hash
        if incomplete_after_stage == "window_features":
            return {
                "generation": generation,
                "incomplete": True,
                "stage_hashes": stage_hashes,
            }
        summary_payload = {
            "status": "committed",
            "summary_count": len(state_payload.get("summaries", [])),
            "completed_entities": state_payload.get("completed_entities", []),
            "lease_version": state_payload.get("lease_version"),
        }
        summary_hash = _write_json(stage_dir / "entity_summary.json", summary_payload)
        stage_hashes["entity_summary"] = summary_hash
        manifest: dict[str, Any] = {
            **manifest_fields,
            "format_version": FORMAT_VERSION,
            "generation": generation,
            "state_sha256": state_hash,
            "stages": {
                "load_validation": {"status": "committed", "sha256": load_validation_hash},
                "window_features": {"status": "committed", "sha256": window_hash},
                "entity_summary": {"status": "committed", "sha256": summary_hash},
            },
        }
        manifest["manifest_sha256"] = sha256_json(manifest)
        _write_json(directory / "manifest.json", manifest)
        with (directory / "COMMITTED").open("w", encoding="ascii", newline="\n") as handle:
            handle.write(manifest["manifest_sha256"] + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return manifest

    def commit(
        self,
        *,
        state_payload: dict[str, Any],
        manifest_fields: dict[str, Any],
    ) -> dict[str, Any]:
        with self._lock:
            generation = self._next_generation()
            temp = self.root / f".tmp-{generation:08d}-{uuid.uuid4().hex}"
            manifest = self._write_generation(
                temp,
                generation=generation,
                state_payload=state_payload,
                manifest_fields=manifest_fields,
            )
            final_dir = self.generations_dir / f"gen-{generation:08d}"
            os.rename(temp, final_dir)
            pointer = {
                "generation": generation,
                "manifest_sha256": manifest["manifest_sha256"],
            }
            pointer_temp = self.root / f".CURRENT-{uuid.uuid4().hex}"
            _write_json(pointer_temp, pointer)
            _replace_file(pointer_temp, self.current_file)
            return manifest

    def stage_incomplete(
        self,
        *,
        state_payload: dict[str, Any],
        manifest_fields: dict[str, Any],
        incomplete_after_stage: str = "window_features",
    ) -> Path:
        """Write a deliberately unpublishable generation, for crash tests."""

        with self._lock:
            generation = self._next_generation()
            temp = self.root / f".tmp-crash-{generation:08d}-{uuid.uuid4().hex}"
            self._write_generation(
                temp,
                generation=generation,
                state_payload=state_payload,
                manifest_fields=manifest_fields,
                incomplete_after_stage=incomplete_after_stage,
            )
            return temp

    def load_current(self) -> tuple[dict[str, Any], dict[str, Any], int] | None:
        with self._lock:
            pointer = self._current_payload()
            if pointer is None:
                return None
            generation = pointer.get("generation")
            manifest_sha = pointer.get("manifest_sha256")
            if isinstance(generation, bool) or not isinstance(generation, int):
                raise AppError(
                    E_SNAPSHOT_STALE,
                    "CURRENT generation is invalid",
                    stage="snapshot",
                    details={"pointer": pointer},
                )
            generation_dir = self.generations_dir / f"gen-{generation:08d}"
            committed_marker = generation_dir / "COMMITTED"
            manifest_path = generation_dir / "manifest.json"
            state_path = generation_dir / "state.json"
            if not (generation_dir.is_dir() and committed_marker.exists() and manifest_path.exists()):
                raise AppError(
                    E_SNAPSHOT_STALE,
                    "CURRENT points to a non-committed snapshot generation",
                    stage="snapshot",
                    details={"generation": generation, "path": str(generation_dir)},
                )
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                state_text = state_path.read_text(encoding="utf-8")
                state_payload = json.loads(state_text)
                marker = committed_marker.read_text(encoding="ascii").strip()
            except Exception as exc:  # noqa: BLE001
                raise AppError(
                    E_SNAPSHOT_STALE,
                    f"cannot read committed snapshot: {exc}",
                    stage="snapshot",
                    details={"generation": generation},
                ) from exc
            recorded_sha = manifest.get("manifest_sha256")
            manifest_without_sha = dict(manifest)
            manifest_without_sha.pop("manifest_sha256", None)
            if (
                recorded_sha != sha256_json(manifest_without_sha)
                or marker != recorded_sha
                or manifest_sha != recorded_sha
            ):
                raise AppError(
                    E_SNAPSHOT_STALE,
                    "snapshot manifest hash or COMMITTED marker mismatch",
                    stage="snapshot",
                    details={"generation": generation},
                )
            if manifest.get("state_sha256") != sha256_text(state_text):
                raise AppError(
                    E_SNAPSHOT_STALE,
                    "snapshot state.json does not match its recorded hash",
                    stage="snapshot",
                    details={"generation": generation},
                )
            stages = manifest.get("stages", {})
            for stage_name in REQUIRED_STAGES:
                stage = stages.get(stage_name, {})
                if stage.get("status") != "committed":
                    raise AppError(
                        E_SNAPSHOT_STALE,
                        f"snapshot stage {stage_name} is incomplete",
                        stage="snapshot",
                        details={"generation": generation},
                    )
                stage_path = generation_dir / "stages" / f"{stage_name}.json"
                if not stage_path.exists():
                    raise AppError(
                        E_SNAPSHOT_STALE,
                        f"snapshot stage file {stage_name}.json is missing",
                        stage="snapshot",
                        details={"generation": generation},
                    )
                if sha256_text(stage_path.read_text(encoding="utf-8")) != stage.get("sha256"):
                    raise AppError(
                        E_SNAPSHOT_STALE,
                        f"snapshot stage {stage_name} hash mismatch",
                        stage="snapshot",
                        details={"generation": generation},
                    )
            return manifest, state_payload, generation

    def assert_compatible(
        self,
        manifest: dict[str, Any],
        *,
        config_hash: str,
        plan_hash: str,
        base_input_hash: str,
        expected_lease_version: str,
        expected_global_version: int,
        expected_dataset_fingerprint: str,
        expected_feature_fingerprint: str,
        expected_lease_versions: dict[str, str] | None = None,
        expected_completed_batches: list[int] | None = None,
        expected_pending: list[str] | None = None,
        expected_completed_entities: list[str] | None = None,
    ) -> None:
        checks = {
            "config_hash": config_hash,
            "plan_hash": plan_hash,
            "base_input_hash": base_input_hash,
            "lease_version": expected_lease_version,
            "global_version": expected_global_version,
            "dataset_fingerprint": expected_dataset_fingerprint,
            "feature_fingerprint": expected_feature_fingerprint,
        }
        mismatches = {
            key: {"expected": expected, "actual": manifest.get(key)}
            for key, expected in checks.items()
            if manifest.get(key) != expected
        }
        if expected_lease_versions is not None and manifest.get("lease_versions") != expected_lease_versions:
            mismatches["lease_versions"] = {
                "expected": expected_lease_versions,
                "actual": manifest.get("lease_versions"),
            }
        if expected_completed_batches is not None and manifest.get("completed_batches") != expected_completed_batches:
            mismatches["completed_batches"] = {
                "expected": expected_completed_batches,
                "actual": manifest.get("completed_batches"),
            }
        if expected_pending is not None and manifest.get("pending") != expected_pending:
            mismatches["pending"] = {
                "expected": expected_pending,
                "actual": manifest.get("pending"),
            }
        if (
            expected_completed_entities is not None
            and manifest.get("completed_entities") != expected_completed_entities
        ):
            mismatches["completed_entities"] = {
                "expected": expected_completed_entities,
                "actual": manifest.get("completed_entities"),
            }
        if mismatches:
            raise AppError(
                E_SNAPSHOT_STALE,
                "snapshot is stale or belongs to a different run; refusing partial resume",
                stage="snapshot",
                details={"mismatches": mismatches, "generation": manifest.get("generation")},
            )

    def require_current(self) -> tuple[dict[str, Any], dict[str, Any], int]:
        loaded = self.load_current()
        if loaded is None:
            raise AppError(
                E_SNAPSHOT_MISSING,
                "resume requested but no committed snapshot exists",
                stage="snapshot",
                details={"root": str(self.root)},
            )
        return loaded

    def uncommitted_temporary_generations(self) -> list[str]:
        return sorted(
            str(child)
            for child in self.root.iterdir()
            if child.is_dir() and child.name.startswith(".tmp-")
        )
