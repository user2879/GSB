from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .cache import CallState
from .errors import BusinessError, BusinessFailure
from .jsonutil import (
    fsync_directory,
    fsync_file,
    sanitize_json,
    sha256_file,
    atomic_write_json,
)
from .models import FORMAT_VERSION, CallSpec


def _snapshot_failure(message: str, details: dict[str, Any] | None = None) -> BusinessFailure:
    return BusinessFailure(
        code="E_CONFIG_MISMATCH",
        message=message,
        stage="input",
        details=details or {},
    )


@dataclass
class SnapshotData:
    tensor: np.ndarray
    call_states: list[CallState]
    global_version: int
    snapshot_version: int
    next_command_index: int
    state: dict[str, Any] = field(default_factory=dict)


class SnapshotStore:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.snapshots_dir = self.output_dir / "snapshots"
        self.current_path = self.output_dir / "current.json"
        self._next_snapshot_version = 1
        if self.current_path.exists():
            try:
                import json

                current = json.loads(self.current_path.read_text(encoding="utf-8"))
                self._next_snapshot_version = int(current.get("snapshot_version", 0)) + 1
            except (OSError, ValueError, TypeError):
                pass

    def has_current(self) -> bool:
        return self.current_path.exists()

    def load_current(
        self,
        *,
        config_hash: str,
        source_input_fingerprint: str,
        commands_digest: str,
    ) -> SnapshotData | None:
        if not self.current_path.exists():
            return None
        try:
            import json

            current = json.loads(self.current_path.read_text(encoding="utf-8"))
            manifest_path = Path(current["manifest"])
            if not manifest_path.is_absolute():
                manifest_path = self.output_dir / manifest_path
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise BusinessError(
                _snapshot_failure(f"current snapshot pointer is invalid: {exc}")
            ) from exc

        if int(manifest.get("format_version", -1)) != FORMAT_VERSION:
            raise BusinessError(
                _snapshot_failure(
                    "snapshot format version mismatch",
                    {"expected": FORMAT_VERSION, "actual": manifest.get("format_version")},
                )
            )
        if current.get("config_hash") != manifest.get("config_hash"):
            raise BusinessError(
                _snapshot_failure(
                    "snapshot pointer and manifest configuration hashes disagree",
                    {
                        "pointer": current.get("config_hash"),
                        "manifest": manifest.get("config_hash"),
                    },
                )
            )
        if int(current.get("snapshot_version", -1)) != int(manifest.get("snapshot_version", -2)):
            raise BusinessError(_snapshot_failure("snapshot pointer and manifest versions disagree"))
        if int(current.get("global_version", -1)) != int(manifest.get("global_version", -2)):
            raise BusinessError(_snapshot_failure("snapshot pointer and manifest global versions disagree"))
        if manifest.get("config_hash") != config_hash:
            raise BusinessError(
                _snapshot_failure(
                    "configuration hash mismatch; refusing to mix old snapshot artifacts",
                    {"snapshot": manifest.get("config_hash"), "current": config_hash},
                )
            )
        if manifest.get("source_input_fingerprint") != source_input_fingerprint:
            raise BusinessError(
                _snapshot_failure(
                    "source input tensor fingerprint mismatch",
                    {
                        "snapshot": manifest.get("source_input_fingerprint"),
                        "current": source_input_fingerprint,
                    },
                )
            )
        if manifest.get("commands_digest") != commands_digest:
            raise BusinessError(
                _snapshot_failure(
                    "interactive command stream digest mismatch",
                    {"snapshot": manifest.get("commands_digest"), "current": commands_digest},
                )
            )

        snapshot_dir = manifest_path.parent
        stages = manifest.get("stages", {})
        required = {"input", "results", "state"}
        if not isinstance(stages, dict) or not required.issubset(stages):
            raise BusinessError(_snapshot_failure("snapshot manifest has missing stages"))
        for stage_name, stage in stages.items():
            file_path = snapshot_dir / str(stage["file"])
            if not file_path.is_file():
                raise BusinessError(
                    _snapshot_failure(
                        f"snapshot stage file is missing: {stage_name}",
                        {"file": str(file_path)},
                    )
                )
            if sha256_file(file_path) != stage.get("sha256"):
                raise BusinessError(
                    _snapshot_failure(
                        f"snapshot stage checksum mismatch: {stage_name}",
                        {"file": str(file_path)},
                    )
                )
            if int(stage.get("version", -1)) != int(manifest.get("global_version", -2)):
                raise BusinessError(
                    _snapshot_failure(
                        f"snapshot stage version mismatch: {stage_name}",
                        {"stage": stage.get("version"), "global": manifest.get("global_version")},
                    )
                )

        try:
            import json

            state = json.loads((snapshot_dir / stages["state"]["file"]).read_text(encoding="utf-8"))
            with np.load(snapshot_dir / stages["input"]["file"], allow_pickle=False) as archive:
                tensor = np.asarray(archive["tensor"], dtype=np.float32)
            with np.load(snapshot_dir / stages["results"]["file"], allow_pickle=False) as archive:
                arrays = {name: np.asarray(archive[name]) for name in archive.files}
        except (OSError, ValueError, KeyError) as exc:
            raise BusinessError(
                _snapshot_failure(f"snapshot artifacts cannot be decoded: {exc}")
            ) from exc

        if state.get("config_hash") != config_hash:
            raise BusinessError(_snapshot_failure("snapshot state configuration hash mismatch"))
        if int(state.get("global_version", -1)) != int(manifest.get("global_version", -2)):
            raise BusinessError(_snapshot_failure("snapshot state global version mismatch"))

        states: list[CallState] = []
        for index, call_payload in enumerate(state.get("call_states", [])):
            try:
                spec = CallSpec.from_dict(call_payload["spec"])
                expected_softmax_keys = spec.rows if spec.softmax_axis == 1 else spec.cols
                expected_layernorm_keys = spec.rows if spec.layernorm_axis == 1 else spec.cols
                if len(call_payload.get("softmax_keys", [])) != expected_softmax_keys:
                    raise ValueError(f"softmax cache key count mismatch for {spec.call_id}")
                if len(call_payload.get("layernorm_keys", [])) != expected_layernorm_keys:
                    raise ValueError(f"layernorm cache key count mismatch for {spec.call_id}")
                prefix = f"call_{index}"
                state_item = CallState(
                    spec=spec,
                    softmax_result=np.asarray(arrays[f"{prefix}_softmax"], dtype=np.float32).copy(),
                    layernorm_result=np.asarray(arrays[f"{prefix}_layernorm"], dtype=np.float32).copy(),
                    layernorm_mean=np.asarray(arrays[f"{prefix}_mean"], dtype=np.float32).copy(),
                    layernorm_variance=np.asarray(arrays[f"{prefix}_variance"], dtype=np.float32).copy(),
                    layernorm_epsilon_applied=np.asarray(
                        arrays[f"{prefix}_epsilon_applied"], dtype=bool
                    ).copy(),
                    softmax_keys=list(call_payload.get("softmax_keys", [])),
                    layernorm_keys=list(call_payload.get("layernorm_keys", [])),
                    softmax_failures={
                        int(lane): BusinessFailure.from_dict(failure)
                        for lane, failure in call_payload.get("softmax_failures", {}).items()
                    },
                    layernorm_failures={
                        int(lane): BusinessFailure.from_dict(failure)
                        for lane, failure in call_payload.get("layernorm_failures", {}).items()
                    },
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise BusinessError(
                    _snapshot_failure(f"snapshot call state is invalid: {exc}")
                ) from exc
            for array_name in (
                "softmax_result",
                "layernorm_result",
                "layernorm_mean",
                "layernorm_variance",
                "layernorm_epsilon_applied",
            ):
                if np.asarray(getattr(state_item, array_name)).shape != spec.shape:
                    raise BusinessError(
                        _snapshot_failure(
                            f"snapshot array shape mismatch for {spec.call_id}:{array_name}"
                        )
                    )
            if tensor.shape != spec.shape:
                raise BusinessError(
                    _snapshot_failure(
                        f"snapshot tensor shape mismatch for {spec.call_id}",
                        {"tensor_shape": list(tensor.shape), "call_shape": list(spec.shape)},
                    )
                )
            states.append(state_item)

        self._next_snapshot_version = int(current.get("snapshot_version", 0)) + 1
        return SnapshotData(
            tensor=tensor,
            call_states=states,
            global_version=int(manifest.get("global_version", 0)),
            snapshot_version=int(current.get("snapshot_version", 0)),
            next_command_index=int(state.get("next_command_index", 0)),
            state=state,
        )

    def commit(
        self,
        *,
        tensor: np.ndarray,
        states: list[CallState],
        global_version: int,
        next_command_index: int,
        config_hash: str,
        source_input_fingerprint: str,
        commands_digest: str,
        state_payload: dict[str, Any],
        reuse_recompute_payload: dict[str, Any],
        failures_payload: list[dict[str, Any]],
        reference_arrays: dict[str, np.ndarray] | None = None,
    ) -> tuple[int, Path]:
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        snapshot_version = self._next_snapshot_version
        self._next_snapshot_version += 1
        unique = uuid.uuid4().hex[:10]
        temporary_dir = self.snapshots_dir / f".tmp-{snapshot_version:08d}-{unique}"
        final_dir = self.snapshots_dir / f"v{snapshot_version:08d}-{unique}"
        temporary_dir.mkdir(parents=True, exist_ok=False)

        try:
            input_path = temporary_dir / "input.npz"
            np.savez_compressed(input_path, tensor=np.asarray(tensor, dtype="<f4"))
            fsync_file(input_path)

            results_path = temporary_dir / "results.npz"
            result_arrays: dict[str, np.ndarray] = {}
            for index, state in enumerate(states):
                assert state.softmax_result is not None
                assert state.layernorm_result is not None
                assert state.layernorm_mean is not None
                assert state.layernorm_variance is not None
                assert state.layernorm_epsilon_applied is not None
                prefix = f"call_{index}"
                result_arrays[f"{prefix}_softmax"] = state.softmax_result
                result_arrays[f"{prefix}_layernorm"] = state.layernorm_result
                result_arrays[f"{prefix}_mean"] = state.layernorm_mean
                result_arrays[f"{prefix}_variance"] = state.layernorm_variance
                result_arrays[f"{prefix}_epsilon_applied"] = state.layernorm_epsilon_applied
            np.savez_compressed(results_path, **result_arrays)
            fsync_file(results_path)

            normalized_state = dict(state_payload)
            normalized_state.update(
                {
                    "format_version": FORMAT_VERSION,
                    "global_version": global_version,
                    "snapshot_version": snapshot_version,
                    "next_command_index": next_command_index,
                    "config_hash": config_hash,
                    "source_input_fingerprint": source_input_fingerprint,
                    "current_input_fingerprint": state_payload.get("current_input_fingerprint"),
                    "commands_digest": commands_digest,
                    "call_states": [
                        {
                            "spec": state.spec.to_dict(),
                            "softmax_keys": list(state.softmax_keys),
                            "layernorm_keys": list(state.layernorm_keys),
                            "softmax_failures": {
                                str(lane): failure.to_dict()
                                for lane, failure in sorted(state.softmax_failures.items())
                            },
                            "layernorm_failures": {
                                str(lane): failure.to_dict()
                                for lane, failure in sorted(state.layernorm_failures.items())
                            },
                        }
                        for state in states
                    ],
                }
            )
            state_path = temporary_dir / "state.json"
            atomic_write_json(state_path, normalized_state)

            reuse_path = temporary_dir / "reuse_recompute.json"
            atomic_write_json(reuse_path, reuse_recompute_payload)
            failures_path = temporary_dir / "failures.json"
            atomic_write_json(failures_path, failures_payload)

            stages: dict[str, dict[str, Any]] = {
                "input": {
                    "file": "input.npz",
                    "version": global_version,
                    "sha256": sha256_file(input_path),
                },
                "results": {
                    "file": "results.npz",
                    "version": global_version,
                    "sha256": sha256_file(results_path),
                },
                "state": {
                    "file": "state.json",
                    "version": global_version,
                    "sha256": sha256_file(state_path),
                },
                "reuse_recompute": {
                    "file": "reuse_recompute.json",
                    "version": global_version,
                    "sha256": sha256_file(reuse_path),
                },
                "failures": {
                    "file": "failures.json",
                    "version": global_version,
                    "sha256": sha256_file(failures_path),
                },
            }

            if reference_arrays:
                reference_path = temporary_dir / "reference.npz"
                np.savez_compressed(reference_path, **reference_arrays)
                fsync_file(reference_path)
                stages["reference"] = {
                    "file": "reference.npz",
                    "version": global_version,
                    "sha256": sha256_file(reference_path),
                }

            manifest = {
                "format_version": FORMAT_VERSION,
                "global_version": global_version,
                "snapshot_version": snapshot_version,
                "config_hash": config_hash,
                "source_input_fingerprint": source_input_fingerprint,
                "current_input_fingerprint": state_payload.get("current_input_fingerprint"),
                "input_tensor_fingerprint": state_payload.get("current_input_fingerprint"),
                "commands_digest": commands_digest,
                "stages": stages,
            }
            manifest_path = temporary_dir / "manifest.json"
            atomic_write_json(manifest_path, manifest)
            fsync_directory(temporary_dir)
            os.replace(temporary_dir, final_dir)
            fsync_directory(self.snapshots_dir)

            current_payload = {
                "format_version": FORMAT_VERSION,
                "snapshot_version": snapshot_version,
                "global_version": global_version,
                "config_hash": config_hash,
                "manifest": str((final_dir / "manifest.json").relative_to(self.output_dir)),
            }
            atomic_write_json(self.current_path, current_payload)
        except Exception:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir, ignore_errors=True)
            raise

        return snapshot_version, final_dir


def write_run_outputs(
    output_dir: Path,
    report: dict[str, Any],
    failures: list[dict[str, Any]],
) -> tuple[Path, Path]:
    reports_dir = output_dir / "reports"
    failures_dir = output_dir / "failures"
    report_path = reports_dir / "latest.json"
    failure_path = failures_dir / "latest.json"
    atomic_write_json(report_path, report)
    atomic_write_json(failure_path, {"failures": sanitize_json(failures)})
    return report_path, failure_path
