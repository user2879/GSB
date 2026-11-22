"""Whole-generation snapshot commit, stale detection, and recovery."""

from __future__ import annotations

import copy
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from .cache import ArtifactCache
from .config import RenderConfig
from .errors import E_SNAPSHOT_STALE, RenderError
from .renderer import DeterministicRenderer, RenderResult
from .state import RenderState, effective_config_fingerprint
from .util import (
    array_sha256,
    atomic_write_json,
    encode_png,
    read_json,
    sha256_bytes,
    write_bytes_fast,
    write_json_fast,
)


SNAPSHOT_SCHEMA_VERSION = 1


class SnapshotManager:
    def __init__(self, workdir: str | Path) -> None:
        self.workdir = Path(workdir).resolve()
        self.snapshots_dir = self.workdir / "snapshots"
        self.pointer_path = self.snapshots_dir / "CURRENT.json"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._current = self._read_pointer_or_none()
        self._base_generation = self._current.get("generation") if self._current else None

    # ------------------------------------------------------------------
    # Pointer and recovery
    # ------------------------------------------------------------------
    def _read_pointer_or_none(self) -> dict[str, Any] | None:
        if not self.pointer_path.exists():
            return None
        try:
            pointer = read_json(self.pointer_path)
        except Exception:
            return None
        return pointer if isinstance(pointer, dict) else None

    def current_pointer(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._current)

    def load_for_resume(
        self,
        *,
        config: RenderConfig,
        command_stream_hash: str,
        expected_version: int | None = None,
        expected_projection_fp: str | None = None,
    ) -> dict[str, Any]:
        pointer = self._read_pointer_or_none()
        if pointer is None:
            raise RenderError(E_SNAPSHOT_STALE, "no committed snapshot is available to resume")
        if pointer.get("snapshot_schema_version") != SNAPSHOT_SCHEMA_VERSION:
            raise RenderError(
                E_SNAPSHOT_STALE,
                "snapshot pointer schema version does not match this application",
                {
                    "snapshot_schema_version": pointer.get("snapshot_schema_version"),
                    "expected": SNAPSHOT_SCHEMA_VERSION,
                },
            )
        generation = pointer.get("generation")
        if not isinstance(generation, str) or not generation:
            raise RenderError(E_SNAPSHOT_STALE, "snapshot pointer has no valid generation")
        generation_dir = self.snapshots_dir / generation
        manifest_path = generation_dir / "manifest.json"
        try:
            manifest = read_json(manifest_path)
        except Exception as exc:
            raise RenderError(
                E_SNAPSHOT_STALE,
                "committed snapshot manifest is unreadable",
                {"generation": generation, "error": str(exc)},
            ) from exc
        if not isinstance(manifest, dict):
            raise RenderError(E_SNAPSHOT_STALE, "committed snapshot manifest is invalid")
        if manifest.get("snapshot_schema_version") != SNAPSHOT_SCHEMA_VERSION:
            raise RenderError(
                E_SNAPSHOT_STALE,
                "snapshot manifest schema version mismatch",
                {
                    "manifest_version": manifest.get("snapshot_schema_version"),
                    "expected": SNAPSHOT_SCHEMA_VERSION,
                },
            )
        if pointer.get("generation") != manifest.get("generation"):
            raise RenderError(E_SNAPSHOT_STALE, "snapshot pointer and manifest generation differ")
        if pointer.get("global_version") != manifest.get("global_version"):
            raise RenderError(E_SNAPSHOT_STALE, "snapshot pointer and manifest version differ")
        if expected_version is not None and int(manifest.get("global_version", -1)) != int(expected_version):
            raise RenderError(
                E_SNAPSHOT_STALE,
                "requested recovery version does not match the committed snapshot",
                {
                    "requested_version": expected_version,
                    "manifest_version": manifest.get("global_version"),
                },
            )
        if manifest.get("source_config_hash") != config.source_hash:
            raise RenderError(
                E_SNAPSHOT_STALE,
                "committed snapshot configuration hash does not match the requested configuration",
                {
                    "snapshot_config_hash": manifest.get("source_config_hash"),
                    "requested_config_hash": config.source_hash,
                },
            )
        if manifest.get("command_stream_hash") != command_stream_hash:
            raise RenderError(
                E_SNAPSHOT_STALE,
                "committed snapshot command stream hash does not match the requested command stream",
                {
                    "snapshot_command_hash": manifest.get("command_stream_hash"),
                    "requested_command_hash": command_stream_hash,
                },
            )
        if expected_projection_fp is not None:
            actual_projection_fp = manifest.get("determinism", {}).get("projection_fingerprint")
            if actual_projection_fp != expected_projection_fp:
                raise RenderError(
                    E_SNAPSHOT_STALE,
                    "committed projection fingerprint does not match deterministic regeneration",
                    {
                        "snapshot_projection_fp": actual_projection_fp,
                        "regenerated_projection_fp": expected_projection_fp,
                    },
                )
        try:
            state = RenderState.from_dict(manifest["state"])
            failures = read_json(generation_dir / "reports" / "failures.json")
            reports = read_json(generation_dir / "reports" / "commands.json")
            latest_preview_path = generation_dir / "reports" / "latest_preview.json"
            latest_preview = read_json(latest_preview_path) if latest_preview_path.exists() else None
        except Exception as exc:
            raise RenderError(
                E_SNAPSHOT_STALE,
                "committed snapshot state or reports are corrupt",
                {"generation": generation, "error": str(exc)},
            ) from exc
        cache = ArtifactCache.load(generation_dir / "cache")
        if manifest.get("cache_counts") != cache.counts():
            raise RenderError(
                E_SNAPSHOT_STALE,
                "committed cache counts do not match its manifest",
                {"manifest": manifest.get("cache_counts"), "loaded": cache.counts()},
            )
        if cache.counts()["total"] != len(read_json(generation_dir / "cache" / "index.json").get("entries", [])):
            raise RenderError(E_SNAPSHOT_STALE, "committed cache index is internally inconsistent")
        self._current = pointer
        self._base_generation = str(generation)
        return {
            "state": state,
            "cache": cache,
            "attempt_index": int(manifest.get("attempt_index", state.command_cursor)),
            "failures": failures,
            "command_reports": reports,
            "latest_preview": latest_preview,
            "manifest": manifest,
            "generation": generation,
        }

    # ------------------------------------------------------------------
    # Whole-generation commit
    # ------------------------------------------------------------------
    def _next_generation_name(self) -> str:
        highest = 0
        for entry in self.snapshots_dir.iterdir():
            if entry.is_dir() and entry.name.startswith("gen-"):
                try:
                    highest = max(highest, int(entry.name.split("-")[1]))
                except (IndexError, ValueError):
                    continue
        return f"gen-{highest + 1:06d}"

    def _verify_base_generation(self) -> None:
        pointer = self._read_pointer_or_none()
        observed = pointer.get("generation") if pointer else None
        if observed != self._base_generation:
            raise RenderError(
                E_SNAPSHOT_STALE,
                "snapshot generation changed before whole-snapshot replacement",
                {
                    "expected_base_generation": self._base_generation,
                    "observed_generation": observed,
                },
            )

    def commit(
        self,
        *,
        config: RenderConfig,
        command_stream_hash: str,
        state: RenderState,
        cache: ArtifactCache,
        renderer: DeterministicRenderer,
        attempt_index: int,
        failures: list[dict[str, Any]],
        command_reports: list[dict[str, Any]],
        latest_preview: dict[str, Any] | None,
        render_result: RenderResult | None,
    ) -> dict[str, Any]:
        self._verify_base_generation()
        generation = self._next_generation_name()
        temp_dir = self.snapshots_dir / f".tmp-{generation}-{os.getpid()}-{time.time_ns()}"
        final_dir = self.snapshots_dir / generation
        temp_dir.mkdir(parents=True, exist_ok=False)
        try:
            required_keys = {record["key"] for record in render_result.audit} if render_result else None
            cache_index = cache.save(temp_dir / "cache", include_keys=required_keys)
            cache_counts = cache_index["counts"]
            self._write_determinism(temp_dir, renderer)
            self._write_reports(temp_dir, failures, command_reports, latest_preview)
            output_manifest = self._write_outputs(temp_dir, cache, render_result, latest_preview)
            completed_stages = self._completed_stages(render_result, latest_preview)
            effective_fp = effective_config_fingerprint(config, state)
            manifest = {
                "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
                "generation": generation,
                "global_version": state.version,
                "attempt_index": attempt_index,
                "source_config_hash": config.source_hash,
                "command_stream_hash": command_stream_hash,
                "effective_config_fingerprint": effective_fp,
                "seed": state.seed,
                "steps": state.steps,
                "completed_stages": completed_stages,
                "cursors": {
                    "command": attempt_index,
                    "step": state.steps - 1,
                    "frame": state.frame_count - 1,
                    "block_row": min(config.height, state.chunk_rows) - 1,
                },
                "state": state.to_dict(),
                "determinism": renderer.determinism_metadata(),
                "cache_counts": cache_counts,
                "outputs": output_manifest,
                "failure_count": len(failures),
                "command_count": len(command_reports),
            }
            atomic_write_json(temp_dir / "manifest.json", manifest)
            atomic_write_json(temp_dir / "state.json", state.to_dict())

            # Rename the complete generation (not individual files) into place.
            os.replace(temp_dir, final_dir)
            self._verify_base_generation()
            pointer = {
                "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
                "generation": generation,
                "global_version": state.version,
                "attempt_index": attempt_index,
                "source_config_hash": config.source_hash,
                "command_stream_hash": command_stream_hash,
                "effective_config_fingerprint": effective_fp,
                "manifest": f"{generation}/manifest.json",
            }
            pointer_tmp = self.snapshots_dir / f".CURRENT.{os.getpid()}.{time.time_ns()}.tmp"
            atomic_write_json(pointer_tmp, pointer)
            os.replace(pointer_tmp, self.pointer_path)
            self._current = pointer
            self._base_generation = generation
            return pointer
        except BaseException:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def _write_determinism(self, root: Path, renderer: DeterministicRenderer) -> None:
        directory = root / "determinism"
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "model_tables.npz",
            projection=renderer.projection,
            pixel_bias=renderer.pixel_bias,
            condition_matrix=renderer.condition_matrix,
        )
        write_json_fast(directory / "fingerprints.json", renderer.determinism_metadata())

    def _write_reports(
        self,
        root: Path,
        failures: list[dict[str, Any]],
        command_reports: list[dict[str, Any]],
        latest_preview: dict[str, Any] | None,
    ) -> None:
        reports = root / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        write_json_fast(reports / "failures.json", failures)
        write_json_fast(reports / "commands.json", command_reports)
        write_json_fast(reports / "latest_preview.json", latest_preview)
        write_json_fast(
            reports / "reuse_recompute.json",
            [
                {
                    "command_id": item.get("command_id"),
                    "reuse_keys": item.get("reuse_keys", []),
                    "recompute_keys": item.get("recompute_keys", []),
                }
                for item in command_reports
            ],
        )

    def _write_outputs(
        self,
        root: Path,
        cache: ArtifactCache,
        result: RenderResult | None,
        latest_preview: dict[str, Any] | None,
    ) -> dict[str, Any]:
        outputs = root / "outputs"
        frames_dir = outputs / "frames"
        blocks_dir = outputs / "blocks"
        frames_dir.mkdir(parents=True, exist_ok=True)
        blocks_dir.mkdir(parents=True, exist_ok=True)
        if result is None:
            return {"frames": [], "preview_strip": None}

        frame_records: list[dict[str, Any]] = []
        for frame_index, frame_key in enumerate(result.frame_keys):
            decoded = cache.decodes.get(frame_key)
            if decoded is None:
                raise RenderError(E_SNAPSHOT_STALE, "final frame is missing from the cache bundle")
            frame_name = f"frame_{frame_index:03d}.png"
            frame_meta_name = f"frame_{frame_index:03d}.json"
            write_bytes_fast(frames_dir / frame_name, decoded.png)
            block_records = []
            for block in decoded.blocks:
                block_start = int(block["row_start"])
                block_end = int(block["row_end"])
                block_index = int(block["block_index"])
                block_pixels = decoded.pixels[block_start:block_end, :, :]
                block_png = encode_png(block_pixels)
                block_name = f"frame_{frame_index:03d}_block_{block_index:03d}.png"
                write_bytes_fast(blocks_dir / block_name, block_png)
                block_records.append(
                    {
                        **block,
                        "png_file": f"blocks/{block_name}",
                        "png_sha256": sha256_bytes(block_png),
                    }
                )
            frame_meta = {
                "frame_index": frame_index,
                "frame_fingerprint": decoded.content_fp,
                "pixel_sha256": array_sha256(decoded.pixels),
                "png_sha256": sha256_bytes(decoded.png),
                "png_file": f"frames/{frame_name}",
                "decode_params": decoded.params,
                "blocks": block_records,
            }
            write_json_fast(frames_dir / frame_meta_name, frame_meta)
            frame_records.append(frame_meta)

        preview_strip_name = "preview_strip.png"
        write_bytes_fast(outputs / preview_strip_name, result.preview_png)
        preview = {
            "schema_version": 1,
            "generation_format": "ordered-png-frame-sequence",
            "seed": result.seed,
            "steps": result.steps,
            "config_fingerprint": result.config_fingerprint,
            "composition_fingerprint": result.composition_fp,
            "preview_png_sha256": sha256_bytes(result.preview_png),
            "preview_strip": preview_strip_name,
            "frame_count": len(frame_records),
            "frames": frame_records,
            "latest_preview": latest_preview,
        }
        write_json_fast(outputs / "preview.json", preview)
        return {
            "preview_strip": preview_strip_name,
            "preview_manifest": "preview.json",
            "frames": [record["png_file"] for record in frame_records],
            "block_count": sum(len(record["blocks"]) for record in frame_records),
        }

    def _completed_stages(
        self, result: RenderResult | None, latest_preview: dict[str, Any] | None
    ) -> list[str]:
        if result is not None:
            stages = ["encoding"]
            if result.frame_traces:
                first = result.frame_traces[0]
                if first.get("initialization", {}).get("mode") == "image":
                    stages.append("image_encoding")
                stages.extend(["initialization", "denoise", "decode", "composition"])
            return stages
        if latest_preview is not None:
            return ["encoding", "initialization", "denoise", "decode", "composition"]
        return ["config_validation"]


def write_top_level_diagnostics(workdir: str | Path, diagnostics: dict[str, Any]) -> Path:
    path = Path(workdir).resolve() / "reports" / "diagnostics.json"
    atomic_write_json(path, diagnostics)
    return path
