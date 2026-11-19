from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, digest, read_json
from .cache import PIPELINE_CACHE_VERSION
from .font import FONT_VERSION


class SnapshotError(Exception):
    pass


class SnapshotStore:
    """A snapshot is one whole JSON document, replaced atomically per version."""

    def __init__(self, root: Path, config_hash: str) -> None:
        self.root = root
        self.config_hash = config_hash
        self.current_path = root / "current.json"
        self.last_written_version = 0
        self.last_document_digest: str | None = None
        self.write_count = 0

    def load(self) -> dict[str, Any] | None:
        if not self.current_path.is_file():
            return None
        value = read_json(self.current_path)
        if not isinstance(value, dict):
            raise SnapshotError("snapshot root is not an object")
        self.last_written_version = int(value.get("global_version", 0))
        self.last_document_digest = digest(value)
        return value

    def commit(
        self,
        *,
        global_version: int,
        command_index: int,
        command_id: str,
        stages: dict[str, dict[str, Any]],
        config_hash: str | None = None,
    ) -> dict[str, Any]:
        if global_version <= 0:
            raise SnapshotError("global_version must be positive")
        if config_hash is not None:
            self.config_hash = config_hash
        previous = None
        try:
            previous = self.load()
        except (OSError, ValueError) as exc:
            raise SnapshotError(f"existing snapshot is unreadable: {exc}") from exc
        previous_version = int(previous.get("global_version", 0)) if previous else 0
        if previous_version == global_version and previous is not None:
            document = previous
            self.last_written_version = previous_version
            self.last_document_digest = digest(document)
            return {
                "written": False,
                "reason": "version_unchanged",
                "global_version": global_version,
                "document_digest": self.last_document_digest,
            }
        stage_docs: dict[str, dict[str, Any]] = {}
        for stage_name in ("detection", "line_aggregation", "segmentation", "recognition"):
            stage = stages.get(stage_name, {})
            stage_docs[stage_name] = {
                "stage": stage_name,
                "config_hash": self.config_hash,
                "font_version": FONT_VERSION,
                "completed_images": list(stage.get("completed_images", [])),
                "consumed_points": int(stage.get("consumed_points", 0)),
                "product_summary": list(stage.get("product_summary", [])),
                "global_version": global_version,
            }
        document = {
            "schema": "layout-stage-snapshot-v1",
            "global_version": global_version,
            "previous_global_version": previous_version,
            "command_index": command_index,
            "command_id": command_id,
            "config_hash": self.config_hash,
            "font_version": FONT_VERSION,
            "pipeline_cache_version": PIPELINE_CACHE_VERSION,
            "replacement_mode": "whole_document_atomic_replace",
            "stages": stage_docs,
        }
        atomic_write_json(self.current_path, document)
        self.last_written_version = global_version
        self.last_document_digest = digest(document)
        self.write_count += 1
        return {
            "written": True,
            "reason": "version_changed" if previous_version else "initial",
            "global_version": global_version,
            "previous_global_version": previous_version,
            "document_digest": self.last_document_digest,
        }

    def verify(self, expected_config_hash: str | None = None) -> dict[str, Any]:
        document = self.load()
        if document is None:
            raise SnapshotError("snapshot is missing")
        if document.get("schema") != "layout-stage-snapshot-v1":
            raise SnapshotError("unsupported snapshot schema")
        expected_hash = expected_config_hash or self.config_hash
        if document.get("config_hash") != expected_hash:
            raise SnapshotError("snapshot config hash does not match active config")
        if document.get("font_version") != FONT_VERSION:
            raise SnapshotError("snapshot font version does not match built-in fonts")
        if document.get("pipeline_cache_version") != PIPELINE_CACHE_VERSION:
            raise SnapshotError("snapshot pipeline cache version does not match implementation")
        version = document.get("global_version")
        if not isinstance(version, int) or version <= 0:
            raise SnapshotError("snapshot global version is invalid")
        stages = document.get("stages")
        if not isinstance(stages, dict):
            raise SnapshotError("snapshot stages are missing")
        expected = {"detection", "line_aggregation", "segmentation", "recognition"}
        if set(stages) != expected:
            raise SnapshotError("snapshot must contain all four stages")
        summaries = 0
        for stage_name, stage in stages.items():
            if stage.get("stage") != stage_name:
                raise SnapshotError(f"stage identity mismatch: {stage_name}")
            if stage.get("global_version") != version:
                raise SnapshotError(f"stage version mismatch: {stage_name}")
            if stage.get("config_hash") != expected_hash:
                raise SnapshotError(f"stage config hash mismatch: {stage_name}")
            summaries += len(stage.get("product_summary", []))
        return {
            "valid": True,
            "global_version": version,
            "config_hash": document["config_hash"],
            "font_version": document["font_version"],
            "stage_count": len(stages),
            "summary_count": summaries,
            "replacement_mode": document.get("replacement_mode"),
        }


def snapshot_stage_docs(
    image_reports: list[dict[str, Any]],
    cache_keys: dict[str, dict[str, str]],
    *,
    consumed_points: int,
    pending_images: set[str],
) -> dict[str, dict[str, Any]]:
    stages: dict[str, dict[str, Any]] = {}
    for stage_name in ("detection", "line_aggregation", "segmentation", "recognition"):
        completed_images: list[str] = []
        summaries: list[dict[str, Any]] = []
        for image in image_reports:
            name = image["name"]
            if image.get("status") != "ok":
                continue
            key_doc = cache_keys.get(name, {})
            key = key_doc.get(stage_name)
            if not key:
                continue
            if stage_name == "recognition" and name in pending_images:
                continue
            completed_images.append(name)
            summary_doc = {
                "image": name,
                "stage": stage_name,
                "cache_key": key,
                "state": image.get("stages", {}).get(stage_name, {}).get("state"),
                "product_count": image.get("stages", {}).get(stage_name, {}).get("product_count"),
                "reused_items": image.get("stages", {}).get(stage_name, {}).get("reused_items"),
                "recomputed_items": image.get("stages", {}).get(stage_name, {}).get("recomputed_items"),
                "pending_characters": image.get("stages", {}).get("recognition", {}).get("pending_characters"),
            }
            if stage_name == "recognition":
                summary_doc["char_cache_keys"] = list(
                    image.get("stages", {}).get("recognition", {}).get("char_cache_keys", [])
                )
            summaries.append(
                {
                    "image": name,
                    "cache_key": key,
                    "product_summary": summary_doc,
                    "product_summary_digest": digest(summary_doc),
                }
            )
        stages[stage_name] = {
            "completed_images": completed_images,
            "consumed_points": consumed_points if stage_name == "recognition" else 0,
            "product_summary": summaries,
        }
    return stages


def snapshot_cache_audit(document: dict[str, Any], cache: Any) -> dict[str, Any]:
    missing: list[dict[str, str]] = []
    checked = 0
    stages = document.get("stages", {})
    for stage_name, stage in stages.items():
        for summary in stage.get("product_summary", []):
            image = summary.get("image", "")
            if stage_name == "recognition":
                keys = summary.get("product_summary", {}).get("char_cache_keys", [])
            else:
                keys = [summary.get("cache_key")]
            for key in keys:
                if not key:
                    continue
                checked += 1
                if not cache.exists(str(key)):
                    missing.append(
                        {"image": str(image), "stage": str(stage_name), "cache_key": str(key)}
                    )
    return {
        "valid": not missing,
        "checked_cache_objects": checked,
        "missing": missing,
    }
