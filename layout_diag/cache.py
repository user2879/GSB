"""Content-addressed stage cache with deterministic key derivation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import EffectiveConfig
from .utils import stable_hash


ALGORITHM_TAG = "layout-diagnostics-v1"


@dataclass
class CacheLookup:
    status: str
    key: str
    result: Any | None


class DiagnosticCache:
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.data = data if isinstance(data, dict) else {}
        if self.data.get("schema_version") != 1:
            self.data = {"schema_version": 1, "images": {}}
        if not isinstance(self.data.get("images"), dict):
            self.data["images"] = {}
        self.next_data: dict[str, Any] = {"schema_version": 1, "images": {}}

    def image_bucket(self, image: str) -> dict[str, Any]:
        images = self.data["images"]
        bucket = images.get(image)
        if not isinstance(bucket, dict):
            return {}
        return bucket

    def lookup_image(self, image: str, stage: str, key: str, image_fp: str) -> CacheLookup:
        bucket = self.image_bucket(image)
        if bucket.get("image_fp") != image_fp:
            return CacheLookup("miss", key, None)
        stage_data = bucket.get(stage)
        if isinstance(stage_data, dict) and stage_data.get("key") == key:
            return CacheLookup("hit", key, stage_data.get("result"))
        return CacheLookup("miss", key, None)

    def lookup_column(self, image: str, column: str, key: str, image_fp: str) -> CacheLookup:
        bucket = self.image_bucket(image)
        if bucket.get("image_fp") != image_fp:
            return CacheLookup("miss", key, None)
        columns = bucket.get("keypoints", {})
        if not isinstance(columns, dict):
            return CacheLookup("miss", key, None)
        stage_data = columns.get(column)
        if isinstance(stage_data, dict) and stage_data.get("key") == key:
            return CacheLookup("hit", key, stage_data.get("result"))
        return CacheLookup("miss", key, None)

    def put_image(self, image: str, image_fp: str, stage: str, key: str, result: Any) -> None:
        bucket = self.next_data["images"].setdefault(image, {"image_fp": image_fp})
        bucket["image_fp"] = image_fp
        bucket[stage] = {"key": key, "result": result}

    def put_column(self, image: str, image_fp: str, column: str, key: str, result: Any) -> None:
        bucket = self.next_data["images"].setdefault(image, {"image_fp": image_fp})
        bucket["image_fp"] = image_fp
        columns = bucket.setdefault("keypoints", {})
        columns[column] = {"key": key, "result": result}

    def finish(self) -> dict[str, Any]:
        for image, bucket in self.next_data["images"].items():
            images = bucket.setdefault("keypoints", {})
            if isinstance(images, dict):
                self.next_data["images"][image]["keypoints"] = dict(sorted(images.items()))
        self.next_data["images"] = dict(sorted(self.next_data["images"].items()))
        return self.next_data


def fingerprint_bundle(image_fp: str, effective: EffectiveConfig) -> dict[str, str]:
    params_fp = effective.params_hash()
    quant_fp = stable_hash(effective.precision.as_dict(), length=32)
    return {
        "image_fp": image_fp,
        "param_fp": params_fp,
        "quant_fp": quant_fp,
        "algorithm": ALGORITHM_TAG,
    }


def sliding_key(image_fp: str, effective: EffectiveConfig) -> str:
    return stable_hash(
        {
            "image_fp": image_fp,
            "parameter_fp": stable_hash(
                {
                    "window": effective.window.as_dict(),
                    "ink_threshold": effective.cc.ink_threshold,
                    "resolution": effective.resolution,
                },
                length=32,
            ),
            "algorithm": ALGORITHM_TAG,
            "stage": "sliding",
            "window": effective.window.as_dict(),
            "ink_threshold": effective.cc.ink_threshold,
        }
    )


def component_key(image_fp: str, effective: EffectiveConfig) -> str:
    return stable_hash(
        {
            "image_fp": image_fp,
            "parameter_fp": stable_hash(
                {
                    "cc": effective.cc.as_dict(),
                    "resolution": effective.resolution,
                },
                length=32,
            ),
            "algorithm": ALGORITHM_TAG,
            "stage": "components",
            "cc": effective.cc.as_dict(),
        }
    )


def keypoint_column_key(
    image_fp: str,
    effective: EffectiveConfig,
    sliding_key_value: str,
    component_key_value: str,
    column_id: str,
    component_ids: list[str],
    allocated_points: int,
) -> str:
    return stable_hash(
        {
            "image_fp": image_fp,
            "parameter_fp": stable_hash(
                {
                    "sliding_key": sliding_key_value,
                    "component_key": component_key_value,
                    "column": column_id,
                    "component_ids": component_ids,
                    "allocated_points": allocated_points,
                    "precision": effective.precision.as_dict(),
                    "regression": {"alpha": 0.65, "boundary_expand_fraction": 0.06},
                },
                length=32,
            ),
            "quant_fp": stable_hash(effective.precision.as_dict(), length=32),
            "algorithm": ALGORITHM_TAG,
            "stage": "keypoints",
            "sliding_key": sliding_key_value,
            "component_key": component_key_value,
            "column": column_id,
            "component_ids": component_ids,
            "allocated_points": allocated_points,
            "precision": effective.precision.as_dict(),
            "regression": {"alpha": 0.65, "boundary_expand_fraction": 0.06},
        }
    )


def reading_key(
    image_fp: str,
    effective: EffectiveConfig,
    column_keys: list[tuple[str, str]],
    processed_columns: list[str],
    ordered_keypoint_ids: list[str],
) -> str:
    return stable_hash(
        {
            "image_fp": image_fp,
            "parameter_fp": stable_hash(
                {
                    "column_keys": sorted(column_keys),
                    "processed_columns": sorted(processed_columns),
                    "ordered_keypoint_ids": ordered_keypoint_ids,
                    "precision": effective.precision.as_dict(),
                },
                length=32,
            ),
            "quant_fp": stable_hash(effective.precision.as_dict(), length=32),
            "algorithm": ALGORITHM_TAG,
            "stage": "reading_order",
            "column_keys": sorted(column_keys),
            "processed_columns": sorted(processed_columns),
            "ordered_keypoint_ids": ordered_keypoint_ids,
        }
    )


def reuse_entry(
    *,
    image: str,
    scope: str,
    stage: str,
    product_id: str,
    key: str,
    status: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "image": image,
        "scope": scope,
        "stage": stage,
        "product_id": product_id,
        "key": key,
        "status": status,
        "reason": reason,
    }
