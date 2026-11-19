from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import numpy as np

from .codec import V1_HEADER, V1_MAGIC, V1_RECORD, V1_RUN, V1_VERSION, parse_v1_package, parse_v2_package
from .config import PipelineConfig
from .io_utils import atomic_write_bytes, atomic_write_json, canonical_json_bytes, sha256_bytes
from .models import InstanceMask, ParseBundle, instance_numeric_sort_key
from .rle import mask_to_runs
from .scheduler import OrderedTask, ordered_parallel_map


def _mask_digest(instances: list[InstanceMask]) -> str:
    ordered = sorted(instances, key=lambda item: (item.instance_id, item.class_id))
    material = canonical_json_bytes(
        [
            {
                "instance_id": item.instance_id,
                "class_id": item.class_id,
                "run_fingerprint": item.run_fingerprint,
            }
            for item in ordered
        ]
    )
    return sha256_bytes(material)


def _comparison(before: ParseBundle, after: ParseBundle) -> dict[str, Any]:
    before_by_id = {item.instance_id: item for item in before.instances}
    after_by_id = {item.instance_id: item for item in after.instances}
    common_ids = sorted(set(before_by_id).intersection(after_by_id))
    pixel_mismatches: list[str] = []
    class_mismatches: list[str] = []
    for instance_id in common_ids:
        left = before_by_id[instance_id]
        right = after_by_id[instance_id]
        if left.class_id != right.class_id:
            class_mismatches.append(str(instance_id))
        if not np.array_equal(left.mask, right.mask):
            pixel_mismatches.append(str(instance_id))
    return {
        "before_instance_count": len(before.instances),
        "after_instance_count": len(after.instances),
        "instance_count_equal": len(before.instances) == len(after.instances),
        "before_class_ids": sorted({item.class_id for item in before.instances}),
        "after_class_ids": sorted({item.class_id for item in after.instances}),
        "class_ids_equal": {item.class_id for item in before.instances}
        == {item.class_id for item in after.instances},
        "common_instance_ids": common_ids,
        "class_mismatches": class_mismatches,
        "pixel_mismatches": pixel_mismatches,
        "pixel_equal": not pixel_mismatches
        and before_by_id.keys() == after_by_id.keys(),
        "before_mask_digest": _mask_digest(before.instances),
        "after_mask_digest": _mask_digest(after.instances),
    }


def v1_to_v2(
    bundle: ParseBundle,
    output_path: str | Path,
    *,
    config: PipelineConfig,
) -> dict[str, Any]:
    ordered = sorted(bundle.instances, key=instance_numeric_sort_key)

    def encode_instance(item: InstanceMask) -> tuple[dict[str, Any], dict[str, Any] | None]:
        mask_bytes = np.ascontiguousarray(item.mask, dtype=np.uint8).tobytes()
        entry = {
            "instance_id": item.instance_id,
            "class_id": item.class_id,
            "confidence": item.confidence,
            "crowd": item.crowd,
            "palette": list(item.palette),
            "mask_base64": base64.b64encode(mask_bytes).decode("ascii"),
        }
        defaulted = None
        if item.defaulted_fields:
            defaulted = {
                "instance_key": item.key,
                "fields": list(item.defaulted_fields),
                "confidence": item.confidence,
                "crowd": item.crowd,
                "palette": list(item.palette),
            }
        return entry, defaulted

    encoded = ordered_parallel_map(
        [
            OrderedTask(
                instance_id=item.instance_id,
                stage="convert_v1_to_v2",
                payload=item,
                source_format="v1",
            )
            for item in ordered
        ],
        encode_instance,
        config.max_workers,
    )
    entries = [item[0] for item in encoded]
    defaulted_instances = [item[1] for item in encoded if item[1] is not None]
    payload = {
        "format_version": "v2",
        "width": bundle.width,
        "height": bundle.height,
        "palette": {
            str(class_id): list(entry)
            for class_id, entry in sorted(config.palette.items())
        },
        "instances": entries,
    }
    atomic_write_json(output_path, payload)
    reparsed = parse_v2_package(output_path, config=config, max_workers=config.max_workers)
    comparison = _comparison(bundle, reparsed)
    return {
        "direction": "v1_to_v2",
        "output": str(output_path),
        "defaulted_instances": defaulted_instances,
        "degradation_losses": [],
        "comparison": comparison,
        "reparse_failure_codes": sorted({failure.code for failure in reparsed.failures}),
    }


def _serialize_v1(
    bundle: ParseBundle,
    instances: list[InstanceMask],
    *,
    max_workers: int,
) -> bytes:
    ordered = sorted(instances, key=instance_numeric_sort_key)
    serialized_runs = ordered_parallel_map(
        [
            OrderedTask(
                instance_id=item.instance_id,
                stage="convert_v2_to_v1",
                payload=item,
                source_format="v2",
            )
            for item in ordered
        ],
        lambda item: mask_to_runs(item.mask),
        max_workers,
    )
    pieces = [
        V1_HEADER.pack(
            V1_MAGIC,
            V1_VERSION,
            0,
            bundle.width,
            bundle.height,
            len(ordered),
        )
    ]
    for item, runs in zip(ordered, serialized_runs):
        pieces.append(V1_RECORD.pack(item.instance_id, item.class_id, len(runs)))
        pieces.extend(V1_RUN.pack(start, length) for start, length in runs)
    return b"".join(pieces)


def v2_to_v1(
    bundle: ParseBundle,
    output_path: str | Path,
    sidecar_path: str | Path,
    *,
    config: PipelineConfig,
) -> dict[str, Any]:
    payload = _serialize_v1(
        bundle,
        bundle.instances,
        max_workers=config.max_workers,
    )
    atomic_write_bytes(output_path, payload)
    sidecar = {
        "source_format": "v2",
        "target_format": "v1",
        "width": bundle.width,
        "height": bundle.height,
        "instances": {
            item.key: {
                "instance_id": item.instance_id,
                "class_id": item.class_id,
                "confidence": item.confidence,
                "crowd": item.crowd,
                "palette": list(item.palette),
            }
            for item in sorted(bundle.instances, key=instance_numeric_sort_key)
        },
    }
    degradation_losses = [
        {
            "instance_key": item.key,
            "instance_id": item.instance_id,
            "lost_fields": ["confidence", "crowd", "palette"],
            "confidence": item.confidence,
            "crowd": item.crowd,
            "palette": list(item.palette),
            "sidecar": str(sidecar_path),
        }
        for item in sorted(bundle.instances, key=instance_numeric_sort_key)
    ]
    sidecar["degradation_losses"] = degradation_losses
    atomic_write_json(sidecar_path, sidecar)
    reparsed = parse_v1_package(output_path, config=config, max_workers=config.max_workers)
    comparison = _comparison(bundle, reparsed)
    return {
        "direction": "v2_to_v1",
        "output": str(output_path),
        "sidecar": str(sidecar_path),
        "defaulted_instances": [],
        "degradation_losses": degradation_losses,
        "comparison": comparison,
        "reparse_failure_codes": sorted({failure.code for failure in reparsed.failures}),
    }


def compare_bundles(left: ParseBundle, right: ParseBundle) -> dict[str, Any]:
    return _comparison(left, right)
