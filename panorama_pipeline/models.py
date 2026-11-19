from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import Failure
from .io_utils import canonical_json_bytes, sha256_bytes
from .rle import mask_to_runs


PaletteEntry = tuple[int, int, int, int]


def instance_key(source_format: str, instance_id: int) -> str:
    return f"{source_format}:{instance_id}"


def parse_instance_key(value: str) -> tuple[str, int]:
    source_format, raw_id = value.split(":", 1)
    if source_format not in {"v1", "v2"}:
        raise ValueError(f"unsupported source format in instance key: {value}")
    instance_id = int(raw_id)
    if instance_id < 0:
        raise ValueError(f"negative instance id in instance key: {value}")
    return source_format, instance_id


def lexical_instance_id(instance_id: int) -> str:
    return f"{instance_id:012d}"


def instance_numeric_sort_key(item: "InstanceMask") -> tuple:
    return (item.source_format, int(item.instance_id), item.key)


def instance_key_numeric_sort_key(value: str) -> tuple:
    try:
        source_format, instance_id = parse_instance_key(value)
        return (0, source_format, instance_id, value)
    except (TypeError, ValueError):
        prefix = value.split(":", 1)[0]
        return (1, prefix, 0, value)


def metadata_fingerprint(
    *,
    class_id: int,
    confidence: float,
    crowd: bool,
    palette: PaletteEntry,
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "class_id": class_id,
                "confidence": float(confidence),
                "crowd": bool(crowd),
                "palette": list(palette),
            }
        )
    )


def instance_cache_key(
    *,
    source_format: str,
    run_fingerprint: str,
    class_id: int,
    metadata_fp: str,
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "format": source_format,
                "run_fingerprint": run_fingerprint,
                "class_id": class_id,
                "metadata_fingerprint": metadata_fp,
            }
        )
    )


@dataclass
class InstanceMask:
    source_format: str
    instance_id: int
    class_id: int
    mask: np.ndarray
    confidence: float
    crowd: bool
    palette: PaletteEntry
    run_fingerprint: str
    metadata_fingerprint: str
    cache_key: str
    defaulted_fields: tuple[str, ...] = ()
    width: int = 0
    height: int = 0

    @property
    def key(self) -> str:
        return instance_key(self.source_format, self.instance_id)

    @property
    def lexical_id(self) -> str:
        return lexical_instance_id(self.instance_id)

    @property
    def area(self) -> int:
        return int(np.count_nonzero(self.mask))

    @property
    def run_count(self) -> int:
        return len(mask_to_runs(self.mask))

    def with_metadata(
        self,
        *,
        class_id: int | None = None,
        confidence: float | None = None,
        crowd: bool | None = None,
        palette: PaletteEntry | None = None,
        defaulted_fields: tuple[str, ...] | None = None,
    ) -> "InstanceMask":
        new_class = self.class_id if class_id is None else int(class_id)
        new_confidence = self.confidence if confidence is None else float(confidence)
        new_crowd = self.crowd if crowd is None else bool(crowd)
        new_palette = self.palette if palette is None else tuple(palette)
        meta_fp = metadata_fingerprint(
            class_id=new_class,
            confidence=new_confidence,
            crowd=new_crowd,
            palette=new_palette,
        )
        return dataclasses.replace(
            self,
            class_id=new_class,
            confidence=new_confidence,
            crowd=new_crowd,
            palette=new_palette,
            metadata_fingerprint=meta_fp,
            cache_key=instance_cache_key(
                source_format=self.source_format,
                run_fingerprint=self.run_fingerprint,
                class_id=new_class,
                metadata_fp=meta_fp,
            ),
            defaulted_fields=(
                self.defaulted_fields
                if defaulted_fields is None
                else tuple(defaulted_fields)
            ),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "instance_key": self.key,
            "source_format": self.source_format,
            "instance_id": self.instance_id,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "crowd": self.crowd,
            "palette": list(self.palette),
            "area": self.area,
            "run_count": self.run_count,
            "run_fingerprint": self.run_fingerprint,
            "metadata_fingerprint": self.metadata_fingerprint,
            "cache_key": self.cache_key,
            "defaulted_fields": list(self.defaulted_fields),
        }


@dataclass
class ParseBundle:
    source_format: str
    width: int
    height: int
    instances: list[InstanceMask] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)

    def sorted_instances(self) -> list[InstanceMask]:
        return sorted(self.instances, key=instance_numeric_sort_key)

    def sorted_failures(self) -> list[Failure]:
        return sorted(self.failures, key=lambda item: item.stable_sort_key)
