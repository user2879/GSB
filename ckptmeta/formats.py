from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import load_json, pretty_json_bytes, atomic_write_bytes
from .config import step_id
from .errors import fail
from .view import canonical_view


CURRENT_VERSION = 3
DIGEST_RE = re.compile(r"^[0-9a-fA-F]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
STATE_RE = re.compile(r"^[0-9a-fA-F]{32}$")

V1_TOP_FIELDS = {"format_version", "global_step", "shard_count", "shards"}
V2_TOP_FIELDS = V1_TOP_FIELDS | {"rng_state"}
V3_TOP_FIELDS = V2_TOP_FIELDS | {"manifest_id", "partition_mode", "parent_manifest"}
V1_SHARD_FIELDS = {"shard_index", "parameter_digest"}
V2_SHARD_FIELDS = V1_SHARD_FIELDS | {"momentum_digest"}
V3_SHARD_FIELDS = V2_SHARD_FIELDS | {"data_cursor"}
RNG_FIELDS = {"algorithm", "seed", "state", "inc", "has_uint32", "uinteger", "draw_step"}
CURSOR_FIELDS = {"epoch", "offset", "next_sample_index"}
PARENT_FIELDS = {"manifest_id", "digest"}


@dataclass(frozen=True)
class ShardRecord:
    index: int
    parameter_digest: str
    momentum_digest: str | None = None
    data_cursor: dict[str, Any] | None = None

    def to_dict(self, version: int) -> dict[str, Any]:
        result: dict[str, Any] = {
            "shard_index": self.index,
            "parameter_digest": self.parameter_digest,
        }
        if version >= 2:
            result["momentum_digest"] = self.momentum_digest
        if version >= 3:
            result["data_cursor"] = dict(self.data_cursor or {})
        return result


@dataclass(frozen=True)
class Manifest:
    version: int
    global_step: int
    shard_count: int
    shards: tuple[ShardRecord, ...]
    source_path: Path | None = None
    manifest_id: str | None = None
    partition_mode: str | None = None
    rng_state: dict[str, Any] | None = None
    parent_manifest: dict[str, Any] | None = None
    parent_field_present: bool = False

    def derived_id(self) -> str:
        return self.manifest_id or step_id(self.global_step)

    def to_view(self) -> dict[str, Any]:
        if self.version != CURRENT_VERSION:
            fail(
                "E_FORMAT",
                "a complete metadata view requires v3",
                path="format_version",
                manifest_path=str(self.source_path) if self.source_path else None,
            )
        return canonical_view(
            {
                "manifest_id": self.derived_id(),
                "global_step": self.global_step,
                "shard_count": self.shard_count,
                "partition_mode": self.partition_mode,
                "shards": [shard.to_dict(3) for shard in self.shards],
                "rng_state": self.rng_state,
                "parent_manifest": self.parent_manifest,
            }
        )

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "format_version": self.version,
            "global_step": self.global_step,
            "shard_count": self.shard_count,
            "shards": [shard.to_dict(self.version) for shard in self.shards],
        }
        if self.version >= 2:
            record["rng_state"] = dict(self.rng_state or {})
        if self.version >= 3:
            record["manifest_id"] = self.derived_id()
            record["partition_mode"] = self.partition_mode
            record["parent_manifest"] = (
                None if self.parent_manifest is None else dict(self.parent_manifest)
            )
        return record


def _format_error(message: str, path: str, *, source_path: Path | None = None, **details: Any) -> None:
    fail(
        "E_FORMAT",
        message,
        path=path,
        manifest_path=str(source_path) if source_path else None,
        details={"field": path, **details},
    )


def _check_fields(
    value: Any,
    allowed: set[str],
    *,
    prefix: str,
    source_path: Path | None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        _format_error("expected a JSON object", prefix, source_path=source_path)
    unknown = sorted(set(value) - allowed)
    if unknown:
        path = f"{prefix}.{unknown[0]}" if prefix else unknown[0]
        _format_error(f"unknown field {unknown[0]}", path, source_path=source_path)
    missing = sorted(allowed - set(value))
    if missing:
        path = f"{prefix}.{missing[0]}" if prefix else missing[0]
        _format_error(f"missing field {missing[0]}", path, source_path=source_path)
    return value


def _int_field(
    value: Any,
    *,
    path: str,
    source_path: Path | None,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _format_error("must be an integer", path, source_path=source_path)
    if value < minimum or (maximum is not None and value > maximum):
        _format_error("integer is out of range", path, source_path=source_path)
    return value


def _digest_field(value: Any, *, path: str, source_path: Path | None) -> str:
    if not isinstance(value, str) or DIGEST_RE.fullmatch(value) is None:
        _format_error("must be a 64-character hexadecimal digest", path, source_path=source_path)
    return value.lower()


def _validate_rng(value: Any, *, source_path: Path | None) -> dict[str, Any]:
    raw = _check_fields(value, RNG_FIELDS, prefix="rng_state", source_path=source_path)
    if raw["algorithm"] != "PCG64":
        _format_error("unsupported rng algorithm", "rng_state.algorithm", source_path=source_path)
    seed = _int_field(raw["seed"], path="rng_state.seed", source_path=source_path, maximum=2**63 - 1)
    state = raw["state"]
    inc = raw["inc"]
    if not isinstance(state, str) or STATE_RE.fullmatch(state) is None:
        _format_error("must be a 32-character hexadecimal state", "rng_state.state", source_path=source_path)
    if not isinstance(inc, str) or STATE_RE.fullmatch(inc) is None:
        _format_error("must be a 32-character hexadecimal increment", "rng_state.inc", source_path=source_path)
    has_uint32 = _int_field(raw["has_uint32"], path="rng_state.has_uint32", source_path=source_path, maximum=1)
    uinteger = _int_field(
        raw["uinteger"], path="rng_state.uinteger", source_path=source_path, maximum=2**32 - 1
    )
    draw_step = _int_field(raw["draw_step"], path="rng_state.draw_step", source_path=source_path)
    return {
        "algorithm": "PCG64",
        "seed": seed,
        "state": state.lower(),
        "inc": inc.lower(),
        "has_uint32": has_uint32,
        "uinteger": uinteger,
        "draw_step": draw_step,
    }


def _validate_cursor(value: Any, *, path: str, source_path: Path | None) -> dict[str, Any]:
    raw = _check_fields(value, CURSOR_FIELDS, prefix=path, source_path=source_path)
    return {
        "epoch": _int_field(raw["epoch"], path=f"{path}.epoch", source_path=source_path),
        "offset": _int_field(raw["offset"], path=f"{path}.offset", source_path=source_path),
        "next_sample_index": _int_field(
            raw["next_sample_index"], path=f"{path}.next_sample_index", source_path=source_path
        ),
    }


def _validate_parent(
    value: Any,
    *,
    global_step: int,
    source_path: Path | None,
) -> dict[str, Any] | None:
    if value is None:
        if global_step == 0:
            return None
        fail(
            "E_CHAIN",
            "non-genesis v3 manifests require a parent_manifest reference",
            path="parent_manifest",
            manifest_path=str(source_path) if source_path else None,
            details={"field": "parent_manifest", "global_step": global_step},
        )
    raw = _check_fields(value, PARENT_FIELDS, prefix="parent_manifest", source_path=source_path)
    manifest_id = raw["manifest_id"]
    if not isinstance(manifest_id, str) or not manifest_id or ID_RE.fullmatch(manifest_id) is None:
        _format_error("parent manifest_id must be a non-empty identifier", "parent_manifest.manifest_id", source_path=source_path)
    return {
        "manifest_id": manifest_id,
        "digest": _digest_field(raw["digest"], path="parent_manifest.digest", source_path=source_path),
    }


def decode_manifest_data(raw: Any, *, source_path: Path | None = None) -> Manifest:
    version = raw.get("format_version") if isinstance(raw, dict) else None
    if isinstance(version, bool) or not isinstance(version, int) or version not in (1, 2, 3):
        _format_error(
            "unknown or missing manifest format_version",
            "format_version",
            source_path=source_path,
            value=version,
        )

    allowed_top = {1: V1_TOP_FIELDS, 2: V2_TOP_FIELDS, 3: V3_TOP_FIELDS}[version]
    record = _check_fields(raw, allowed_top, prefix="", source_path=source_path)
    global_step = _int_field(
        record["global_step"], path="global_step", source_path=source_path, maximum=1_000_000_000
    )
    shard_count = _int_field(
        record["shard_count"], path="shard_count", source_path=source_path, minimum=1, maximum=256
    )

    raw_shards = record["shards"]
    if not isinstance(raw_shards, list):
        _format_error("shards must be an array", "shards", source_path=source_path)
    if len(raw_shards) != shard_count:
        _format_error(
            "shard_count does not match the number of shard records",
            "shards",
            source_path=source_path,
            shard_count=shard_count,
            observed=len(raw_shards),
        )

    allowed_shard = {1: V1_SHARD_FIELDS, 2: V2_SHARD_FIELDS, 3: V3_SHARD_FIELDS}[version]
    shards: list[ShardRecord] = []
    for index, raw_shard in enumerate(raw_shards):
        prefix = f"shards[{index}]"
        shard = _check_fields(raw_shard, allowed_shard, prefix=prefix, source_path=source_path)
        shard_index = _int_field(
            shard["shard_index"], path=f"{prefix}.shard_index", source_path=source_path
        )
        if shard_index != index:
            _format_error(
                "shard records must be sorted and indexed 0..shard_count-1",
                f"{prefix}.shard_index",
                source_path=source_path,
                expected=index,
                actual=shard_index,
            )
        parameter_digest = _digest_field(
            shard["parameter_digest"], path=f"{prefix}.parameter_digest", source_path=source_path
        )
        momentum_digest = None
        data_cursor = None
        if version >= 2:
            momentum_digest = _digest_field(
                shard["momentum_digest"], path=f"{prefix}.momentum_digest", source_path=source_path
            )
        if version >= 3:
            data_cursor = _validate_cursor(
                shard["data_cursor"], path=f"{prefix}.data_cursor", source_path=source_path
            )
        shards.append(
            ShardRecord(
                index=shard_index,
                parameter_digest=parameter_digest,
                momentum_digest=momentum_digest,
                data_cursor=data_cursor,
            )
        )

    manifest_id = None
    partition_mode = None
    rng_state = None
    parent_manifest = None
    parent_field_present = False
    if version >= 2:
        rng_state = _validate_rng(record["rng_state"], source_path=source_path)
    if version >= 3:
        manifest_id = record["manifest_id"]
        expected_id = step_id(global_step)
        if not isinstance(manifest_id, str) or manifest_id != expected_id:
            _format_error(
                "manifest_id must match global_step",
                "manifest_id",
                source_path=source_path,
                expected=expected_id,
            )
        partition_mode = record["partition_mode"]
        if partition_mode not in ("modulo", "contiguous"):
            _format_error(
                "partition_mode must be modulo or contiguous",
                "partition_mode",
                source_path=source_path,
            )
        parent_field_present = True
        parent_manifest = _validate_parent(
            record["parent_manifest"], global_step=global_step, source_path=source_path
        )

    # Deliberately construct only after every field has validated: no partial loads.
    return Manifest(
        version=version,
        global_step=global_step,
        shard_count=shard_count,
        shards=tuple(shards),
        source_path=source_path,
        manifest_id=manifest_id,
        partition_mode=partition_mode,
        rng_state=rng_state,
        parent_manifest=parent_manifest,
        parent_field_present=parent_field_present,
    )


def decode_manifest(path: str | Path) -> Manifest:
    source_path = Path(path)
    raw = load_json(source_path, code="E_FORMAT")
    if not isinstance(raw, dict):
        _format_error("manifest root must be a JSON object", "$", source_path=source_path)
    return decode_manifest_data(raw, source_path=source_path)


def encode_manifest(manifest: Manifest) -> bytes:
    return pretty_json_bytes(manifest.to_record())


def write_manifest(path: str | Path, manifest: Manifest) -> None:
    atomic_write_bytes(Path(path), encode_manifest(manifest))
