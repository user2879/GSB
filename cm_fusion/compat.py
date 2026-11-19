from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ValidationError
from .util import canonical_json, read_json, sha256_file, sha256_json


@dataclass(frozen=True)
class VersionSpec:
    name: str
    header_kind: str
    header_fields: dict[str, Any]
    tier_semantics: dict[int, str]
    supported_tiers: tuple[int, ...]
    supports_block_checksum: bool
    supports_position_skiplist: bool
    compression_encoding: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "header_kind": self.header_kind,
            "header_fields": self.header_fields,
            "tier_semantics": {str(k): v for k, v in sorted(self.tier_semantics.items())},
            "supported_tiers": list(self.supported_tiers),
            "supports_block_checksum": self.supports_block_checksum,
            "supports_position_skiplist": self.supports_position_skiplist,
            "compression_encoding": self.compression_encoding,
        }


@dataclass(frozen=True)
class CompatTable:
    raw: dict
    file_hash: str
    semantic_hash: str
    versions: dict[str, VersionSpec]
    query_stream_versions: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> "CompatTable":
        try:
            raw = read_json(path)
        except (OSError, ValueError) as exc:
            raise ValidationError("E_INDEX_INVALID", f"cannot load compatibility table: {exc}") from exc
        return cls.from_mapping(raw, file_hash=sha256_file(path))

    @classmethod
    def from_mapping(cls, raw: dict, file_hash: str | None = None) -> "CompatTable":
        if not isinstance(raw, dict):
            raise ValidationError("E_INDEX_INVALID", "compatibility table must be an object")
        versions_raw = raw.get("artifact_versions")
        if not isinstance(versions_raw, dict) or not versions_raw:
            raise ValidationError("E_INDEX_INVALID", "compatibility table needs artifact_versions")

        versions: dict[str, VersionSpec] = {}
        for name, item in versions_raw.items():
            if not isinstance(item, dict):
                raise ValidationError("E_INDEX_INVALID", f"artifact version {name} must be an object")
            header = item.get("header", {})
            if not isinstance(header, dict):
                raise ValidationError("E_INDEX_INVALID", f"artifact version {name} header must be an object")
            header_kind = header.get("kind")
            header_fields = header.get("fields", {})
            if header_kind not in {"fixed", "varint"} or not isinstance(header_fields, dict):
                raise ValidationError("E_INDEX_INVALID", f"artifact version {name} has an invalid header")
            for field_name, field_spec in header_fields.items():
                if not isinstance(field_spec, dict):
                    raise ValidationError("E_INDEX_INVALID", f"artifact version {name} header field {field_name} must be an object")
                if header_kind == "fixed":
                    if field_spec.get("encoding") != "uint" or field_spec.get("width") not in (1, 2, 4, 8):
                        raise ValidationError(
                            "E_INDEX_INVALID",
                            f"fixed header field {name}.{field_name} needs uint encoding and width 1/2/4/8",
                        )
                elif field_spec.get("encoding") != "uvarint":
                    raise ValidationError(
                        "E_INDEX_INVALID",
                        f"varint header field {name}.{field_name} needs uvarint encoding",
                    )
            semantics_raw = item.get("tier_semantics", {})
            if not isinstance(semantics_raw, dict):
                raise ValidationError("E_INDEX_INVALID", f"artifact version {name} tier_semantics must be an object")
            semantics: dict[int, str] = {}
            for tier, semantic in semantics_raw.items():
                try:
                    tier_id = int(tier)
                except (TypeError, ValueError) as exc:
                    raise ValidationError("E_INDEX_INVALID", f"invalid tier key in {name}: {tier}") from exc
                if tier_id <= 0 or not isinstance(semantic, str) or not semantic:
                    raise ValidationError("E_INDEX_INVALID", f"invalid tier semantic in {name}")
                semantics[tier_id] = semantic
            supported = item.get("supported_tiers")
            if supported is None:
                supported = sorted(semantics)
            if not isinstance(supported, list) or not supported or any(not isinstance(t, int) or t <= 0 for t in supported):
                raise ValidationError("E_INDEX_INVALID", f"invalid supported_tiers in {name}")
            features = item.get("features", {})
            if not isinstance(features, dict):
                raise ValidationError("E_INDEX_INVALID", f"version {name} features must be an object")
            spec = VersionSpec(
                name=name,
                header_kind=header_kind,
                header_fields=header_fields,
                tier_semantics=semantics,
                supported_tiers=tuple(sorted(set(supported))),
                supports_block_checksum=bool(features.get("block_checksum", False)),
                supports_position_skiplist=bool(features.get("position_skiplist", False)),
                compression_encoding=str(item.get("compression_encoding", "delta_varint")),
            )
            if any(tier not in spec.tier_semantics for tier in spec.supported_tiers):
                raise ValidationError("E_INDEX_INVALID", f"version {name} supports a tier without semantics")
            versions[name] = spec

        stream_versions = raw.get("query_stream_versions")
        if not isinstance(stream_versions, list) or not stream_versions:
            raise ValidationError("E_INDEX_INVALID", "query_stream_versions must be a non-empty array")
        if any(not isinstance(v, str) or not v for v in stream_versions):
            raise ValidationError("E_INDEX_INVALID", "query stream versions must be non-empty strings")
        if len(stream_versions) != len(set(stream_versions)):
            raise ValidationError("E_INDEX_INVALID", "query stream versions must be unique")

        semantic_payload = {
            "artifact_versions": {name: spec.to_dict() for name, spec in sorted(versions.items())},
            "query_stream_versions": sorted(stream_versions),
            "tier_mapping": raw.get("tier_mapping", {}),
        }
        return cls(
            raw=raw,
            file_hash=file_hash or sha256_json(raw),
            semantic_hash=sha256_json(semantic_payload),
            versions=versions,
            query_stream_versions=tuple(stream_versions),
        )

    def require_version(self, name: str) -> VersionSpec:
        try:
            return self.versions[name]
        except KeyError as exc:
            raise ValidationError("E_INDEX_INVALID", f"undeclared index format version: {name}") from exc

    def require_query_stream(self, name: str) -> None:
        if name not in self.query_stream_versions:
            raise ValidationError("E_INDEX_INVALID", f"undeclared query stream version: {name}")

    def tier_semantic(self, version: str, tier: int) -> str:
        spec = self.require_version(version)
        if tier not in spec.supported_tiers or tier not in spec.tier_semantics:
            raise ValidationError("E_INDEX_INVALID", f"compression tier {tier} is not supported by {version}")
        return spec.tier_semantics[tier]

    def to_dict(self) -> dict:
        return self.raw


def compat_hash_of_raw(raw: dict) -> str:
    return sha256_json(raw)


def compat_semantic_payload(table: CompatTable) -> str:
    return canonical_json({name: spec.to_dict() for name, spec in sorted(table.versions.items())})
