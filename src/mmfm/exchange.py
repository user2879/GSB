"""Loss-aware v1 fixed-width line and v2 JSON intermediate exchange.

v1 intentionally carries only four fixed fields.  Everything else that can be
recovered (step number, references, source digest, original float guidance,
and truncation residual) is stored in a hash-bound sidecar.  A v2 -> v1 -> v2
round trip therefore restores the exact source values when that sidecar is
available, while a sidecar-free v1 document remains self-describing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Iterable

from .canonical import canonical_json, content_digest, sha256_text
from .errors import (
    E_ENUM_INVALID,
    E_ENUM_MISSING,
    E_SIDECAR,
    E_V1_PARSE,
    E_V1_WIDTH,
    E_V2_PARSE,
    fail,
)

V1_FIELDS: tuple[tuple[str, int], ...] = (
    ("frame", 4),
    ("guidance_permille", 5),
    ("noise_code", 2),
    ("quant_bits", 2),
)
V1_HEADER = "# MMFM-V1 fields=frame:4,guidance_permille:5,noise_code:2,quant_bits:2"
V2_SCHEMA = "mmfm-stage-v2"


@dataclass
class StageRecord:
    stage: str
    frame: int | None = None
    step: int | None = None
    guidance_permille: int | None = None
    guidance: float | None = None
    noise_code: int | None = None
    noise_scheme: str | None = None
    quant_bits: int | None = None
    quant_profile: str | None = None
    refs: list[str] = field(default_factory=list)
    digest: str | None = None
    source_version: str = "internal"
    missing_fields: list[str] = field(default_factory=list)
    precision_labels: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def v2_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "frame": self.frame,
            "step": self.step,
            "guidance": self.guidance,
            "noise_scheme": self.noise_scheme,
            "quant_profile": self.quant_profile,
            "refs": list(self.refs),
            "digest": self.digest,
            "source_version": self.source_version,
            "missing_fields": list(self.missing_fields),
            "precision_labels": dict(self.precision_labels),
            "metadata": dict(self.metadata),
        }


@dataclass
class ExchangeDocument:
    stage: str
    records: list[StageRecord]
    provenance: dict[str, Any] = field(default_factory=dict)

    def v2_document(self) -> dict[str, Any]:
        return {
            "schema": V2_SCHEMA,
            "stage": self.stage,
            "provenance": dict(self.provenance),
            "records": [record.v2_dict() for record in self.records],
        }


def _format_fixed(name: str, width: int, value: int | None) -> str:
    if value is None:
        return " " * width
    if isinstance(value, bool) or not isinstance(value, int):
        fail(E_V1_PARSE, "v1 fixed field value must be an integer", field=name, value=value)
    text = str(value)
    if len(text) > width:
        fail(
            E_V1_WIDTH,
            "value does not fit the fixed v1 field width",
            field=name,
            width=width,
            value=value,
        )
    return text.rjust(width, "0") if width <= 6 and value >= 0 else text.rjust(width)


def serialize_v1(
    records: Iterable[StageRecord],
    *,
    stage: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Serialise records to v1 and produce the lossless sidecar."""

    record_list = list(records)
    rows: list[str] = []
    sidecar_records: list[dict[str, Any]] = []
    for index, record in enumerate(record_list):
        if record.frame is not None and record.frame <= 0:
            fail(E_V1_PARSE, "v1 frame numbers must be positive", index=index, frame=record.frame)
        if record.guidance_permille is not None and record.guidance_permille < 0:
            fail(E_V1_PARSE, "v1 guidance permille must be non-negative", index=index)
        row = "|".join(
            [
                _format_fixed("frame", 4, record.frame),
                _format_fixed("guidance_permille", 5, record.guidance_permille),
                _format_fixed("noise_code", 2, record.noise_code),
                _format_fixed("quant_bits", 2, record.quant_bits),
            ]
        )
        rows.append(row)
        sidecar_records.append(
            {
                "index": index,
                "stage": record.stage,
                "step": record.step,
                "original_guidance": record.guidance,
                "rounded_permille": record.guidance_permille,
                "guidance_residual": (
                    None
                    if record.guidance is None or record.guidance_permille is None
                    else record.guidance - record.guidance_permille / 1000.0
                ),
                "precision_labels": dict(record.precision_labels),
                "overridden_references": list(record.refs),
                "digest": record.digest,
                "missing_fields": list(record.missing_fields),
                "metadata": dict(record.metadata),
            }
        )
    stages = {record.stage for record in record_list}
    resolved_stage = stage or (record_list[0].stage if len(stages) == 1 and record_list else "mixed")
    text = "\n".join([V1_HEADER, f"# stage={resolved_stage}", *rows, ""])
    sidecar = {
        "schema": "mmfm-v1-sidecar-v1",
        "v1_sha256": sha256_text(text),
        "stage": resolved_stage,
        "precision_note": "guidance is truncated from v2 float to permille with half-up rounding",
        "records": sidecar_records,
    }
    return text, sidecar


def _parse_fixed_field(name: str, width: int, text: str) -> int | None:
    if len(text) > width:
        fail(
            E_V1_WIDTH,
            "v1 field exceeds its declared fixed width",
            field=name,
            width=width,
            actual=len(text),
        )
    if len(text) < width:
        fail(
            E_V1_PARSE,
            "v1 field is narrower than its declared fixed width",
            field=name,
            width=width,
            actual=len(text),
        )
    if text.strip() == "":
        return None
    if not text.isdigit():
        fail(E_V1_PARSE, "v1 fixed field must contain decimal digits", field=name, value=text)
    return int(text)


def parse_v1(
    text: str,
    *,
    sidecar: dict[str, Any] | None = None,
    noise_code_map: dict[str, int] | None = None,
    profile_bits: dict[str, int] | None = None,
    stage: str | None = None,
) -> ExchangeDocument:
    lines = text.splitlines()
    stage_from_header: str | None = None
    for line in lines:
        if line.startswith("# stage="):
            stage_from_header = line.split("=", 1)[1].strip()
            break
    metadata_by_index: dict[int, dict[str, Any]] = {}
    if sidecar is not None:
        if not isinstance(sidecar, dict) or sidecar.get("schema") != "mmfm-v1-sidecar-v1":
            fail(E_SIDECAR, "unsupported v1 sidecar schema")
        if "v1_sha256" in sidecar and sidecar["v1_sha256"] != sha256_text(text):
            fail(E_SIDECAR, "v1 sidecar hash does not match the fixed-width text")
        sidecar_records = sidecar.get("records", [])
        if not isinstance(sidecar_records, list):
            fail(E_SIDECAR, "v1 sidecar records must be a list")
        for item in sidecar_records:
            if isinstance(item, dict) and isinstance(item.get("index"), int):
                metadata_by_index[item["index"]] = item
        stage_from_header = stage_from_header or str(sidecar.get("stage", "denoise"))
    resolved_stage = stage or stage_from_header or "denoise"
    inverse_noise = {int(code): name for name, code in (noise_code_map or {}).items()}
    inverse_profile = {int(bits): name for name, bits in (profile_bits or {}).items()}
    records: list[StageRecord] = []
    data_index = 0
    for line_number, line in enumerate(lines, start=1):
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) != len(V1_FIELDS):
            fail(
                E_V1_PARSE,
                "v1 line does not contain exactly four fields",
                line=line_number,
                fields=len(parts),
            )
        values = [_parse_fixed_field(name, width, part) for (name, width), part in zip(V1_FIELDS, parts)]
        frame, guidance_permille, noise_code, quant_bits = values
        if frame is not None and frame <= 0:
            fail(E_V1_PARSE, "v1 frame number must be a positive integer", line=line_number, frame=frame)
        meta = metadata_by_index.get(data_index, {})
        missing: list[str] = []
        if frame is None:
            missing.append("frame")
        if guidance_permille is None:
            missing.append("guidance")
        if noise_code is None:
            missing.append("noise_code")
        if quant_bits is None:
            missing.append("quant_bits")
        noise_scheme = inverse_noise.get(noise_code) if noise_code is not None else None
        if noise_code is not None and noise_scheme is None and noise_code_map:
            fail(
                E_ENUM_INVALID,
                "v1 integer noise code is not present in the configured code map",
                noise_code=noise_code,
                allowed_codes=sorted(int(code) for code in noise_code_map.values()),
            )
        quant_profile = inverse_profile.get(quant_bits) if quant_bits is not None else None
        if quant_bits is not None and quant_profile is None and profile_bits:
            fail(
                E_ENUM_INVALID,
                "v1 integer quantisation width is not present in the configured profile map",
                bits=quant_bits,
                allowed_bits=sorted(inverse_profile),
            )
        original_guidance = meta.get("original_guidance") if meta else None
        restored_guidance: int | None = None
        if isinstance(original_guidance, (int, float)) and not isinstance(original_guidance, bool):
            restored_guidance, _ = _round_guidance_to_permille(original_guidance)
        if (
            restored_guidance is None
            or guidance_permille is None
            or restored_guidance != guidance_permille
        ):
            original_guidance = None if guidance_permille is None else guidance_permille / 1000.0
        precision = dict(meta.get("precision_labels", {})) if meta else {}
        if guidance_permille is not None:
            precision.setdefault("guidance_from_v1_permille", "exact_denominator_1000")
        if noise_code is not None:
            precision.setdefault("noise_from_v1_code", "code_table_lookup")
        if quant_bits is not None:
            precision.setdefault("quant_from_v1_bits", "profile_width_lookup")
        missing_from_meta = list(meta.get("missing_fields", [])) if meta else []
        refs = list(meta.get("overridden_references", [])) if meta else []
        records.append(
            StageRecord(
                stage=str(meta.get("stage", resolved_stage)) if meta else resolved_stage,
                frame=frame,
                step=meta.get("step") if meta else None,
                guidance_permille=guidance_permille,
                guidance=float(original_guidance) if original_guidance is not None else None,
                noise_code=noise_code,
                noise_scheme=noise_scheme,
                quant_bits=quant_bits,
                quant_profile=quant_profile,
                refs=refs,
                digest=meta.get("digest") if meta else None,
                source_version="v1",
                missing_fields=sorted(set(missing + missing_from_meta)),
                precision_labels=precision,
                metadata={
                    "sidecar_restored": bool(meta),
                    "original_guidance": meta.get("original_guidance") if meta else None,
                    "guidance_residual": meta.get("guidance_residual") if meta else None,
                    **(dict(meta.get("metadata", {})) if meta else {}),
                },
            )
        )
        data_index += 1
    return ExchangeDocument(
        stage=resolved_stage,
        records=records,
        provenance={
            "source_version": "v1",
            "sidecar_restored": bool(metadata_by_index),
            "sidecar_hash_verified": bool(sidecar and sidecar.get("v1_sha256") == sha256_text(text)),
        },
    )


def v1_to_v2(
    text: str,
    *,
    sidecar: dict[str, Any] | None = None,
    noise_code_map: dict[str, int],
    profile_bits: dict[str, int],
) -> dict[str, Any]:
    return parse_v1(
        text,
        sidecar=sidecar,
        noise_code_map=noise_code_map,
        profile_bits=profile_bits,
    ).v2_document()


def _round_guidance_to_permille(value: Any) -> tuple[int | None, float | None]:
    if value is None:
        return None, None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(E_V2_PARSE, "v2 guidance must be a number", value=value)
    numeric = float(value)
    if numeric < 0:
        fail(E_V2_PARSE, "v2 guidance must be non-negative", value=value)
    death = Decimal(str(value)) * Decimal(1000)
    rounded = int(death.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    residual = numeric - rounded / 1000.0
    return rounded, residual


def validate_v2_document(
    document: Any,
    *,
    allow_missing: bool = False,
    noise_schemes: Any | None = None,
    quant_profiles: Any | None = None,
) -> ExchangeDocument:
    if not isinstance(document, dict):
        fail(E_V2_PARSE, "v2 document must be an object")
    if document.get("schema") != V2_SCHEMA:
        fail(E_V2_PARSE, "unsupported v2 exchange schema", schema=document.get("schema"))
    records_raw = document.get("records")
    if not isinstance(records_raw, list):
        fail(E_V2_PARSE, "v2 records must be a list")
    records: list[StageRecord] = []
    for index, raw in enumerate(records_raw):
        if not isinstance(raw, dict):
            fail(E_V2_PARSE, "v2 record must be an object", index=index)
        stage = raw.get("stage")
        if not isinstance(stage, str) or not stage:
            fail(E_V2_PARSE, "v2 record stage is missing", index=index)
        missing: list[str] = list(raw.get("missing_fields", []))
        for name, code in (("guidance", "guidance"), ("noise_scheme", "noise_scheme"), ("quant_profile", "quant_profile")):
            if name not in raw or raw.get(name) is None:
                missing.append(name)
                if not allow_missing:
                    fail(E_ENUM_MISSING, f"v2 record {name} is missing", index=index, field=name)
        guidance = raw.get("guidance")
        noise_scheme = raw.get("noise_scheme")
        quant_profile = raw.get("quant_profile")
        if not allow_missing:
            if guidance is None:
                fail(E_ENUM_MISSING, "v2 guidance is missing", index=index)
            if not isinstance(noise_scheme, str) or not noise_scheme:
                fail(E_ENUM_MISSING, "v2 noise_scheme is missing", index=index)
            if not isinstance(quant_profile, str) or not quant_profile:
                fail(E_ENUM_MISSING, "v2 quant_profile is missing", index=index)
        if noise_schemes is not None and noise_scheme is not None:
            allowed = set(noise_schemes)
            if noise_scheme not in allowed:
                fail(
                    E_ENUM_INVALID,
                    "v2 noise_scheme is not present in the configured code table",
                    index=index,
                    noise_scheme=noise_scheme,
                    allowed=sorted(allowed),
                )
        if quant_profiles is not None and quant_profile is not None:
            allowed = set(quant_profiles)
            if quant_profile not in allowed:
                fail(
                    E_ENUM_INVALID,
                    "v2 quant_profile is not present in the configured profile enumeration",
                    index=index,
                    quant_profile=quant_profile,
                    allowed=sorted(allowed),
                )
        frame = raw.get("frame")
        if frame is not None and (isinstance(frame, bool) or not isinstance(frame, int) or frame <= 0):
            fail(E_V2_PARSE, "v2 frame must be a positive integer or null", index=index)
        step = raw.get("step")
        if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step <= 0):
            fail(E_V2_PARSE, "v2 step must be a positive integer or null", index=index)
        refs = raw.get("refs", [])
        if not isinstance(refs, list) or any(not isinstance(item, str) for item in refs):
            fail(E_V2_PARSE, "v2 refs must be a list of strings", index=index)
        records.append(
            StageRecord(
                stage=stage,
                frame=frame,
                step=step,
                guidance=guidance,
                noise_scheme=noise_scheme,
                quant_profile=quant_profile,
                refs=refs,
                digest=raw.get("digest"),
                source_version=str(raw.get("source_version", "v2")),
                missing_fields=sorted(set(missing)),
                precision_labels=dict(raw.get("precision_labels", {})),
                metadata=dict(raw.get("metadata", {})),
            )
        )
    return ExchangeDocument(
        stage=str(document.get("stage", "mixed")),
        records=records,
        provenance=dict(document.get("provenance", {})),
    )


def v2_to_v1(
    document: Any,
    *,
    noise_schemes: dict[str, int],
    profiles: dict[str, int],
    allow_missing: bool = True,
) -> tuple[str, dict[str, Any]]:
    parsed = validate_v2_document(document, allow_missing=allow_missing)
    records: list[StageRecord] = []
    residuals: list[dict[str, Any]] = []
    for index, record in enumerate(parsed.records):
        permille, residual = _round_guidance_to_permille(record.guidance)
        if record.noise_scheme is None:
            noise_code = None
        elif record.noise_scheme not in noise_schemes:
            fail(
                E_ENUM_INVALID,
                "v2 noise_scheme is not present in the code table",
                scheme=record.noise_scheme,
                allowed=sorted(noise_schemes),
            )
        else:
            noise_code = int(noise_schemes[record.noise_scheme])
        if record.quant_profile is None:
            quant_bits = None
        elif record.quant_profile not in profiles:
            fail(
                E_ENUM_INVALID,
                "v2 quant_profile is not present in the width table",
                profile=record.quant_profile,
                allowed=sorted(profiles),
            )
        else:
            quant_bits = int(profiles[record.quant_profile])
        precision = dict(record.precision_labels)
        if permille is not None:
            precision["v2_to_v1_guidance_rounding"] = "ROUND_HALF_UP_to_permille"
            precision["v1_denominator"] = 1000
        copied = StageRecord(
            stage=record.stage,
            frame=record.frame,
            step=record.step,
            guidance_permille=permille,
            guidance=record.guidance,
            noise_code=noise_code,
            noise_scheme=record.noise_scheme,
            quant_bits=quant_bits,
            quant_profile=record.quant_profile,
            refs=list(record.refs),
            digest=record.digest,
            source_version="v2",
            missing_fields=list(record.missing_fields),
            precision_labels=precision,
            metadata=dict(record.metadata),
        )
        records.append(copied)
        residuals.append(
            {
                "index": index,
                "original_guidance": record.guidance,
                "rounded_permille": permille,
                "guidance_residual": residual,
                "precision_labels": precision,
                "overridden_references": list(record.refs),
            }
        )
    text, sidecar = serialize_v1(records, stage=parsed.stage)
    sidecar["source_version"] = "v2"
    sidecar["downgrade_residuals"] = residuals
    sidecar["overridden_reference_count"] = sum(len(item.refs) for item in parsed.records)
    # serialize_v1 already hashed the text; adding metadata does not alter it.
    return text, sidecar


def dump_v2(document: Any) -> str:
    return canonical_json(document) + "\n"


def as_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value
