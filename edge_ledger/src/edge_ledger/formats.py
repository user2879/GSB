from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from .canonical import canonical_hash
from .errors import FormatError
from .integrity import compute_shard_proof, parse_shard_seq
from .spec import SpecBundle


V1_SHARD_ID_WIDTH = 32
V1_CONTENT_KEY_WIDTH = 40
V1_DIGEST_WIDTH = 64
V1_LINE_WIDTH = V1_SHARD_ID_WIDTH + V1_CONTENT_KEY_WIDTH + V1_DIGEST_WIDTH
DIGEST_HEX_RE_BYTES = 64

CANONICAL_FIELDS = (
    "source_format",
    "source_line",
    "shard_id",
    "content_key",
    "digest",
    "shard_seq",
    "sample_bytes",
    "node_id",
    "lease_version",
    "quota_version",
    "invalid_ref_id",
    "invalid_ref_shard_id",
    "content_bytes_digest",
    "filled_fields",
    "fill_basis",
    "normalized_hash",
)


@dataclass(frozen=True)
class DecodeContext:
    spec: SpecBundle | None = None
    quota_version: int | None = None
    lease_version: int | None = None
    active_lease_id: str | None = None
    derive_content_digest: bool = True

    @property
    def effective_quota_version(self) -> int:
        if self.quota_version is not None:
            return self.quota_version
        if self.spec is not None:
            return self.spec.quota_version
        return 0

    @property
    def effective_lease_version(self) -> int:
        return self.lease_version if self.lease_version is not None else 0


@dataclass(frozen=True)
class NormalizedReceipt:
    source_format: str
    shard_id: str
    content_key: str
    digest: str
    source_line: int = 1
    shard_seq: int | None = None
    sample_bytes: int | None = None
    node_id: str | None = None
    lease_version: int = 0
    quota_version: int = 0
    invalid_ref_id: str | None = None
    invalid_ref_shard_id: str | None = None
    content_bytes_digest: str | None = None
    filled_fields: tuple[str, ...] = ()
    fill_basis: tuple[tuple[str, str], ...] = ()
    normalized_hash: str = ""

    def to_canonical(self) -> dict[str, Any]:
        return {
            "source_format": self.source_format,
            "source_line": self.source_line,
            "shard_id": self.shard_id,
            "content_key": self.content_key,
            "digest": self.digest,
            "shard_seq": self.shard_seq,
            "sample_bytes": self.sample_bytes,
            "node_id": self.node_id,
            "lease_version": self.lease_version,
            "quota_version": self.quota_version,
            "invalid_ref_id": self.invalid_ref_id,
            "invalid_ref_shard_id": self.invalid_ref_shard_id,
            "content_bytes_digest": self.content_bytes_digest,
            "filled_fields": list(self.filled_fields),
            "fill_basis": [list(item) for item in self.fill_basis],
        }

    def to_ordered_tuple(self) -> tuple[Any, ...]:
        data = self.to_canonical()
        return tuple(canonical_hash(data[key]) if key in ("filled_fields", "fill_basis") else data[key] for key in CANONICAL_FIELDS if key != "normalized_hash")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def _fixed_field(line: str, start: int, width: int, field: str, lineno: int) -> str:
    end = start + width
    if len(line) < end:
        raise FormatError(f"v1 field {field} is shorter than fixed width {width}", field, lineno)
    raw = line[start:end]
    value = raw.rstrip(" ")
    if not value:
        raise FormatError(f"v1 field {field} is empty", field, lineno)
    if raw != value.ljust(width):
        raise FormatError(f"v1 field {field} has invalid fixed-width padding", field, lineno)
    if len(value) > width:
        raise FormatError(f"v1 field {field} exceeds width {width}", field, lineno)
    return value


def encode_v1(shard_id: str, content_key: str, digest: str) -> str:
    if len(shard_id) > V1_SHARD_ID_WIDTH:
        raise FormatError("shard_id exceeds v1 width", "shard_id")
    if len(content_key) > V1_CONTENT_KEY_WIDTH:
        raise FormatError("content_key exceeds v1 width", "content_key")
    if not _is_hex64(digest):
        raise FormatError("digest must be 64 lowercase hex characters", "digest")
    return shard_id.ljust(V1_SHARD_ID_WIDTH) + content_key.ljust(V1_CONTENT_KEY_WIDTH) + digest.ljust(V1_DIGEST_WIDTH)


def _check_common_strings(obj: dict[str, Any], lineno: int) -> tuple[str, str, str]:
    for key in ("shard_id", "content_key", "digest"):
        if key not in obj or not isinstance(obj[key], str) or not obj[key]:
            raise FormatError(f"{key} must be a non-empty string", key, lineno)
    shard_id = obj["shard_id"]
    content_key = obj["content_key"]
    digest = obj["digest"]
    if len(shard_id) > V1_SHARD_ID_WIDTH:
        raise FormatError("shard_id exceeds fixed compatibility width", "shard_id", lineno)
    if len(content_key) > V1_CONTENT_KEY_WIDTH:
        raise FormatError("content_key exceeds fixed compatibility width", "content_key", lineno)
    if not _is_hex64(digest):
        raise FormatError("digest must be 64 lowercase hex characters", "digest", lineno)
    return shard_id, content_key, digest


def _consistency(
    shard_id: str,
    content_key: str,
    shard_seq: int | None,
    sample_bytes: int | None,
    context: DecodeContext | None,
    lineno: int,
) -> tuple[int, int | None, Any | None]:
    try:
        derived_seq = parse_shard_seq(shard_id)
    except ValueError as exc:
        raise FormatError(str(exc), "shard_id", lineno) from exc
    if shard_seq is not None and shard_seq != derived_seq:
        raise FormatError(
            f"shard_seq {shard_seq} does not match shard_id suffix {derived_seq}",
            "shard_seq",
            lineno,
        )
    seq = derived_seq if shard_seq is None else shard_seq
    if context is None or context.spec is None:
        return seq, sample_bytes, None
    try:
        content = context.spec.content_for(content_key)
    except Exception as exc:
        raise FormatError(f"content_key {content_key!r} is absent from manifest", "content_key", lineno) from exc
    if not (0 <= seq < content.shard_count):
        raise FormatError(f"shard_seq {seq} is outside manifest range", "shard_seq", lineno)
    expected_id = content.shard_id(seq)
    if shard_id != expected_id:
        raise FormatError(f"shard_id must be {expected_id!r}", "shard_id", lineno)
    expected_bytes = content.shard_size
    if sample_bytes is not None and sample_bytes != expected_bytes:
        raise FormatError(
            f"sample_bytes {sample_bytes} does not match manifest shard byte count {expected_bytes}",
            "sample_bytes",
            lineno,
        )
    return seq, expected_bytes, content


def _derive_content_digest(
    context: DecodeContext | None,
    node_id: str,
    content: Any | None,
    seq: int,
) -> tuple[str | None, str]:
    if context is None or context.spec is None or content is None or not context.derive_content_digest:
        return None, "default:null"
    try:
        node = context.spec.node(node_id)
    except Exception:
        return None, "default:null(unknown_node)"
    proof = compute_shard_proof(node, content, seq)
    return proof.content_bytes_digest, "derived:integrity.compute_shard_proof"


def _make_normalized(
    *,
    source_format: str,
    source_line: int,
    shard_id: str,
    content_key: str,
    digest: str,
    shard_seq: int,
    sample_bytes: int | None,
    node_id: str,
    lease_version: int,
    quota_version: int,
    invalid_ref_id: str | None,
    invalid_ref_shard_id: str | None,
    content_bytes_digest: str | None,
    basis: dict[str, str],
) -> NormalizedReceipt:
    filled = tuple(sorted(basis))
    fill_basis = tuple(sorted(basis.items()))
    provisional = NormalizedReceipt(
        source_format=source_format,
        source_line=source_line,
        shard_id=shard_id,
        content_key=content_key,
        digest=digest,
        shard_seq=shard_seq,
        sample_bytes=sample_bytes,
        node_id=node_id,
        lease_version=lease_version,
        quota_version=quota_version,
        invalid_ref_id=invalid_ref_id,
        invalid_ref_shard_id=invalid_ref_shard_id,
        content_bytes_digest=content_bytes_digest,
        filled_fields=filled,
        fill_basis=fill_basis,
    )
    return NormalizedReceipt(**{**provisional.__dict__, "normalized_hash": canonical_hash(provisional.to_canonical())})


def decode_v1(line: str, lineno: int = 1, context: DecodeContext | None = None) -> NormalizedReceipt:
    if len(line) != V1_LINE_WIDTH:
        if len(line) < V1_SHARD_ID_WIDTH:
            field = "shard_id"
        elif len(line) < V1_SHARD_ID_WIDTH + V1_CONTENT_KEY_WIDTH:
            field = "content_key"
        elif len(line) <= V1_LINE_WIDTH:
            field = "digest"
        else:
            field = "line"
        raise FormatError(
            f"v1 line must be exactly {V1_LINE_WIDTH} characters, got {len(line)}",
            field,
            lineno,
        )
    shard_id = _fixed_field(line, 0, V1_SHARD_ID_WIDTH, "shard_id", lineno)
    content_key = _fixed_field(line, V1_SHARD_ID_WIDTH, V1_CONTENT_KEY_WIDTH, "content_key", lineno)
    digest = _fixed_field(
        line,
        V1_SHARD_ID_WIDTH + V1_CONTENT_KEY_WIDTH,
        V1_DIGEST_WIDTH,
        "digest",
        lineno,
    )
    seq, manifest_sample_bytes, content = _consistency(shard_id, content_key, None, None, context, lineno)
    sample_bytes = manifest_sample_bytes
    node_id = context.spec.policy.legacy_node_id if context and context.spec else "legacy-v1"
    quota_version = context.effective_quota_version if context else 0
    lease_version = context.effective_lease_version if context else 0
    content_digest, digest_basis = _derive_content_digest(context, node_id, content, seq)
    basis = {
        "shard_seq": "derived:shard_id_suffix",
        "sample_bytes": "manifest:ContentSpec.shard_size",
        "node_id": "default:lease_policy.legacy_node_id",
        "lease_version": "active_lease:expected_version_or_zero",
        "quota_version": "context:effective_quota_version",
        "invalid_ref_id": "default:null",
        "invalid_ref_shard_id": "default:null",
        "content_bytes_digest": digest_basis,
    }
    return _make_normalized(
        source_format="v1",
        source_line=lineno,
        shard_id=shard_id,
        content_key=content_key,
        digest=digest,
        shard_seq=seq,
        sample_bytes=sample_bytes,
        node_id=node_id,
        lease_version=lease_version,
        quota_version=quota_version,
        invalid_ref_id=None,
        invalid_ref_shard_id=None,
        content_bytes_digest=content_digest,
        basis=basis,
    )


def _validate_invalid_ref(
    value: Any,
    content: Any | None,
    lineno: int,
) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    if isinstance(value, str):
        ref_id = value
        shard_id = None
    elif isinstance(value, dict):
        ref_id = value.get("ref_id") or value.get("id")
        shard_id = value.get("shard_id")
    else:
        raise FormatError("invalid_shard_ref must be null, a ref id, or an object", "invalid_shard_ref", lineno)
    if not isinstance(ref_id, str) or not ref_id:
        raise FormatError("invalid_shard_ref.ref_id must be a non-empty string", "invalid_shard_ref.ref_id", lineno)
    if content is not None:
        matches = [ref for ref in content.invalid_refs if ref.ref_id == ref_id]
        if not matches:
            raise FormatError(
                f"invalid_shard_ref {ref_id!r} is not declared in manifest",
                "invalid_shard_ref.ref_id",
                lineno,
            )
        declared = matches[0]
        if shard_id is not None and shard_id != declared.shard_id:
            raise FormatError(
                "invalid_shard_ref target does not match declared reference",
                "invalid_shard_ref.shard_id",
                lineno,
            )
        shard_id = declared.shard_id
    elif shard_id is not None:
        try:
            parse_shard_seq(shard_id)
        except ValueError as exc:
            raise FormatError("invalid_shard_ref.shard_id is dangling", "invalid_shard_ref.shard_id", lineno) from exc
    else:
        raise FormatError("invalid_shard_ref is missing resolvable shard_id", "invalid_shard_ref.shard_id", lineno)
    return ref_id, shard_id


def decode_v2(obj: dict[str, Any], lineno: int = 1, context: DecodeContext | None = None) -> NormalizedReceipt:
    shard_id, content_key, digest = _check_common_strings(obj, lineno)
    if "shard_seq" not in obj or not _is_int(obj["shard_seq"]) or obj["shard_seq"] < 0:
        raise FormatError("shard_seq must be a non-negative integer", "shard_seq", lineno)
    if "sample_bytes" not in obj or not _is_int(obj["sample_bytes"]) or obj["sample_bytes"] < 0:
        raise FormatError("sample_bytes must be a non-negative integer", "sample_bytes", lineno)
    if "node_id" not in obj or not isinstance(obj["node_id"], str) or not obj["node_id"]:
        raise FormatError("node_id must be a non-empty string", "node_id", lineno)
    node_id = obj["node_id"]
    if context and context.spec:
        try:
            context.spec.node(node_id)
        except Exception as exc:
            raise FormatError(f"node_id {node_id!r} is not declared", "node_id", lineno) from exc
    seq, expected_bytes, content = _consistency(
        shard_id, content_key, obj["shard_seq"], obj["sample_bytes"], context, lineno
    )
    sample_bytes = expected_bytes if expected_bytes is not None else obj["sample_bytes"]
    quota_version = context.effective_quota_version if context else 0
    lease_version = context.effective_lease_version if context else 0
    content_digest, digest_basis = _derive_content_digest(context, node_id, content, seq)
    basis = {
        "lease_version": "active_lease:expected_version_or_zero",
        "quota_version": "context:effective_quota_version",
        "invalid_ref_id": "default:null",
        "invalid_ref_shard_id": "default:null",
        "content_bytes_digest": digest_basis,
    }
    return _make_normalized(
        source_format="v2",
        source_line=lineno,
        shard_id=shard_id,
        content_key=content_key,
        digest=digest,
        shard_seq=seq,
        sample_bytes=sample_bytes,
        node_id=node_id,
        lease_version=lease_version,
        quota_version=quota_version,
        invalid_ref_id=None,
        invalid_ref_shard_id=None,
        content_bytes_digest=content_digest,
        basis=basis,
    )


def decode_v3(obj: dict[str, Any], lineno: int = 1, context: DecodeContext | None = None) -> NormalizedReceipt:
    shard_id, content_key, digest = _check_common_strings(obj, lineno)
    for key in ("shard_seq", "sample_bytes", "lease_version", "quota_version"):
        if key not in obj or not _is_int(obj[key]) or obj[key] < 0:
            raise FormatError(f"{key} must be a non-negative integer", key, lineno)
    if "node_id" not in obj or not isinstance(obj["node_id"], str) or not obj["node_id"]:
        raise FormatError("node_id must be a non-empty string", "node_id", lineno)
    if "content_bytes_digest" not in obj or not _is_hex64(obj["content_bytes_digest"]):
        raise FormatError("content_bytes_digest must be 64 lowercase hex characters", "content_bytes_digest", lineno)
    node_id = obj["node_id"]
    if context and context.spec:
        try:
            context.spec.node(node_id)
        except Exception as exc:
            raise FormatError(f"node_id {node_id!r} is not declared", "node_id", lineno) from exc
    seq, expected_bytes, content = _consistency(
        shard_id, content_key, obj["shard_seq"], obj["sample_bytes"], context, lineno
    )
    sample_bytes = expected_bytes if expected_bytes is not None else obj["sample_bytes"]
    ref_id, ref_shard_id = _validate_invalid_ref(obj.get("invalid_shard_ref"), content, lineno)
    basis: dict[str, str] = {}
    if "invalid_shard_ref" not in obj:
        basis["invalid_ref_id"] = "v3:field_absent_default_null"
        basis["invalid_ref_shard_id"] = "v3:field_absent_default_null"
    return _make_normalized(
        source_format="v3",
        source_line=lineno,
        shard_id=shard_id,
        content_key=content_key,
        digest=digest,
        shard_seq=seq,
        sample_bytes=sample_bytes,
        node_id=node_id,
        lease_version=int(obj["lease_version"]),
        quota_version=int(obj["quota_version"]),
        invalid_ref_id=ref_id,
        invalid_ref_shard_id=ref_shard_id,
        content_bytes_digest=obj["content_bytes_digest"],
        basis=basis,
    )


def decode_receipt_line(
    line: str,
    lineno: int = 1,
    context: DecodeContext | None = None,
    fmt: str | None = None,
) -> NormalizedReceipt:
    if not line:
        raise FormatError("receipt line is empty", "line", lineno)
    stripped = line.lstrip()
    if fmt == "v1" or (fmt is None and not stripped.startswith("{")):
        return decode_v1(line.rstrip("\r\n"), lineno, context)
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        raise FormatError(f"invalid JSON receipt: {exc.msg}", "$", lineno) from exc
    if not isinstance(obj, dict):
        raise FormatError("JSON receipt must be an object", "$", lineno)
    version = fmt or obj.get("format") or obj.get("receipt_format")
    if version == "v2":
        return decode_v2(obj, lineno, context)
    if version == "v3":
        return decode_v3(obj, lineno, context)
    raise FormatError(f"unsupported receipt format {version!r}", "format", lineno)


def decode_receipts(
    lines: Iterable[str],
    context: DecodeContext | None = None,
    fmt: str | None = None,
) -> list[NormalizedReceipt]:
    result: list[NormalizedReceipt] = []
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        result.append(decode_receipt_line(line, lineno, context, fmt))
    return result


def encode_v2(proof_like: Any, *, lease_version: int = 0) -> str:
    obj = {
        "format": "v2",
        "shard_id": proof_like.shard_id,
        "content_key": proof_like.content_key,
        "digest": proof_like.digest,
        "shard_seq": proof_like.shard_seq,
        "sample_bytes": proof_like.sample_bytes,
        "node_id": proof_like.node_id,
    }
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def encode_v3(proof_like: Any, *, lease_version: int, quota_version: int, invalid_ref: dict | None = None) -> str:
    obj = {
        "format": "v3",
        "shard_id": proof_like.shard_id,
        "content_key": proof_like.content_key,
        "digest": proof_like.digest,
        "shard_seq": proof_like.shard_seq,
        "sample_bytes": proof_like.sample_bytes,
        "node_id": proof_like.node_id,
        "lease_version": lease_version,
        "quota_version": quota_version,
        "invalid_shard_ref": invalid_ref,
        "content_bytes_digest": proof_like.content_bytes_digest,
    }
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def verify_normalized_receipt(receipt: NormalizedReceipt, spec: SpecBundle) -> tuple[bool, str]:
    try:
        node = spec.node(receipt.node_id or "")
    except Exception:
        return False, "rejected:unknown_node"
    try:
        content = spec.content_for(receipt.content_key)
    except Exception:
        return False, "rejected:unknown_content"
    if receipt.shard_seq is None:
        return False, "rejected:missing_shard_seq"
    proof = compute_shard_proof(node, content, receipt.shard_seq)
    if receipt.sample_bytes != proof.sample_bytes:
        return False, "rejected:sample_bytes_mismatch"
    if receipt.content_bytes_digest is not None and receipt.content_bytes_digest != proof.content_bytes_digest:
        return False, "rejected:content_digest_mismatch"
    if receipt.digest != proof.digest:
        return False, "rejected:digest_mismatch"
    return True, "adopted:exact_proof_match"
