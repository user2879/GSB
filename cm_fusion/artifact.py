from __future__ import annotations

import json
import os
import shutil
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .compat import CompatTable, VersionSpec
from .config import Config
from .errors import PipelineError
from .postings import FIELDS, LogicalIndex, Posting, PostingBlock, logical_index_from_postings, logical_postings_digest
from .util import (
    atomic_write_bytes,
    canonical_json,
    read_json,
    sha256_bytes,
    sha256_json,
)
from .varint import decode_uvarint, encode_uvarint


FLAG_POSITIONS = 1 << 0
FLAG_LOWFREQ = 1 << 1
FLAG_SKIPLIST = 1 << 2
FLAG_TIER2 = 1 << 3
SKIPLIST_STEP = 16


@dataclass(frozen=True)
class EncodedBlock:
    block: PostingBlock
    term_ref: int
    flags: int
    data: bytes
    core: bytes
    checksum: int
    skiplist: tuple[tuple[int, int, int], ...]
    lane: str
    block_sha256: str
    reuse_source: str | None = None

    @property
    def key(self) -> str:
        return self.block.key


@dataclass(frozen=True)
class EncodedIndex:
    meta: dict
    sidecar: dict
    blocks: tuple[EncodedBlock, ...]
    reused_keys: tuple[str, ...]
    rebuilt_keys: tuple[str, ...]
    logical_digest: str
    fingerprint: str

    def block_path(self, block: EncodedBlock) -> str:
        key_hash = sha256_bytes(block.key.encode("utf-8"))[:24]
        lane_dir = "lowfreq" if block.lane == "lowfreq" else "main"
        return f"blocks/{lane_dir}/{key_hash}.bin"


def encode_index(
    logical: LogicalIndex,
    config: Config,
    previous_root: str | Path | None = None,
    previous_meta: dict | None = None,
    format_version: str | None = None,
    tier_id: int | None = None,
    keep_positions: bool | None = None,
    checkpoint: Callable[[], None] | None = None,
) -> EncodedIndex:
    target_format = format_version or config.index_format_version
    target_tier = config.active_tier if tier_id is None else tier_id
    target_positions = config.keep_positions if keep_positions is None else keep_positions
    spec = config.compat.require_version(target_format)
    tier = config.tier(target_tier)
    blocks = logical.blocks(config, tier_id=target_tier)

    previous_blocks: dict[str, dict] = {}
    if previous_meta is not None and previous_root is not None:
        if (
            previous_meta.get("format_version") == target_format
            and previous_meta.get("compression_tier") == target_tier
            and bool(previous_meta.get("keep_positions")) == target_positions
        ):
            previous_blocks = {item["key"]: item for item in previous_meta.get("blocks", [])}

    encoded_blocks: list[EncodedBlock] = []
    sidecar_blocks: dict[str, dict] = {}
    loss_features: dict[str, list[str]] = {}
    reused: list[str] = []
    rebuilt: list[str] = []
    doc_count = len(logical.doc_ordinals)
    for block in blocks:
        if checkpoint is not None:
            checkpoint()
        data, core, checksum, skiplist, flags = _encode_block_data(
            block, logical, target_format, target_tier, target_positions, spec
        )
        block_sha = sha256_bytes(data)
        previous = previous_blocks.get(block.key)
        reuse_source = None
        if previous is not None and previous.get("block_sha256") == block_sha:
            reuse_source = str(Path(previous_root) / previous["file"]) if previous_root is not None else None
            if reuse_source is not None and Path(reuse_source).is_file():
                reused.append(block.key)
            else:
                reuse_source = None
                rebuilt.append(block.key)
        else:
            rebuilt.append(block.key)

        sidecar_item: dict = {}
        if not spec.supports_block_checksum:
            sidecar_item["block_checksum"] = checksum
            loss_features.setdefault("block_checksum", []).append(block.key)
        if target_positions and not spec.supports_position_skiplist:
            sidecar_item["position_skiplist"] = [list(item) for item in skiplist]
            loss_features.setdefault("position_skiplist", []).append(block.key)
        if sidecar_item:
            sidecar_blocks[block.key] = sidecar_item

        encoded_blocks.append(
            EncodedBlock(
                block=block,
                term_ref=logical.term_refs[f"{block.field}\x1f{block.term}"],
                flags=flags,
                data=data,
                core=core,
                checksum=checksum,
                skiplist=skiplist,
                lane=block.lane,
                block_sha256=block_sha,
                reuse_source=reuse_source,
            )
        )

    sidecar = {
        "schema_version": 1,
        "format_version": target_format,
        "blocks": {key: sidecar_blocks[key] for key in sorted(sidecar_blocks)},
    }
    sidecar_digest = sha256_json(sidecar)
    losses = []
    for feature, keys in sorted(loss_features.items()):
        losses.append(
            {
                "feature": feature,
                "target_version": target_format,
                "strategy": "sidecar",
                "reason": f"{target_format} cannot express {feature} in its block header/payload",
                "block_count": len(keys),
                "block_keys": sorted(keys),
            }
        )

    doc_table = [
        {"ordinal": str(ordinal), "doc_id": doc_id}
        for doc_id, ordinal in sorted(logical.doc_ordinals.items(), key=lambda item: (item[1], item[0]))
    ]
    term_table = [
        {"ref": ref, "field": key.split("\x1f", 1)[0], "term": key.split("\x1f", 1)[1]}
        for key, ref in sorted(logical.term_refs.items(), key=lambda item: item[1])
    ]
    block_summaries = []
    for encoded in sorted(encoded_blocks, key=lambda item: (item.block.field, item.block.term, item.block.block_index)):
        block_summaries.append(
            {
                "key": encoded.key,
                "field": encoded.block.field,
                "term": encoded.block.term,
                "term_ref": encoded.term_ref,
                "block_index": encoded.block.block_index,
                "doc_count": len(encoded.block.postings),
                "tf_sum": sum(posting.tf for posting in encoded.block.postings),
                "position_count": sum(len(posting.positions) if posting.positions is not None else 0 for posting in encoded.block.postings),
                "lane": encoded.lane,
                "flags": encoded.flags,
                "has_positions": bool(encoded.flags & FLAG_POSITIONS),
                "has_skiplist": bool(encoded.flags & FLAG_SKIPLIST),
                "block_sha256": encoded.block_sha256,
                "block_checksum": encoded.checksum,
                "file": _block_relative_path(encoded),
            }
        )

    fingerprint_payload = {
        "schema": "cm-fusion-index-v1",
        "format_version": target_format,
        "compression_tier": target_tier,
        "tier_semantic": config.compat.tier_semantic(target_format, target_tier),
        "keep_positions": target_positions,
        "block_size": config.posting_block_size,
        "index_config_hash": config.index_hash(),
        "logical_digest": logical.logical_digest(),
        "blocks": block_summaries,
        "sidecar_digest": sidecar_digest,
        "degradation_losses": losses,
    }
    fingerprint = sha256_json(fingerprint_payload)
    meta = {
        "schema_version": 1,
        "format_version": target_format,
        "header_kind": spec.header_kind,
        "compression_tier": target_tier,
        "tier_semantic": config.compat.tier_semantic(target_format, target_tier),
        "keep_positions": target_positions,
        "block_size": config.posting_block_size,
        "skiplist_step": SKIPLIST_STEP,
        "document_count": len(logical.documents),
        "posting_count": logical.posting_count(),
        "block_count": len(encoded_blocks),
        "doc_table": doc_table,
        "term_table": term_table,
        "blocks": block_summaries,
        "sidecar_file": "sidecar.json",
        "sidecar_digest": sidecar_digest,
        "logical_digest": logical.logical_digest(),
        "index_fingerprint": fingerprint,
        "degradation_losses": losses,
        "reuse_report": {
            "reused_block_count": len(reused),
            "rebuilt_block_count": len(rebuilt),
            "reused_block_keys": sorted(reused),
            "rebuilt_block_keys": sorted(rebuilt),
        },
    }
    return EncodedIndex(
        meta=meta,
        sidecar=sidecar,
        blocks=tuple(encoded_blocks),
        reused_keys=tuple(sorted(reused)),
        rebuilt_keys=tuple(sorted(rebuilt)),
        logical_digest=logical.logical_digest(),
        fingerprint=fingerprint,
    )


def _encode_block_data(
    block: PostingBlock,
    logical: LogicalIndex,
    format_version: str,
    tier_id: int,
    keep_positions: bool,
    spec: VersionSpec,
) -> tuple[bytes, bytes, int, tuple[tuple[int, int, int], ...], int]:
    postings = sorted(block.postings, key=lambda posting: logical.doc_ordinals[posting.doc_id])
    flags = 0
    if keep_positions:
        flags |= FLAG_POSITIONS
    if block.lane == "lowfreq":
        flags |= FLAG_LOWFREQ
    if tier_id == 2:
        flags |= FLAG_TIER2
    if spec.supports_position_skiplist and keep_positions:
        flags |= FLAG_SKIPLIST

    out = bytearray()
    out.extend(encode_uvarint(len(postings)))
    previous_ordinal = 0
    for posting in postings:
        ordinal = logical.doc_ordinals[posting.doc_id]
        out.extend(encode_uvarint(ordinal - previous_ordinal))
        previous_ordinal = ordinal
        out.extend(encode_uvarint(posting.tf))
        if keep_positions:
            positions = posting.positions or ()
            out.extend(encode_uvarint(len(positions)))
            previous_position = 0
            for position in positions:
                out.extend(encode_uvarint(position - previous_position))
                previous_position = position
    core = bytes(out)
    checksum = zlib.crc32(core) & 0xFFFFFFFF
    skiplist = _build_skiplist(postings) if keep_positions else ()

    payload = bytearray(core)
    if flags & FLAG_SKIPLIST:
        payload.extend(encode_uvarint(len(skiplist)))
        for posting_index, position_index, position in skiplist:
            payload.extend(encode_uvarint(posting_index))
            payload.extend(encode_uvarint(position_index))
            payload.extend(encode_uvarint(position))
    payload_bytes = bytes(payload)

    term_ref = logical.term_refs[f"{block.field}\x1f{block.term}"]
    if spec.header_kind == "fixed":
        layout = _fixed_header_layout(spec)
        values = {
            "term_ref": term_ref,
            "doc_count": len(postings),
            "flags": flags,
            "tier": tier_id,
            "payload_length": len(payload_bytes),
            "checksum": checksum,
        }
        for name, _ in _V1_FIELDS:
            field = spec.header_fields.get(name)
            width = field.get("width") if isinstance(field, dict) else field
            if int(values[name]) >= 1 << (8 * int(width)):
                raise PipelineError(
                    "E_INDEX_INVALID",
                    f"value for {name} does not fit declared {width}-byte fixed header field",
                )
        header = struct.pack(layout, *(values[name] for name, _ in _V1_FIELDS))
    elif spec.header_kind == "varint":
        header = b"".join(
            encode_uvarint(value)
            for value in (term_ref, len(postings), flags, tier_id, len(payload_bytes))
        )
    else:
        raise PipelineError("E_INDEX_INVALID", f"unsupported header kind: {spec.header_kind}")
    return header + payload_bytes, core, checksum, skiplist, flags


def _build_skiplist(postings: list[Posting]) -> tuple[tuple[int, int, int], ...]:
    items: list[tuple[int, int, int]] = []
    for posting_index, posting in enumerate(postings):
        for position_index, position in enumerate(posting.positions or ()):
            if position_index % SKIPLIST_STEP == 0:
                items.append((posting_index, position_index, position))
    return tuple(items)


_V1_FIELDS = (
    ("term_ref", "I"),
    ("doc_count", "H"),
    ("flags", "B"),
    ("tier", "B"),
    ("payload_length", "I"),
    ("checksum", "I"),
)

_FIXED_WIDTH_FORMAT = {1: "B", 2: "H", 4: "I", 8: "Q"}


def _fixed_header_layout(spec: VersionSpec) -> str:
    chars = []
    for name, _ in _V1_FIELDS:
        field = spec.header_fields.get(name)
        if isinstance(field, dict):
            width = field.get("width")
        else:
            width = field
        char = _FIXED_WIDTH_FORMAT.get(width)
        if char is None:
            raise PipelineError(
                "E_INDEX_INVALID",
                f"compatibility table width for {spec.name}.{name} must be one of 1, 2, 4, 8 bytes",
            )
        chars.append(char)
    return ">" + "".join(chars)


def _block_relative_path(block: EncodedBlock) -> str:
    key_hash = sha256_bytes(block.key.encode("utf-8"))[:24]
    lane_dir = "lowfreq" if block.lane == "lowfreq" else "main"
    return f"blocks/{lane_dir}/{key_hash}.bin"


def write_index(
    encoded: EncodedIndex,
    root: str | Path,
    checkpoint: Callable[[], None] | None = None,
) -> None:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    for block in encoded.blocks:
        if checkpoint is not None:
            checkpoint()
        relative = _block_relative_path(block)
        target = root_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if block.reuse_source is not None and Path(block.reuse_source).is_file():
            try:
                os.link(block.reuse_source, target)
            except OSError:
                shutil.copyfile(block.reuse_source, target)
        else:
            atomic_write_bytes(target, block.data)
    atomic_write_bytes(root_path / "sidecar.json", (canonical_json(encoded.sidecar) + "\n").encode("utf-8"))
    atomic_write_bytes(root_path / "index.json", (json.dumps(encoded.meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))


@dataclass(frozen=True)
class DecodedIndex:
    postings: dict[str, dict[str, tuple[Posting, ...]]]
    doc_table: dict[int, str]
    term_table: dict[int, tuple[str, str]]
    meta: dict
    sidecar: dict
    block_count: int
    decoded_block_summaries: tuple[dict, ...]


def load_meta(root: str | Path) -> dict:
    try:
        return read_json(Path(root) / "index.json")
    except (OSError, ValueError) as exc:
        raise PipelineError("E_INDEX_INVALID", f"cannot load index metadata: {exc}") from exc


def decode_index(root: str | Path, meta: dict | None = None) -> DecodedIndex:
    root_path = Path(root)
    if meta is None:
        meta = load_meta(root_path)
    try:
        sidecar = read_json(root_path / meta.get("sidecar_file", "sidecar.json"))
    except (OSError, ValueError) as exc:
        raise PipelineError("E_INDEX_INVALID", f"cannot load sidecar metadata: {exc}") from exc
    if sha256_json(sidecar) != meta.get("sidecar_digest"):
        raise PipelineError("E_INDEX_INVALID", "sidecar digest does not match index metadata")
    doc_table = {int(item["ordinal"]): item["doc_id"] for item in meta["doc_table"]}
    term_table = {int(item["ref"]): (item["field"], item["term"]) for item in meta["term_table"]}
    combined: dict[str, dict[str, list[Posting]]] = {field: {} for field in FIELDS}
    decoded_summaries = []
    seen_blocks: dict[tuple[str, str], set[int]] = {}
    if len(meta["blocks"]) != int(meta.get("block_count", len(meta["blocks"]))):
        raise PipelineError("E_INDEX_INVALID", "block count does not match index metadata")

    for summary in meta["blocks"]:
        path = root_path / summary["file"]
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise PipelineError("E_INDEX_INVALID", f"cannot read block {summary['key']}: {exc}") from exc
        header, core, payload, flags, tier, checksum, skiplist = _decode_block_bytes(data, meta)
        if sha256_bytes(data) != summary["block_sha256"]:
            raise PipelineError("E_INDEX_INVALID", f"block digest mismatch: {summary['key']}")
        if tier != meta["compression_tier"]:
            raise PipelineError("E_INDEX_INVALID", f"block tier mismatch: {summary['key']}")
        sidecar_item = sidecar.get("blocks", {}).get(summary["key"], {})
        if "block_checksum" in sidecar_item and int(sidecar_item["block_checksum"]) != checksum:
            raise PipelineError("E_INDEX_INVALID", f"sidecar checksum mismatch: {summary['key']}")
        field, term = term_table.get(header["term_ref"], ("", ""))
        if field != summary["field"] or term != summary["term"]:
            raise PipelineError("E_INDEX_INVALID", f"term reference mismatch: {summary['key']}")
        decoded_postings = _decode_core(
            core,
            doc_table,
            bool(flags & FLAG_POSITIONS),
            header["doc_count"],
            summary["key"],
        )
        expected_skiplist = _build_skiplist(decoded_postings) if flags & FLAG_POSITIONS else ()
        if bool(flags & FLAG_SKIPLIST) and skiplist != expected_skiplist:
            raise PipelineError("E_INDEX_INVALID", f"payload skiplist mismatch: {summary['key']}")
        if "position_skiplist" in sidecar_item:
            expected = [list(item) for item in expected_skiplist]
            if sidecar_item["position_skiplist"] != expected:
                raise PipelineError("E_INDEX_INVALID", f"sidecar skiplist mismatch: {summary['key']}")
        combined[field].setdefault(term, []).extend(decoded_postings)
        block_index = int(summary["block_index"])
        seen_blocks.setdefault((field, term), set()).add(block_index)
        decoded_summaries.append(
            {
                "key": summary["key"],
                "doc_count": len(decoded_postings),
                "has_positions": bool(flags & FLAG_POSITIONS),
                "skiplist_entries": len(skiplist),
            }
        )

    frozen: dict[str, dict[str, tuple[Posting, ...]]] = {}
    for field in FIELDS:
        frozen[field] = {}
        for term, postings in sorted(combined[field].items()):
            expected_count = len(seen_blocks.get((field, term), set()))
            max_index = max(seen_blocks.get((field, term), {-1}))
            if expected_count != max_index + 1:
                raise PipelineError("E_INDEX_INVALID", f"non-contiguous blocks for {field}:{term}")
            frozen[field][term] = tuple(sorted(postings, key=lambda posting: posting.doc_id))
    decoded_logical_digest = logical_postings_digest(frozen, bool(meta.get("keep_positions")))
    if decoded_logical_digest != meta.get("logical_digest"):
        raise PipelineError("E_INDEX_INVALID", "decoded logical posting digest does not match index metadata")
    return DecodedIndex(
        postings=frozen,
        doc_table=doc_table,
        term_table=term_table,
        meta=meta,
        sidecar=sidecar,
        block_count=len(meta["blocks"]),
        decoded_block_summaries=tuple(decoded_summaries),
    )


def _decode_block_bytes(
    data: bytes,
    meta: dict,
) -> tuple[dict, bytes, bytes, int, int, int, tuple[tuple[int, int, int], ...]]:
    compat_spec = meta.get("_compat_spec")
    if compat_spec is None:
        compat_spec = _spec_from_meta(meta)
    if compat_spec.header_kind == "fixed":
        layout = _fixed_header_layout(compat_spec)
        header_size = struct.calcsize(layout)
        if len(data) < header_size:
            raise PipelineError("E_INDEX_INVALID", "truncated fixed block header")
        values = struct.unpack(layout, data[:header_size])
        header = {name: value for (name, _), value in zip(_V1_FIELDS, values)}
        payload = data[header_size:]
        if len(payload) != header["payload_length"]:
            raise PipelineError("E_INDEX_INVALID", "fixed block payload length mismatch")
        header["checksum"]
    elif compat_spec.header_kind == "varint":
        offset = 0
        values = []
        for _ in range(5):
            value, offset = decode_uvarint(data, offset)
            values.append(value)
        header = {
            "term_ref": values[0],
            "doc_count": values[1],
            "flags": values[2],
            "tier": values[3],
            "payload_length": values[4],
            "checksum": 0,
        }
        payload = data[offset:]
        if len(payload) != header["payload_length"]:
            raise PipelineError("E_INDEX_INVALID", "varint block payload length mismatch")
    else:
        raise PipelineError("E_INDEX_INVALID", "unsupported block header kind")

    offset = 0
    count, offset = decode_uvarint(payload, offset)
    if count != header["doc_count"]:
        raise PipelineError("E_INDEX_INVALID", "block posting count mismatch")
    has_positions = bool(header["flags"] & FLAG_POSITIONS)
    for _ in range(count):
        _, offset = decode_uvarint(payload, offset)
        _, offset = decode_uvarint(payload, offset)
        if has_positions:
            position_count, offset = decode_uvarint(payload, offset)
            for _ in range(position_count):
                _, offset = decode_uvarint(payload, offset)
    core = payload[:offset]
    checksum = zlib.crc32(core) & 0xFFFFFFFF
    if compat_spec.header_kind == "fixed" and checksum != header["checksum"]:
        raise PipelineError("E_INDEX_INVALID", "v1 block checksum mismatch")

    skiplist: list[tuple[int, int, int]] = []
    if header["flags"] & FLAG_SKIPLIST:
        skip_count, offset = decode_uvarint(payload, offset)
        for _ in range(skip_count):
            posting_index, offset = decode_uvarint(payload, offset)
            position_index, offset = decode_uvarint(payload, offset)
            position, offset = decode_uvarint(payload, offset)
            skiplist.append((posting_index, position_index, position))
    if offset != len(payload):
        raise PipelineError("E_INDEX_INVALID", "trailing bytes in block payload")
    return header, core, payload, header["flags"], header["tier"], checksum, tuple(skiplist)


def _spec_from_meta(meta: dict):
    header_fields = meta.get("header_fields")
    if not isinstance(header_fields, dict):
        if meta.get("format_version") == "v1":
            header_fields = {
                "term_ref": {"width": 4},
                "doc_count": {"width": 2},
                "flags": {"width": 1},
                "tier": {"width": 1},
                "payload_length": {"width": 4},
                "checksum": {"width": 4},
            }
        else:
            header_fields = {}
    return VersionSpec(
        name=meta.get("format_version", "unknown"),
        header_kind=meta.get("header_kind", "varint"),
        header_fields=header_fields,
        tier_semantics={int(meta.get("compression_tier", 1)): meta.get("tier_semantic", "delta_varint")},
        supported_tiers=(int(meta.get("compression_tier", 1)),),
        supports_block_checksum=True,
        supports_position_skiplist=bool(meta.get("keep_positions")),
        compression_encoding="delta_varint",
    )


def decode_with_compat(root: str | Path, compat: CompatTable, meta: dict | None = None) -> DecodedIndex:
    root_path = Path(root)
    if meta is None:
        meta = load_meta(root_path)
    enriched = dict(meta)
    enriched["_compat_spec"] = compat.require_version(meta["format_version"])
    return decode_index(root_path, enriched)


def decode_core_postings(
    core: bytes,
    doc_table: dict[int, str],
    has_positions: bool,
    count: int,
    key: str,
) -> list[Posting]:
    return _decode_core(core, doc_table, has_positions, count, key)


def _decode_core(
    core: bytes,
    doc_table: dict[int, str],
    has_positions: bool,
    count: int,
    key: str,
) -> list[Posting]:
    offset = 0
    actual_count, offset = decode_uvarint(core, offset)
    if actual_count != count:
        raise PipelineError("E_INDEX_INVALID", f"posting count mismatch in {key}")
    previous_ordinal = 0
    postings: list[Posting] = []
    for _ in range(count):
        delta, offset = decode_uvarint(core, offset)
        ordinal = previous_ordinal + delta
        previous_ordinal = ordinal
        if ordinal not in doc_table:
            raise PipelineError("E_INDEX_INVALID", f"unknown document ordinal in {key}")
        tf, offset = decode_uvarint(core, offset)
        positions: tuple[int, ...] | None = None
        if has_positions:
            position_count, offset = decode_uvarint(core, offset)
            previous_position = 0
            values = []
            for _ in range(position_count):
                delta_position, offset = decode_uvarint(core, offset)
                previous_position += delta_position
                values.append(previous_position)
            positions = tuple(values)
            if position_count != tf:
                raise PipelineError("E_INDEX_INVALID", f"tf/position count mismatch in {key}")
        postings.append(Posting(doc_table[ordinal], tf, positions))
    if offset != len(core):
        raise PipelineError("E_INDEX_INVALID", f"trailing core bytes in {key}")
    return postings


def convert_index(
    source_root: str | Path,
    config: Config,
    target_format: str,
    previous_meta: dict,
    target_tier: int | None = None,
    keep_positions: bool | None = None,
) -> EncodedIndex:
    decoded = decode_with_compat(source_root, config.compat)
    doc_ids = sorted(decoded.doc_table.values())
    from .loaders import Document

    dummy_documents = [
        Document(doc_id, (config.controlled_vocabulary[0],), (config.controlled_vocabulary[0],), "__conversion__", 0)
        for doc_id in doc_ids
    ]
    target_positions = bool(previous_meta.get("keep_positions")) if keep_positions is None else keep_positions
    logical = logical_index_from_postings(dummy_documents, decoded.postings, target_positions)
    return encode_index(
        logical,
        config,
        previous_root=source_root,
        previous_meta=previous_meta,
        format_version=target_format,
        tier_id=target_tier,
        keep_positions=target_positions,
    )
