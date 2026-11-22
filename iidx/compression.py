from __future__ import annotations

from typing import Any

from .util import first_difference, sha256_json

MAGIC = b"IIDX"
FORMAT_VERSION = 1


class DecodeError(ValueError):
    def __init__(self, message: str, offset: int | None = None):
        super().__init__(message)
        self.offset = offset


def encode_uvarint(value: int) -> bytes:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("uvarint value must be a non-negative integer")
    output = bytearray()
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def read_uvarint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    start = offset
    while True:
        if offset >= len(data):
            raise DecodeError("truncated variable-length integer", start)
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift > 63:
            raise DecodeError("variable-length integer is too large", start)


def _write_length_bytes(output: bytearray, raw: bytes) -> None:
    output.extend(encode_uvarint(len(raw)))
    output.extend(raw)


def _read_length_bytes(data: bytes, offset: int) -> tuple[bytes, int]:
    length, offset = read_uvarint(data, offset)
    end = offset + length
    if end > len(data):
        raise DecodeError("truncated length-prefixed payload", offset)
    return data[offset:end], end


def block_fingerprint(terms: list[dict[str, Any]]) -> str:
    return sha256_json({"terms": terms})


def decoded_checksum(terms: list[dict[str, Any]]) -> str:
    return sha256_json({"terms": terms})


def _normalize_terms_input(block: dict[str, Any]) -> list[dict[str, Any]]:
    terms: list[dict[str, Any]] = []
    for term_entry in sorted(block.get("terms", []), key=lambda item: item["term"]):
        term = str(term_entry["term"])
        postings = sorted(term_entry.get("postings", []), key=lambda item: str(item["doc_id"]))
        normalized_postings = []
        previous_doc_id: str | None = None
        for posting in postings:
            doc_id = str(posting["doc_id"])
            if doc_id == previous_doc_id:
                raise ValueError(f"duplicate posting doc_id {doc_id!r} for term {term!r}")
            previous_doc_id = doc_id
            positions = sorted(int(position) for position in posting.get("positions", []))
            if any(position < 0 for position in positions):
                raise ValueError("positions must be non-negative")
            if len(positions) != len(set(positions)):
                raise ValueError(f"duplicate positions for term {term!r} and doc {doc_id!r}")
            normalized_postings.append({"doc_id": doc_id, "positions": positions})
        terms.append(
            {
                "term": term,
                "doc_freq": len(normalized_postings),
                "postings": normalized_postings,
            }
        )
    return terms


def _encode_term_record(output: bytearray, term_entry: dict[str, Any], doc_ordinals: dict[str, int]) -> None:
    _write_length_bytes(output, str(term_entry["term"]).encode("utf-8"))
    postings = term_entry["postings"]
    output.extend(encode_uvarint(len(postings)))
    previous_ordinal = -1
    for posting in postings:
        ordinal = doc_ordinals[str(posting["doc_id"])]
        delta = ordinal - previous_ordinal
        if delta <= 0:
            raise ValueError("document ordinals must be strictly increasing")
        output.extend(encode_uvarint(delta))
        previous_ordinal = ordinal
        positions = posting["positions"]
        output.extend(encode_uvarint(len(positions)))
        previous_position = -1
        for position in positions:
            position_delta = int(position) - previous_position
            if position_delta <= 0:
                raise ValueError("positions must be strictly increasing")
            output.extend(encode_uvarint(position_delta))
            previous_position = int(position)


def _decode_term_record(
    data: bytes,
    offset: int,
    doc_ids: list[str],
) -> tuple[dict[str, Any], int]:
    raw_term, offset = _read_length_bytes(data, offset)
    try:
        term = raw_term.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DecodeError("term is not valid UTF-8", offset) from exc
    if not term:
        raise DecodeError("empty term in encoded block", offset)
    posting_count, offset = read_uvarint(data, offset)
    postings: list[dict[str, Any]] = []
    previous_ordinal = -1
    for _ in range(posting_count):
        delta, offset = read_uvarint(data, offset)
        if delta <= 0:
            raise DecodeError("document ordinal delta must be positive", offset)
        ordinal = previous_ordinal + delta
        if ordinal >= len(doc_ids):
            raise DecodeError("document ordinal is outside the dictionary", offset)
        previous_ordinal = ordinal
        position_count, offset = read_uvarint(data, offset)
        positions: list[int] = []
        previous_position = -1
        for _ in range(position_count):
            position_delta, offset = read_uvarint(data, offset)
            if position_delta <= 0:
                raise DecodeError("position delta must be positive", offset)
            position = previous_position + position_delta
            previous_position = position
            positions.append(position)
        postings.append({"doc_id": doc_ids[ordinal], "positions": positions})
    return {"term": term, "doc_freq": len(postings), "postings": postings}, offset


def encode_block(
    block: dict[str, Any],
    tier: int,
    low_frequency_threshold: int = 1,
) -> tuple[bytes, dict[str, Any]]:
    if tier not in (1, 2):
        raise ValueError("tier must be 1 or 2")
    if low_frequency_threshold < 0:
        raise ValueError("low_frequency_threshold must be non-negative")
    terms = _normalize_terms_input(block)
    fingerprint = block.get("fingerprint") or block_fingerprint(terms)
    if len(fingerprint) != 64:
        raise ValueError("block fingerprint must be a sha256 hex digest")
    if tier == 1:
        retained_terms = terms
        filtered_terms: list[dict[str, Any]] = []
    else:
        retained_terms = [entry for entry in terms if entry["doc_freq"] > low_frequency_threshold]
        filtered_terms = [entry for entry in terms if entry["doc_freq"] <= low_frequency_threshold]

    all_doc_ids = sorted(
        {
            str(posting["doc_id"])
            for term_entry in terms
            for posting in term_entry["postings"]
        }
    )
    doc_ordinals = {doc_id: index for index, doc_id in enumerate(all_doc_ids)}

    output = bytearray()
    output.extend(MAGIC)
    output.append(FORMAT_VERSION)
    output.append(tier)
    output.extend(bytes.fromhex(fingerprint))
    output.extend(encode_uvarint(len(terms)))
    output.extend(encode_uvarint(len(retained_terms)))
    output.extend(encode_uvarint(len(filtered_terms)))
    if tier == 2:
        output.extend(encode_uvarint(low_frequency_threshold))
    output.extend(encode_uvarint(len(all_doc_ids)))
    for doc_id in all_doc_ids:
        _write_length_bytes(output, doc_id.encode("utf-8"))
    for term_entry in retained_terms:
        _encode_term_record(output, term_entry, doc_ordinals)
    for term_entry in filtered_terms:
        _encode_term_record(output, term_entry, doc_ordinals)

    stats = {
        "tier": tier,
        "format_version": FORMAT_VERSION,
        "block_fingerprint": fingerprint,
        "total_terms": len(terms),
        "retained_terms": len(retained_terms),
        "filtered_terms": len(filtered_terms),
        "filtered_term_names": [entry["term"] for entry in filtered_terms],
        "low_frequency_threshold": low_frequency_threshold if tier == 2 else None,
        "doc_count": len(all_doc_ids),
        "posting_count": sum(entry["doc_freq"] for entry in terms),
        "position_count": sum(
            len(posting["positions"])
            for entry in terms
            for posting in entry["postings"]
        ),
        "encoded_bytes": len(output),
        "decoded_checksum": decoded_checksum(terms),
    }
    return bytes(output), stats


def decode_block(data: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    offset = 0
    if data[:4] != MAGIC:
        raise DecodeError("invalid block magic", 0)
    offset = 4
    if offset >= len(data):
        raise DecodeError("missing format version", offset)
    version = data[offset]
    offset += 1
    if version != FORMAT_VERSION:
        raise DecodeError(f"unsupported format version: {version}", offset - 1)
    if offset >= len(data):
        raise DecodeError("missing compression tier", offset)
    tier = data[offset]
    offset += 1
    if tier not in (1, 2):
        raise DecodeError(f"unsupported compression tier: {tier}", offset - 1)
    fingerprint_end = offset + 32
    if fingerprint_end > len(data):
        raise DecodeError("truncated block fingerprint", offset)
    fingerprint = data[offset:fingerprint_end].hex()
    offset = fingerprint_end
    total_terms, offset = read_uvarint(data, offset)
    retained_count, offset = read_uvarint(data, offset)
    filtered_count, offset = read_uvarint(data, offset)
    if retained_count + filtered_count != total_terms:
        raise DecodeError("term counts are inconsistent", offset)
    low_frequency_threshold: int | None = None
    if tier == 2:
        low_frequency_threshold, offset = read_uvarint(data, offset)
    doc_count, offset = read_uvarint(data, offset)
    doc_ids: list[str] = []
    for _ in range(doc_count):
        raw_doc_id, offset = _read_length_bytes(data, offset)
        try:
            doc_id = raw_doc_id.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DecodeError("document ID is not valid UTF-8", offset) from exc
        if not doc_id:
            raise DecodeError("empty document ID", offset)
        doc_ids.append(doc_id)
    if doc_ids != sorted(set(doc_ids)):
        raise DecodeError("document dictionary is not unique and sorted", offset)
    terms: list[dict[str, Any]] = []
    for _ in range(retained_count):
        term_entry, offset = _decode_term_record(data, offset, doc_ids)
        terms.append(term_entry)
    for _ in range(filtered_count):
        term_entry, offset = _decode_term_record(data, offset, doc_ids)
        terms.append(term_entry)
    if offset != len(data):
        raise DecodeError("trailing bytes after encoded block", offset)
    terms.sort(key=lambda item: item["term"])
    if len({item["term"] for item in terms}) != len(terms):
        raise DecodeError("duplicate term in decoded block", offset)
    computed_fingerprint = block_fingerprint(terms)
    block = {
        "terms": terms,
        "doc_count": len(doc_ids),
        "position_count": sum(
            len(posting["positions"])
            for entry in terms
            for posting in entry["postings"]
        ),
        "fingerprint": fingerprint,
    }
    stats = {
        "tier": tier,
        "format_version": version,
        "block_fingerprint": fingerprint,
        "computed_fingerprint": computed_fingerprint,
        "fingerprint_matches": computed_fingerprint == fingerprint,
        "total_terms": total_terms,
        "retained_terms": retained_count,
        "filtered_terms": filtered_count,
        "low_frequency_threshold": low_frequency_threshold,
        "doc_count": len(doc_ids),
        "posting_count": sum(entry["doc_freq"] for entry in terms),
        "position_count": block["position_count"],
        "encoded_bytes": len(data),
        "decoded_checksum": decoded_checksum(terms),
    }
    return block, stats


def compare_decoded_block(
    original: dict[str, Any],
    decoded: dict[str, Any],
) -> tuple[bool, str | None, Any, Any]:
    original_projection = {
        "terms": [
            {
                "term": entry["term"],
                "doc_freq": len(entry["postings"]),
                "postings": [
                    {"doc_id": posting["doc_id"], "positions": list(posting["positions"])}
                    for posting in entry["postings"]
                ],
            }
            for entry in sorted(original["terms"], key=lambda item: item["term"])
        ]
    }
    decoded_projection = {
        "terms": [
            {
                "term": entry["term"],
                "doc_freq": len(entry["postings"]),
                "postings": [
                    {"doc_id": posting["doc_id"], "positions": list(posting["positions"])}
                    for posting in entry["postings"]
                ],
            }
            for entry in sorted(decoded["terms"], key=lambda item: item["term"])
        ]
    }
    difference = first_difference(original_projection, decoded_projection)
    if difference is None:
        return True, None, None, None
    path, left, right = difference
    return False, path, left, right
