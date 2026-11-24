from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

from .config import ConfigBundle
from .corpus import normalize_sequence
from .features import content_fingerprint_for_block


def paragraph_sort_key(record: Dict[str, Any]):
    sequence_key = record.get("sequence_key") or list(normalize_sequence(record.get("sequence_no")))
    return (str(record["document_id"]), sequence_key[0], sequence_key[1], str(record["paragraph_id"]))


def chunk_paragraph(record: Dict[str, Any], config: ConfigBundle) -> List[Dict[str, Any]]:
    window = int(config.data["chunk_window"])
    overlap = int(config.data["chunk_overlap"])
    pad_token = str(config.data["pad_token"])
    step = window - overlap
    terms = list(record["terms"])
    if not terms:
        return []

    chunks: List[Dict[str, Any]] = []
    chunk_ord = 0
    start = 0
    while start < len(terms):
        selected = terms[start:start + window]
        padding_count = window - len(selected)
        padded = padding_count > 0
        chunk_terms = selected + [pad_token] * padding_count
        chunks.append(
            {
                "paragraph_id": record["paragraph_id"],
                "document_id": record["document_id"],
                "paragraph_sequence_no": record["sequence_no"],
                "chunk_ord": chunk_ord,
                "start": start,
                "end": min(start + window, len(terms)),
                "terms": chunk_terms,
                "raw_term_count": len(selected),
                "padded": padded,
                "padding_count": padding_count,
            }
        )
        chunk_ord += 1
        start += step
    return chunks


def build_blocks(
    records: Iterable[Dict[str, Any]],
    config: ConfigBundle,
    overrides: Optional[Dict[str, List[str]]] = None,
    deleted_blocks: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    overrides = overrides or {}
    deleted_blocks = deleted_blocks or set()
    by_document: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        by_document.setdefault(str(record["document_id"]), []).append(record)

    blocks: List[Dict[str, Any]] = []
    for document_id in sorted(by_document):
        block_seq = 0
        for record in sorted(by_document[document_id], key=paragraph_sort_key):
            for chunk in chunk_paragraph(record, config):
                block_id = f"{document_id}-C{block_seq:04d}"
                block = dict(chunk)
                block.update(
                    {
                        "block_id": block_id,
                        "block_seq": block_seq,
                        "stable_sort_key": f"{document_id}|{block_seq:012d}",
                    }
                )
                if block_id in deleted_blocks:
                    block_seq += 1
                    continue
                if block_id in overrides:
                    replacement = list(overrides[block_id])
                    padding_count = max(0, int(config.data["chunk_window"]) - len(replacement))
                    block["terms"] = replacement + [str(config.data["pad_token"])] * padding_count
                    block["raw_term_count"] = len(replacement)
                    block["padding_count"] = padding_count
                    block["padded"] = padding_count > 0
                    block["overridden"] = True
                else:
                    block["overridden"] = False
                block["content_fingerprint"] = content_fingerprint_for_block(block, config)
                block["task_keys"] = {
                    "projection": f"{block['stable_sort_key']}|projection",
                    "bucket_write": f"{block['stable_sort_key']}|bucket_write",
                }
                blocks.append(block)
                block_seq += 1
    return sorted(blocks, key=lambda item: (item["document_id"], item["block_seq"], item["block_id"]))
