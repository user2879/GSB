"""内容指纹与稳定排序键。

缓存键由词序列指纹、模型参数指纹、映射表指纹与版本号派生，
任何输入变化都会改变对应指纹，从而驱动精确失效。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .models import Arc, Sentence

SCHEMA_VERSION = "1.0.0"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def word_sequence_fingerprint(sent: Sentence) -> str:
    parts = [sent.sent_id]
    for word in sent.words:
        parts.append("|".join(word.fingerprint_parts()))
    return sha256_text("\u0001".join(parts))


def model_fingerprint(model_params: dict[str, Any]) -> str:
    # 模型参数可能较大，取规范化 JSON 哈希。
    return content_hash(model_params)


def mapping_fingerprint(mapping: dict[str, Any]) -> str:
    return content_hash(mapping)


def arc_fingerprint(arc: Arc, model_fp: str, mapping_fp: str, version: str) -> str:
    payload = {
        "word": arc.word_idx,
        "head": arc.normalized_head(),
        "relation": arc.relation,
        "score": round(float(arc.score), 6),
        "precision": arc.precision,
        "ref_words": sorted(set(arc.referenced_words)),
        "model": model_fp,
        "mapping": mapping_fp,
        "version": version,
        "schema": SCHEMA_VERSION,
    }
    return sha256_text(canonical_json(payload))


def sentence_fingerprint(
    sent: Sentence, word_fp: str, model_fp: str, mapping_fp: str, version: str
) -> str:
    arc_keys = [
        arc_fingerprint(arc, model_fp, mapping_fp, version) for arc in sorted(sent.arcs, key=lambda a: a.word_idx)
    ]
    payload = {
        "sent_id": sent.sent_id,
        "words": word_fp,
        "arcs": arc_keys,
        "needs_reparse": sent.needs_reparse,
        "model": model_fp,
        "mapping": mapping_fp,
        "version": version,
        "schema": SCHEMA_VERSION,
    }
    return sha256_text(canonical_json(payload))


def artifact_fingerprint(serialized_bytes: bytes, upstream_keys: list[str]) -> str:
    hasher = hashlib.sha256()
    hasher.update(serialized_bytes)
    hasher.update(b"\u0002")
    hasher.update("\u0001".join(sorted(upstream_keys)).encode("utf-8"))
    return hasher.hexdigest()


def stable_task_key(sent_id: str, stage: str) -> str:
    """句 ID 与阶段名派生的稳定排序键。"""
    return f"{sent_id}\u0003{stage}"
