"""偏好对数据模型与 JSONL 加载校验（E_PAIR_INVALID）。"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any

from .errors import E_PAIR_INVALID, FailureEntry, FailureLog
from .hashing import sha256_hex, text_tokens_in_vocab

REQUIRED_FIELDS = ("pair_id", "prompt", "chosen", "rejected", "annotator_id", "event_ms")


@dataclass(frozen=True)
class PreferencePair:
    pair_id: str
    prompt: str
    chosen: str
    rejected: str
    annotator_id: str
    event_ms: int
    batch_id: str

    def content_fingerprint(self) -> str:
        return sha256_hex([
            self.pair_id, self.prompt, self.chosen, self.rejected,
            self.annotator_id, str(self.event_ms), self.batch_id,
        ])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_record(rec: dict, normalize: bool) -> tuple[PreferencePair | None, str]:
    for field in REQUIRED_FIELDS:
        if field not in rec:
            return None, f"missing field {field}"

    pair_id = rec["pair_id"]
    if not isinstance(pair_id, str) or not pair_id:
        return None, "pair_id must be non-empty string"

    for field in ("prompt", "chosen", "rejected"):
        val = rec[field]
        if not isinstance(val, str) or not val.strip():
            return None, f"{field} must be non-empty string"
        if not text_tokens_in_vocab(val, normalize):
            return None, f"{field} contains token outside controlled vocab"

    annotator = rec["annotator_id"]
    if not isinstance(annotator, str) or not annotator:
        return None, "annotator_id must be non-empty string"

    event_ms = rec["event_ms"]
    if isinstance(event_ms, bool) or not isinstance(event_ms, int) or event_ms < 0:
        return None, "event_ms must be non-negative int milliseconds"

    batch_id = rec.get("batch_id", "batch-0")
    if not isinstance(batch_id, str) or not batch_id:
        return None, "batch_id must be non-empty string"

    return PreferencePair(
        pair_id=pair_id,
        prompt=rec["prompt"],
        chosen=rec["chosen"],
        rejected=rec["rejected"],
        annotator_id=annotator,
        event_ms=event_ms,
        batch_id=batch_id,
    ), ""


def load_pairs(path: str, normalize: bool, failures: FailureLog) -> dict[str, PreferencePair]:
    """逐行加载 JSONL；非法行进失败清单并跳过，不中断整批。

    返回以 pair_id 为键的映射，重复 ID 记 E_PAIR_INVALID。
    """
    pairs: dict[str, PreferencePair] = {}
    try:
        handle = open(path, "r", encoding="utf-8")
    except OSError as exc:
        failures.add(FailureEntry(
            sort_key="pairs-file", code=E_PAIR_INVALID, stage="load_pair",
            message=f"cannot open jsonl: {exc}",
        ))
        return pairs

    with handle as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.add(FailureEntry(
                    sort_key=f"line-{lineno:06d}", code=E_PAIR_INVALID,
                    stage="load_pair", message=f"invalid json: {exc}",
                    context=f"line={lineno}",
                ))
                continue
            if not isinstance(rec, dict):
                failures.add(FailureEntry(
                    sort_key=f"line-{lineno:06d}", code=E_PAIR_INVALID,
                    stage="load_pair", message="record must be object",
                    context=f"line={lineno}",
                ))
                continue
            pair, problem = _validate_record(rec, normalize)
            if pair is None:
                failures.add(FailureEntry(
                    sort_key=str(rec.get("pair_id", f"line-{lineno:06d}")),
                    code=E_PAIR_INVALID, stage="load_pair",
                    message=problem, context=f"line={lineno}",
                ))
                continue
            if pair.pair_id in pairs:
                failures.add(FailureEntry(
                    sort_key=pair.pair_id, code=E_PAIR_INVALID,
                    stage="load_pair", message="duplicate pair_id",
                    context=f"line={lineno}",
                ))
                continue
            pairs[pair.pair_id] = pair
    return pairs
