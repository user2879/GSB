"""Content-fingerprint cache keys and atomic persistent stage store."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


STAGE_EMBED = "embed"
STAGE_QUANT = "quant"
STAGE_COMPARE = "compare"
STAGE_SEGMENT = "segment"
STAGES = (STAGE_EMBED, STAGE_QUANT, STAGE_COMPARE, STAGE_SEGMENT)


def _digest(prefix: str, *parts: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(prefix.encode("utf-8"))
    for part in parts:
        hasher.update(b"|")
        hasher.update(str(part).encode("utf-8"))
    return hasher.hexdigest()


def embed_key(content_key: str, config_fp: str = "") -> str:
    # Embeddings depend only on window audio content and the projection; the
    # projection is itself config-derived and stable, while the content hash
    # dominates the key so unchanged audio is reused across threshold edits.
    return _digest("embed", content_key)


def quant_key(embed_key_value: str, config_fp: str, word_bits: int,
              frac_bits: int) -> str:
    return _digest("quant", embed_key_value,
                   str(word_bits), str(frac_bits))


def compare_key(left_quant_key: str, right_quant_key: str,
                threshold_q16: int) -> str:
    lo, hi = sorted((left_quant_key, right_quant_key))
    return _digest("compare", lo, hi, str(threshold_q16))


def segment_key(compare_keys: List[str], config_fp: str, word_bits: int,
                frac_bits: int) -> str:
    joined = ",".join(sorted(compare_keys))
    # Boundary arbitration parameters come from config; keep config hash so a
    # boundary-rule/min-segment edit invalidates segmentation correctly.
    return _digest("segment", joined, config_fp, str(word_bits), str(frac_bits))


def atomic_write_json(path: str, payload: Any) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


@dataclass
class ReuseReport:
    reused: Dict[str, List[str]] = field(default_factory=lambda: {s: [] for s in STAGES})
    recomputed: Dict[str, List[str]] = field(default_factory=lambda: {s: [] for s in STAGES})

    def mark_reused(self, stage: str, key: str) -> None:
        self.reused[stage].append(key)

    def mark_recomputed(self, stage: str, key: str) -> None:
        self.recomputed[stage].append(key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reused": {k: sorted(v) for k, v in self.reused.items()},
            "recomputed": {k: sorted(v) for k, v in self.recomputed.items()},
        }


class CacheStore:
    """Stage-keyed persistent cache with whole-file atomic replacement."""

    def __init__(self, root: str):
        self.root = root
        self.stage_dir = os.path.join(root, "cache")
        os.makedirs(self.stage_dir, exist_ok=True)
        self.report = ReuseReport()
        self._index_path = os.path.join(self.stage_dir, "index.json")
        self.index: Dict[str, Dict[str, Any]] = self._load_index()

    def _stage_path(self, stage: str) -> str:
        return os.path.join(self.stage_dir, f"{stage}.json")

    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        if os.path.exists(self._index_path):
            with open(self._index_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        return {stage: {} for stage in STAGES}

    def flush_index(self) -> None:
        atomic_write_json(self._index_path, self.index)

    def get(self, stage: str, key: str) -> Optional[Any]:
        table = self.index.setdefault(stage, {})
        entry = table.get(key)
        if entry is None:
            return None
        path = os.path.join(self.stage_dir, entry["file"])
        if not os.path.exists(path):
            table.pop(key, None)
            return None
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.report.mark_reused(stage, key)
        return payload

    def put(self, stage: str, key: str, payload: Any) -> None:
        table = self.index.setdefault(stage, {})
        file_name = f"{stage}_{len(table):06d}.json"
        atomic_write_json(os.path.join(self.stage_dir, file_name), payload)
        table[key] = {"file": file_name}
        self.report.mark_recomputed(stage, key)

    def invalidate_prefixes(self, stage: str, key_substrings: Tuple[str, ...]) -> List[str]:
        table = self.index.setdefault(stage, {})
        removed = []
        for key in list(table.keys()):
            if any(token in key for token in key_substrings):
                path = os.path.join(self.stage_dir, table[key]["file"])
                if os.path.exists(path):
                    os.unlink(path)
                table.pop(key)
                removed.append(key)
        return removed
