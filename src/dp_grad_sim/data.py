from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import SimError
from .util import hash_mapping


@dataclass(frozen=True)
class Shard:
    worker_id: int
    shard_id: str
    samples: int
    feature_digest: str
    label_digest: str
    raw: dict[str, Any]

    def as_hash_payload(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "shard_id": self.shard_id,
            "samples": self.samples,
            "feature_digest": self.feature_digest,
            "label_digest": self.label_digest,
        }


@dataclass(frozen=True)
class ShardSet:
    shards: tuple[Shard, ...]
    by_worker: dict[int, tuple[Shard, ...]]
    data_hash: str


def load_shards(path: str | Path, workers: int) -> ShardSet:
    shard_path = Path(path)
    try:
        lines = shard_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise SimError("E_DATA_INVALID", f"shard file not found: {shard_path}") from exc

    shards: list[Shard] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SimError(
                "E_DATA_INVALID", f"invalid JSONL at line {line_number}: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise SimError("E_DATA_INVALID", f"line {line_number} must be a JSON object")
        try:
            worker_id = raw["worker_id"]
            shard_id = raw["shard_id"]
            samples = raw["samples"]
            feature_digest = raw["feature_digest"]
            label_digest = raw["label_digest"]
        except KeyError as exc:
            raise SimError("E_DATA_INVALID", f"line {line_number} is missing {exc.args[0]}") from exc
        if isinstance(worker_id, bool) or not isinstance(worker_id, int) or not 0 <= worker_id < workers:
            raise SimError("E_DATA_INVALID", f"line {line_number} has invalid worker_id {worker_id!r}")
        if not isinstance(shard_id, str) or not shard_id:
            raise SimError("E_DATA_INVALID", f"line {line_number} has invalid shard_id")
        if shard_id in seen_ids:
            raise SimError("E_DATA_INVALID", f"duplicate shard_id {shard_id!r}")
        seen_ids.add(shard_id)
        if isinstance(samples, bool) or not isinstance(samples, int) or samples <= 0:
            raise SimError("E_DATA_INVALID", f"line {line_number} has invalid samples {samples!r}")
        if not isinstance(feature_digest, str) or not feature_digest:
            raise SimError("E_DATA_INVALID", f"line {line_number} has invalid feature_digest")
        if not isinstance(label_digest, str) or not label_digest:
            raise SimError("E_DATA_INVALID", f"line {line_number} has invalid label_digest")
        shards.append(
            Shard(
                worker_id=worker_id,
                shard_id=shard_id,
                samples=samples,
                feature_digest=feature_digest,
                label_digest=label_digest,
                raw=raw,
            )
        )

    if not shards:
        raise SimError("E_DATA_INVALID", "shard description is empty")

    grouped: dict[int, list[Shard]] = {worker_id: [] for worker_id in range(workers)}
    for shard in shards:
        grouped[shard.worker_id].append(shard)
    missing = [worker_id for worker_id, worker_shards in grouped.items() if not worker_shards]
    if missing:
        raise SimError("E_DATA_INVALID", f"workers without shards: {missing}")

    normalized = {
        worker_id: tuple(sorted(worker_shards, key=lambda shard: shard.shard_id))
        for worker_id, worker_shards in grouped.items()
    }
    hash_payload = [shard.as_hash_payload() for shard in sorted(shards, key=lambda item: item.shard_id)]
    return ShardSet(shards=tuple(shards), by_worker=normalized, data_hash=hash_mapping(hash_payload))

