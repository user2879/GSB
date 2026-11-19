from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from .canonical import canonical_json, digest_json
from .errors import fail


@dataclass
class SampleResult:
    table: str
    key: str
    key_type: str
    seed: int
    requested_size: int
    table_row_count: int
    sampled_row_count: int
    null_sample_count: int
    data_digest: str
    frequencies: dict[int | float, int]
    sample_digest: str
    sampled_rids: tuple[str, ...]

    def frequency_rows(self) -> list[dict[str, Any]]:
        return [
            {"key": key, "count": count}
            for key, count in sorted(self.frequencies.items(), key=lambda item: item[0])
        ]

    def as_dict(self, include_frequencies: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "table": self.table,
            "key": self.key,
            "key_type": self.key_type,
            "seed": self.seed,
            "requested_size": self.requested_size,
            "table_row_count": self.table_row_count,
            "sampled_row_count": self.sampled_row_count,
            "null_sample_count": self.null_sample_count,
            "data_digest": self.data_digest,
            "sample_digest": self.sample_digest,
            "distinct_key_count": len(self.frequencies),
        }
        if include_frequencies:
            result["key_frequencies"] = self.frequency_rows()
        return result


def _priority(seed: int, data_digest: str, rid: str) -> str:
    material = f"{seed}\x1f{data_digest}\x1f{rid}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def build_sample(
    table: str,
    key: str,
    key_type: str,
    rows: Iterable[dict[str, Any]],
    sample_size: int,
    seed: int,
    data_digest: str,
) -> SampleResult:
    row_list = list(rows)
    if not isinstance(sample_size, int) or isinstance(sample_size, bool) or sample_size <= 0:
        fail("E_SAMPLE", "抽样规模必须是正整数", "params.sample_size", value=sample_size)
    if not isinstance(seed, int) or isinstance(seed, bool):
        fail("E_SAMPLE", "随机种子必须是整数", "params.seed", value=seed)
    if sample_size > len(row_list):
        fail(
            "E_SAMPLE",
            "抽样规模超过表行数",
            "params.sample_size",
            sample_size=sample_size,
            table_rows=len(row_list),
        )
    ranked = sorted(
        row_list,
        key=lambda row: (
            _priority(seed, data_digest, row["__rid"]),
            row["__rid"],
        ),
    )
    selected = ranked[:sample_size]
    selected.sort(key=lambda row: row["__rid"])
    frequencies: Counter[int | float] = Counter()
    null_count = 0
    for row in selected:
        value = row.get(key)
        if value is None:
            null_count += 1
        else:
            frequencies[value] += 1
    digest_payload = {
        "table": table,
        "key": key,
        "key_type": key_type,
        "seed": seed,
        "requested_size": sample_size,
        "table_row_count": len(row_list),
        "data_digest": data_digest,
        "rows": [
            {"rid": row["__rid"], "key": row.get(key)}
            for row in selected
        ],
    }
    sample_digest = digest_json(digest_payload)
    return SampleResult(
        table=table,
        key=key,
        key_type=key_type,
        seed=seed,
        requested_size=sample_size,
        table_row_count=len(row_list),
        sampled_row_count=len(selected),
        null_sample_count=null_count,
        data_digest=data_digest,
        frequencies=dict(frequencies),
        sample_digest=sample_digest,
        sampled_rids=tuple(row["__rid"] for row in selected),
    )


def sample_bytes(sample: SampleResult) -> bytes:
    return canonical_json(sample.as_dict()).encode("utf-8")

