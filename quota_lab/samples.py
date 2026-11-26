from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .errors import SpecError
from .spec import SessionSpec, is_int


def generate_samples_for_shard(
    spec: SessionSpec,
    shard_id: str,
) -> list[list[int]]:
    shard = spec.shard(shard_id)
    node = spec.node(shard.node_id)
    shard_index = list(spec.topo_order).index(shard_id)
    rng = np.random.default_rng([int(node.seed), int(shard_index) + 1, 982451653])
    groups = rng.integers(
        0,
        spec.accounting.group_count,
        size=int(shard.sample_rows),
        dtype=np.int64,
    )
    values = rng.integers(1, 100, size=int(shard.sample_rows), dtype=np.int64)
    return [[int(group), int(value)] for group, value in zip(groups, values)]


def generate_event_records(spec: SessionSpec) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for shard in spec.shards:
        samples = generate_samples_for_shard(spec, shard.shard_id)
        for seq, (group, value) in enumerate(samples):
            records.append(
                {
                    "session_id": spec.session_id,
                    "shard_id": shard.shard_id,
                    "node_id": shard.node_id,
                    "seq": seq,
                    "group": group,
                    "value": value,
                }
            )
    return records


def write_event_stream(spec: SessionSpec, path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = generate_event_records(spec)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    return len(records)


def load_event_stream(
    path: str | Path,
    spec: SessionSpec,
) -> dict[str, list[list[int]]]:
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SpecError(f"cannot read event stream: {path}", field="events") from exc

    result: dict[str, list[list[int]]] = {shard.shard_id: [] for shard in spec.shards}
    seen_seq: dict[str, set[int]] = {shard.shard_id: set() for shard in spec.shards}
    known_shards = set(result)

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        field_prefix = f"events[{line_number - 1}]"
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SpecError(
                "event line is not valid JSON",
                field=f"{field_prefix}_json",
                details={"line": line_number, "column": exc.colno},
            ) from exc
        if not isinstance(record, dict):
            raise SpecError("event must be an object", field=field_prefix)

        shard_id = record.get("shard_id")
        if not isinstance(shard_id, str) or shard_id not in known_shards:
            raise SpecError("event references unknown shard", field=f"{field_prefix}.shard_id")

        if record.get("session_id") != spec.session_id:
            raise SpecError("event session_id does not match declaration", field=f"{field_prefix}.session_id")

        seq = record.get("seq")
        if not is_int(seq) or seq < 0:
            raise SpecError("event seq must be non-negative integer", field=f"{field_prefix}.seq")
        if seq in seen_seq[shard_id]:
            raise SpecError("duplicate event seq for shard", field=f"{field_prefix}.seq")
        seen_seq[shard_id].add(seq)

        group = record.get("group")
        if not is_int(group) or not 0 <= group < spec.accounting.group_count:
            raise SpecError(
                "event group must be an integer in configured group range",
                field=f"{field_prefix}.group",
            )

        value = record.get("value")
        if not is_int(value) or not 0 <= value <= 1_000_000_000:
            raise SpecError("event value must be integer in [0,1000000000]", field=f"{field_prefix}.value")

        node_id = record.get("node_id")
        expected_node = spec.shard(shard_id).node_id
        if node_id != expected_node:
            raise SpecError(
                "event node_id does not match shard declaration",
                field=f"{field_prefix}.node_id",
                details={"expected": expected_node, "actual": node_id},
            )
        result[shard_id].append([int(group), int(value)])

    for shard in spec.shards:
        samples = result[shard.shard_id]
        if len(samples) != shard.sample_rows:
            raise SpecError(
                "event row count does not match sample_rows",
                field=f"events.{shard.shard_id}",
                details={"expected": shard.sample_rows, "actual": len(samples)},
            )
        if sorted(seen_seq[shard.shard_id]) != list(range(shard.sample_rows)):
            raise SpecError(
                "event seq must be contiguous from zero for each shard",
                field=f"events.{shard.shard_id}.seq",
            )
    return result
