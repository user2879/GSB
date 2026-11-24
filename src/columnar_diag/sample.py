from __future__ import annotations

from pathlib import Path

from .codec import encode_block
from .models import BlockSpec, ColumnDef, QuerySpec, RowGroup, SegmentSpec
from .spec import parse_spec, write_spec
from .util import atomic_write_json


_COLUMNS = (
    ColumnDef("region"),
    ColumnDef("amount"),
    ColumnDef("qty"),
    ColumnDef("channel"),
)

_ROWS = {
    "rg0": {
        "region": (1, 2, 1, 3, 2, 1, 3, 2),
        "amount": (5, 12, 18, 7, 21, 30, 9, 15),
        "qty": (2, 5, 1, 4, 3, 2, 6, 1),
        "channel": (10, 20, 10, 30, 20, 10, 30, 20),
    },
    "rg1": {
        "region": (1, 3, 2, 1, 2),
        "amount": (11, 25, 8, 40, 19),
        "qty": (2, 1, 5, 3, 4),
        "channel": (20, 30, 10, 20, 30),
    },
    "rg2": {
        "region": (3, 1, 2, 3, 1, 2, 1),
        "amount": (6, 22, 14, 33, 10, 27, 4),
        "qty": (5, 2, 1, 4, 3, 2, 6),
        "channel": (10, 20, 30, 10, 20, 30, 10),
    },
}

_ROW_GROUPS = (
    RowGroup("rg0", 0, 8),
    RowGroup("rg1", 8, 5),
    RowGroup("rg2", 13, 7),
)

_BLOCK_LAYOUT = {
    ("region", "rg0"): ("dictionary", True),
    ("region", "rg1"): ("dictionary", False),
    ("region", "rg2"): ("dictionary", True),
    ("amount", "rg0"): ("raw", True),
    ("amount", "rg1"): ("raw", True),
    ("amount", "rg2"): ("rle", False),
    ("qty", "rg0"): ("rle", True),
    ("qty", "rg1"): ("dictionary", True),
    ("qty", "rg2"): ("rle", False),
    ("channel", "rg0"): ("dictionary", True),
    ("channel", "rg1"): ("raw", False),
    ("channel", "rg2"): ("dictionary", True),
}

_QUERIES = (
    QuerySpec(
        "q1",
        ("region", "amount"),
        "region IN (1, 3) AND (amount >= 10)",
    ),
    QuerySpec(
        "q2",
        ("region", "qty"),
        "amount >= 20 AND qty < 5",
    ),
    QuerySpec(
        "q3",
        ("region", "amount", "qty"),
        "",
    ),
    QuerySpec(
        "q4",
        ("qty",),
        "qty = -999",
    ),
    QuerySpec(
        "q5",
        ("channel",),
        "region = 2 AND amount >= 20",
    ),
)


def build_sample_spec() -> SegmentSpec:
    blocks: list[BlockSpec] = []
    for group in _ROW_GROUPS:
        for column in _COLUMNS:
            encoding, compressed = _BLOCK_LAYOUT[(column.name, group.id)]
            values = _ROWS[group.id][column.name]
            block_id = f"{column.name}@{group.id}"
            encoded = encode_block(
                block_id=block_id,
                column=column.name,
                row_group=group.id,
                encoding=encoding,
                compressed=compressed,
                values=values,
            )
            blocks.append(
                BlockSpec(
                    id=block_id,
                    column=column.name,
                    row_group=group.id,
                    encoding=encoding,
                    compressed=compressed,
                    values=values,
                    declared_pre_bytes=encoded.pre_bytes,
                    declared_post_bytes=encoded.post_bytes,
                )
            )
    return SegmentSpec(
        segment_id="sales-local-v1",
        columns=_COLUMNS,
        row_groups=_ROW_GROUPS,
        blocks=tuple(blocks),
        queries=_QUERIES,
    )


def sample_changes() -> dict:
    return {
        "changeset_id": "cs-amount-rg1-001",
        "reason": "重写 amount@rg1 样本、切换为字典编码并关闭压缩",
        "changes": [
            {
                "column": "amount",
                "row_group": "rg1",
                "values": [13, 26, 23, 41, 18],
                "encoding": "dictionary",
                "compressed": False,
            }
        ],
    }


def sample_budget() -> dict[str, int]:
    return {
        "global_parallelism": 4,
        "column_memory_budget_bytes": 128,
        "segment_concurrency": 3,
    }


def write_sample_files(directory: str | Path) -> dict[str, str]:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    spec = build_sample_spec()
    spec = parse_spec(spec.to_dict())
    write_spec(spec, target / "sample_spec.json")
    atomic_write_json(target / "sample_changeset.json", sample_changes())
    atomic_write_json(target / "sample_budget.json", sample_budget())
    return {
        "spec": str(target / "sample_spec.json"),
        "changeset": str(target / "sample_changeset.json"),
        "budget": str(target / "sample_budget.json"),
    }
