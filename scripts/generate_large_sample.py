"""Generate the deterministic above-threshold range-lock sample."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    target = root / "samples" / "large_range.json"
    rows = [f"range:{idx:02d}" for idx in range(16)]
    edges = [
        {"from": rows[idx], "to": rows[idx + 1], "kind": "reference"}
        for idx in range(12)
    ]
    transactions = [{
        "id": "TX01",
        "start_seq": 1,
        "isolation": "serializable",
        "operations": [
            {"seq": 1, "kind": "read", "row": rows[0]},
            {"seq": 2, "kind": "read", "row": rows[1]},
            {"seq": 3, "kind": "read", "row": rows[2]},
        ],
    }]
    for idx in range(2, 13):
        row = rows[idx + 1]
        transactions.append({
            "id": f"TX{idx:02d}",
            "start_seq": idx,
            "isolation": "read_committed",
            "operations": [
                {"seq": 1, "kind": "read", "row": row},
            ],
        })
    document = {
        "schema_version": 1,
        "row_keys": rows,
        "dependency_edges": edges,
        "initial_versions": {
            row: {"value": 0, "version": 0, "status": "committed"} for row in rows
        },
        "transactions": transactions,
    }
    target.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(target)


if __name__ == "__main__":
    main()

