from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROWS = ROOT / "rows"


def dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    rng = random.Random(20261120)
    accounts = []
    for account_id in range(1, 721):
        exposed_id = None if account_id % 149 == 0 else account_id
        segment = 1 if (account_id * 17) % 10 < 7 else 0
        balance = round(rng.uniform(100.0, 50000.0), 2)
        created_day = rng.randrange(0, 365)
        accounts.append([exposed_id, segment, balance, created_day])

    events = []
    for event_id in range(1080):
        account_id = ((event_id * 37) % 720) + 1
        amount = round(rng.uniform(0.0, 200.0), 2)
        channel = event_id % 7
        event_day = rng.randrange(0, 365)
        events.append([account_id, amount, channel, event_day])

    workspace = {
        "format_version": 1,
        "workspace_id": "joinlab-local-demo-v1",
        "attribution_k": 5,
        "tables": [
            {
                "name": "accounts",
                "row_count": len(accounts),
                "skew": False,
                "columns": [
                    {"name": "account_id", "type": "integer"},
                    {"name": "segment", "type": "integer"},
                    {"name": "balance", "type": "float"},
                    {"name": "created_day", "type": "integer"},
                ],
                "data_file": "rows/accounts.json",
            },
            {
                "name": "events",
                "row_count": len(events),
                "skew": False,
                "columns": [
                    {"name": "account_id", "type": "integer"},
                    {"name": "amount", "type": "float"},
                    {"name": "channel", "type": "integer"},
                    {"name": "event_day", "type": "integer"},
                ],
                "data_file": "rows/events.json",
            },
        ],
        "join": {
            "left_table": "accounts",
            "left_key": "account_id",
            "right_table": "events",
            "right_key": "account_id",
        },
        "filters": {
            "accounts": [{"column": "segment", "op": "eq", "value": 1}],
            "events": [{"column": "amount", "op": "ge", "value": 100.0}],
        },
    }
    session = {
        "bucket_count": 10,
        "sample_size": 320,
        "seed": 20261120,
        "quantile_policy": "nearest_rank",
        "attribution_k": 5,
    }
    dump(ROWS / "accounts.json", {"rows": accounts})
    dump(ROWS / "events.json", {"rows": events})
    dump(ROOT / "base_tables.json", workspace)
    dump(ROOT / "session.json", session)


if __name__ == "__main__":
    main()
