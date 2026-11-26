"""Generate deterministic local sample spec and NDJSON event stream."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def generate(out_dir, *, n_events=48, subscriber_limit=3, error_margin_ms=2):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    spec = {
        "subscriber_limit": subscriber_limit,
        "error_margin_ms": error_margin_ms,
        "subscribers": [
            {
                "subscriber_id": "sub-alpha",
                "host": "127.0.0.1",
                "port": 20301,
                "budget_ms": 60,
                "inflight_event_limit": 8,
                "seed": 101,
                "topics": ["orders", "metrics"],
            },
            {
                "subscriber_id": "sub-beta",
                "host": "127.0.0.1",
                "port": 20302,
                "budget_ms": 45,
                "inflight_event_limit": 5,
                "seed": 202,
                "topics": ["orders"],
            },
            {
                "subscriber_id": "sub-gamma",
                "host": "127.0.0.1",
                "port": 20303,
                "budget_ms": 50,
                "inflight_event_limit": 12,
                "seed": 303,
                "topics": ["metrics"],
            },
        ],
        "channels": [
            {
                "channel_id": "chan-orders",
                "topic": "orders",
                "batch_size": 5,
                "rate_limit": 6,
            },
            {
                "channel_id": "chan-metrics",
                "topic": "metrics",
                "batch_size": 4,
                "rate_limit": 8,
            },
        ],
        "events_file": "events.ndjson",
    }
    (out / "spec.json").write_text(
        json.dumps(spec, sort_keys=True, indent=2),
        encoding="utf-8",
    )

    lines = []
    for seq in range(1, n_events + 1):
        topic = "orders" if seq % 2 == 1 else "metrics"
        timestamp = 1000 + seq * 12
        # Deterministic bounded integer payload, positive.
        payload = 100 + ((seq * 37 + (7 if topic == "orders" else 13)) % 900)
        lines.append(
            json.dumps(
                {
                    "seq": seq,
                    "timestamp_ms": timestamp,
                    "topic": topic,
                    "payload": payload,
                },
                sort_keys=True,
            )
        )
    (out / "events.ndjson").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out / "spec.json", out / "events.ndjson"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="samples/demo")
    parser.add_argument("--events", type=int, default=48)
    args = parser.parse_args(argv)
    spec_path, events_path = generate(args.out, n_events=args.events)
    print(f"wrote {spec_path} and {events_path}")


if __name__ == "__main__":
    main()
