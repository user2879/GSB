from __future__ import annotations

import json
import socket
from pathlib import Path


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def write_bundle(
    root: Path,
    *,
    alpha_capacity: int = 1,
    beta_capacity: int = 2,
    alpha_duration: int = 5000,
    beta_duration: int = 5000,
    priorities: dict[str, int] | None = None,
) -> dict[str, str]:
    priorities = priorities or {"h-high": 9, "h-mid": 5, "h-low": 1}
    alpha_port = free_port()
    beta_port = free_port()
    while beta_port == alpha_port:
        beta_port = free_port()
    clusters = {
        "clusters": [
            {
                "capacity": alpha_capacity,
                "cluster_id": "alpha",
                "host": "127.0.0.1",
                "lease_duration_ms": alpha_duration,
                "port": alpha_port,
                "renewal_interval_ms": 1000,
            },
            {
                "capacity": beta_capacity,
                "cluster_id": "beta",
                "host": "127.0.0.1",
                "lease_duration_ms": beta_duration,
                "port": beta_port,
                "renewal_interval_ms": 1000,
            },
        ]
    }
    holders = {
        "holders": [
            {"holder_id": holder_id, "priority": priority, "targets": ["alpha", "beta"]}
            for holder_id, priority in sorted(priorities.items())
        ]
    }
    events = [
        (1, 0, "alpha", "h-mid", "apply"),
        (2, 1000, "alpha", "h-mid", "renew"),
        (3, 1500, "beta", "h-mid", "apply"),
        (4, 2000, "beta", "h-low", "apply"),
        (5, 2500, "beta", "h-high", "apply"),
        (6, 3500, "alpha", "h-high", "apply"),
        (7, 5000, "alpha", "h-high", "renew"),
        (8, 6100, "alpha", "h-mid", "apply"),
        (9, 8100, "alpha", "h-low", "apply"),
        (10, 10001, "alpha", "h-mid", "apply"),
        (11, 10500, "beta", "h-low", "apply"),
        (12, 10600, "beta", "h-high", "apply"),
        (13, 11000, "alpha", "h-mid", "release"),
        (14, 11100, "alpha", "h-low", "apply"),
    ]
    root.mkdir(parents=True, exist_ok=True)
    clusters_path = root / "clusters.json"
    holders_path = root / "holders.json"
    events_path = root / "events.ndjson"
    clusters_path.write_text(json.dumps(clusters, indent=2) + "\n", encoding="utf-8")
    holders_path.write_text(json.dumps(holders, indent=2) + "\n", encoding="utf-8")
    events_path.write_text(
        "".join(
            json.dumps(
                {
                    "cluster_id": cluster_id,
                    "event_seq": event_seq,
                    "event_type": event_type,
                    "holder_id": holder_id,
                    "timestamp_ms": timestamp,
                }
            )
            + "\n"
            for event_seq, timestamp, cluster_id, holder_id, event_type in events
        ),
        encoding="utf-8",
    )
    return {
        "clusters": str(clusters_path),
        "events": str(events_path),
        "holders": str(holders_path),
    }
