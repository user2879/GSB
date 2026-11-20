"""Generate deterministic demo specifications.

NumPy is used only here, with a fixed PCG64 seed, to produce the numeric
tokens embedded in the sample content.  The runtime package itself has no
NumPy dependency.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record(
    path: str,
    etag: str,
    content: bytes,
    ttl_ms: int,
    *,
    derived_from: list[str] | None = None,
    separator: str = "|",
) -> dict:
    dependencies = list(derived_from or [])
    return {
        "path": path,
        "etag": etag,
        "ttl_ms": ttl_ms,
        "content_b64": base64.b64encode(content).decode("ascii"),
        "content_sha256": sha256(content),
        "derived_from": dependencies,
        "derivation": {"rule": "concat_v1", "separator": separator} if dependencies else None,
    }


def build_version(version: int, token: int) -> dict:
    config = b'{"region":"local","release":"config-v1"}'
    root = f"ROOT:{token}:v{version}".encode("utf-8")
    left = root
    right = root
    page = b"\n".join((left, right, config))
    return {
        "schema": "edgecache.spec/v1",
        "policy": {
            "version": "policy-v1",
            "key_fields": ["node_id", "path", "etag"],
            "nodes": [
                {"node_id": "edge-a", "capacity": 4, "eviction_rule": "stable_fifo"},
                {"node_id": "edge-b", "capacity": 5, "eviction_rule": "smallest_key"},
                {"node_id": "edge-c", "capacity": 3, "eviction_rule": "nearest_expiry"},
            ],
        },
        "content": [
            record("/api/config", "config-v1", config, 45_000),
            record("/data/left", f"left-v{version}", left, 40_000, derived_from=["/data/root"]),
            record("/data/right", f"right-v{version}", right, 35_000, derived_from=["/data/root"]),
            record("/data/root", f"root-v{version}", root, 30_000),
            record(
                "/page/home",
                f"page-v{version}",
                page,
                50_000,
                derived_from=["/data/left", "/data/right", "/api/config"],
                separator="\n",
            ),
        ],
    }


def main() -> int:
    rng = np.random.default_rng(20261121)
    tokens = [int(value) for value in rng.integers(100, 999, size=3)]
    samples = ROOT / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    for version, token in enumerate(tokens, start=1):
        target = samples / ("demo_spec.json" if version == 1 else f"demo_spec_v{version}.json")
        target.write_text(
            json.dumps(build_version(version, token), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"wrote {target.relative_to(ROOT)} token={token}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
