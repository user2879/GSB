"""Deterministic local sample generation.

Samples are produced from fixed seeds and explicit, hand-designed failure
patterns so that the sliding-window breaker deterministically opens and later
recovers. Nothing here uses the network or any external account.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_SEED = 20261126


def _requests_for_service(
    service_id: str, count: int, failure_pattern: list[int], batch_size: int
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seq in range(count):
        failed = failure_pattern[seq % len(failure_pattern)]
        batch_index = seq // batch_size
        out.append({
            "request_id": f"{service_id}-R{seq:03d}",
            "service_id": service_id,
            "batch_id": f"{service_id}-B{batch_index:02d}",
            "arrival_seq": seq,
            "failed": failed,
        })
    return out


def build_request_stream() -> dict[str, Any]:
    requests: list[dict[str, Any]] = []
    # svcA: window length 4, threshold 4000 (>=2 fails /4 opens).
    # Pattern forces the first window to 3 failures -> open, then a clean
    # stretch lets the explicit recovery window pass and close the breaker.
    svc_a_pattern = [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1]
    requests += _requests_for_service("svcA", 16, svc_a_pattern, 4)
    # svcB: window length 5, threshold 5000 (>=3 fails /5 opens).
    svc_b_pattern = [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    requests += _requests_for_service("svcB", 15, svc_b_pattern, 5)
    # svcC: rps_limit 0 means every admitted request is rate rejected before
    # compute; failures still present to exercise pure policy paths.
    svc_c_pattern = [1, 1, 0, 0, 1, 0]
    requests += _requests_for_service("svcC", 6, svc_c_pattern, 3)
    return {"requests": requests}


def write_samples(out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    requests_path = out_dir / "requests.json"
    stream = build_request_stream()
    requests_path.write_text(
        json.dumps(stream, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    return {"requests": requests_path}


def build_edited_services(original: dict[str, Any], *,
                          service_id: str, changes: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-copied services document with one service edited and its
    config_version bumped monotonically."""
    import copy

    edited = copy.deepcopy(original)
    for svc in edited["services"]:
        if svc["service_id"] == service_id:
            svc.update(changes)
            svc["config_version"] = int(svc["config_version"]) + 1
            return edited
    raise KeyError(service_id)
