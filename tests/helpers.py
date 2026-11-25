from __future__ import annotations

from typing import Any

from speaker_diag.orchestration import find_free_loopback_port
from speaker_diag.spec import WorkloadSpec, load_workload_data


def tiny_workload(
    *,
    port1: int | None = None,
    port2: int | None = None,
    duration_ms: int = 1600,
) -> dict[str, Any]:
    port1 = port1 or find_free_loopback_port()
    port2 = port2 or find_free_loopback_port()
    if port2 == port1:
        port2 = find_free_loopback_port()
    return {
        "spec_version": 1,
        "session_version": 1,
        "seed": 424242,
        "audio": {
            "duration_ms": duration_ms,
            "sample_rate": 16000,
            "channels": 1,
            "sample_width_bits": 16,
            "speaker_count": 3,
            "pause_ms": [40, 90],
            "overlap_bps": 1200,
        },
        "window": {"window_ms": 400, "hop_ms": 200, "overlap_bps": 5000},
        "batch": {"batch_size": 3},
        "quantization": {"scheme": "symmetric_int", "fractional_bits": 15},
        "workers": [
            {
                "worker_id": "W01",
                "host": "127.0.0.1",
                "port": port1,
                "model_dim": 12,
                "hidden_dim": 8,
                "seed": 9001,
                "parameter_version": 1,
                "max_concurrent": 1,
                "max_worker_leases": 2,
                "lease_ms": 1000,
            },
            {
                "worker_id": "W02",
                "host": "127.0.0.1",
                "port": port2,
                "model_dim": 12,
                "hidden_dim": 8,
                "seed": 9002,
                "parameter_version": 1,
                "max_concurrent": 1,
                "max_worker_leases": 2,
                "lease_ms": 1000,
            },
        ],
        "requests": [
            {"batch_id": "B001", "worker_id": "W01", "window_ids": ["W0001", "W0002"]},
            {"batch_id": "B002", "worker_id": "W02", "window_ids": ["W0003", "W0004"]},
            {"batch_id": "B003", "worker_id": "W01", "window_ids": ["W0002", "W0005"]},
        ],
        "limits": {
            "global_batch_leases": 2,
            "metric_scale": 10000,
            "batch_score_error_bound_units": 2,
            "worker_watermark_error_bound_units": 2,
            "handshake_timeout_ms": 1500,
            "io_timeout_ms": 2000,
        },
    }


def tiny_spec(**kwargs: Any) -> WorkloadSpec:
    return load_workload_data(tiny_workload(**kwargs), source_name="<tiny-test>")

