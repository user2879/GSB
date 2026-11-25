"""Built-in deterministic sample workload generator.

No attachment or external account is consulted.  The same function always
creates the same JSON bytes (apart from no timestamps at all).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .canonical import atomic_write_bytes, atomic_write_json, canonical_sha256, sha256_hex
from .engine import infer_local_batch
from .receipts import context_for, normalize_receipt, strip_to_r1, strip_to_r2
from .spec import WorkloadSpec, load_workload_data
from .synthetic import synthesize


DEFAULT_PORTS = (48_191, 48_192)


def _request_groups() -> list[dict[str, Any]]:
    groups = [
        ("B001", "W01", [1, 2, 3, 4]),
        ("B002", "W02", [5, 6, 7, 8]),
        ("B003", "W01", [9, 10, 11, 12]),
        ("B004", "W02", [13, 14, 15, 16]),
        ("B005", "W01", [17, 18, 19, 20]),
        ("B006", "W02", [21, 22, 23, 24]),
        ("B007", "W01", [25, 26, 27, 28]),
        ("B008", "W02", [29, 30, 31, 32]),
        ("B009", "W01", [33, 34, 35, 36]),
        ("B010", "W02", [3, 15, 35]),
    ]
    return [
        {
            "batch_id": batch_id,
            "worker_id": worker_id,
            "window_ids": [f"W{window_number:04d}" for window_number in window_numbers],
        }
        for batch_id, worker_id, window_numbers in groups
    ]


def default_workload(version: int = 1) -> dict[str, Any]:
    if version not in (1, 2, 3):
        raise ValueError("version must be 1, 2, or 3")
    workers = [
        {
            "worker_id": "W01",
            "host": "127.0.0.1",
            "port": DEFAULT_PORTS[0],
            "model_dim": 16,
            "hidden_dim": 12,
            "seed": 7101,
            "parameter_version": 1,
            "max_concurrent": 1,
            "max_worker_leases": 2,
            "lease_ms": 2500,
        },
        {
            "worker_id": "W02",
            "host": "127.0.0.1",
            "port": DEFAULT_PORTS[1],
            "model_dim": 16,
            "hidden_dim": 12,
            "seed": 7201,
            "parameter_version": 1,
            "max_concurrent": 1,
            "max_worker_leases": 2,
            "lease_ms": 2500,
        },
    ]
    quantization = {"scheme": "symmetric_int", "fractional_bits": 15}
    batch_size = 5
    if version == 2:
        workers[0]["parameter_version"] = 2
    elif version == 3:
        workers[0]["parameter_version"] = 2
        workers[1]["parameter_version"] = 2
        quantization = {"scheme": "symmetric_int", "fractional_bits": 14}
        batch_size = 4
    return {
        "spec_version": 1,
        "session_version": version,
        "seed": 20261125,
        "audio": {
            "duration_ms": 7400,
            "sample_rate": 16000,
            "channels": 1,
            "sample_width_bits": 16,
            "speaker_count": 3,
            "pause_ms": [55, 130],
            "overlap_bps": 1500,
        },
        "window": {"window_ms": 400, "hop_ms": 200, "overlap_bps": 5000},
        "batch": {"batch_size": batch_size},
        "quantization": quantization,
        "workers": copy.deepcopy(workers),
        "requests": _request_groups(),
        "limits": {
            "global_batch_leases": 3,
            "metric_scale": 10000,
            "batch_score_error_bound_units": 2,
            "worker_watermark_error_bound_units": 2,
            "handshake_timeout_ms": 3000,
            "io_timeout_ms": 5000,
        },
    }


def _json_pretty(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def prepare_samples(root: str | Path, *, overwrite: bool = True) -> dict[str, Any]:
    sample_root = Path(root)
    sample_root.mkdir(parents=True, exist_ok=True)
    receipts_dir = sample_root / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, Any]] = []

    def write(relative: str, data: bytes) -> Path:
        path = sample_root / relative
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        atomic_write_bytes(path, data)
        generated.append({"path": relative, "bytes": len(data), "sha256": sha256_hex(data)})
        return path

    workloads: dict[int, WorkloadSpec] = {}
    for version in (1, 2, 3):
        raw = default_workload(version)
        text = json.dumps(raw, ensure_ascii=False, indent=2) + "\n"
        spec = load_workload_data(source_text=text, source_name=str(sample_root / f"workload{'' if version == 1 else '_v' + str(version)}.json"))
        workloads[version] = spec
        filename = "workload.json" if version == 1 else f"workload_v{version}.json"
        write(filename, text.encode("utf-8"))

    base = default_workload(1)
    write(
        "seed.json",
        _json_pretty(
            {
                "schema": "speaker-diag-seed-sample-v1",
                "master_seed": base["seed"],
                "worker_seeds": {worker["worker_id"]: worker["seed"] for worker in base["workers"]},
            }
        ),
    )
    write(
        "audio.json",
        _json_pretty({"schema": "speaker-diag-audio-sample-v1", "seed": base["seed"], **base["audio"]}),
    )
    write("window.json", _json_pretty({"schema": "speaker-diag-window-sample-v1", **base["window"]}))
    write("workers.json", _json_pretty({"schema": "speaker-diag-workers-sample-v1", "workers": base["workers"]}))
    write("requests.json", _json_pretty({"schema": "speaker-diag-requests-sample-v1", "requests": base["requests"]}))

    spec = workloads[1]
    bundle = synthesize(spec)
    raw = infer_local_batch(spec, bundle, spec.requests[0], elapsed_ms=7)
    context = context_for(spec, "B001")
    r3 = raw.receipt
    r2 = strip_to_r2(r3)
    r1 = strip_to_r1(r3)
    for name, receipt in (("r1_B001.json", r1), ("r2_B001.json", r2), ("r3_B001.json", r3)):
        write(f"receipts/{name}", _json_pretty(receipt))
    normalized = normalize_receipt(r3, context=context)

    readme = f"""# Deterministic speaker-diag samples

All files below are generated by `python -m speaker_diag prepare` and contain no
wall-clock timestamps or external references.

- `workload.json`, `workload_v2.json`, `workload_v3.json`: complete, directly
  loadable declarations. Version 2 changes W01's parameter version. Version 3
  changes both workers plus the global quantization width and batch size.
- `seed.json`, `audio.json`, `window.json`, `workers.json`, `requests.json`:
  human-reviewable mirrors of the base declaration.
- `receipts/r1_B001.json`, `r2_B001.json`, `r3_B001.json`: three historical
  receipt encodings of the same B001 inference. Their normalized attribution
  digest is `{normalized.attribution_digest}`.

Declared loopback ports are {DEFAULT_PORTS[0]} and {DEFAULT_PORTS[1]}. Always
run the port probe before starting workers:

```text
python -m speaker_diag probe-ports --config samples/workload.json
```

The expected base declarations produce {len(spec.window_ids)} acoustic windows.
"""
    write("README.md", readme.encode("utf-8"))

    manifest = {
        "schema": "speaker-diag-sample-manifest-v1",
        "generator": "speaker_diag.samples.prepare_samples",
        "versions": {
            str(version): {
                "spec_digest": workloads[version].spec_digest,
                "session_version": workloads[version].session_version,
                "parameter_versions": {
                    worker.worker_id: worker.parameter_version for worker in workloads[version].workers
                },
                "window_count": len(workloads[version].window_ids),
            }
            for version in (1, 2, 3)
        },
        "base_window_digest": bundle.window_digest,
        "base_truth_digest": bundle.truth_digest,
        "base_feature_digest": bundle.feature_digest,
        "base_pcm_digest": bundle.pcm_digest,
        "normalized_r1_r2_r3_attribution_digest": normalized.attribution_digest,
        "files": generated,
        "manifest_digest": canonical_sha256(
            {
                "versions": {
                    str(version): workloads[version].spec_digest for version in (1, 2, 3)
                },
                "files": generated,
            }
        ),
    }
    atomic_write_json(sample_root / "manifest.json", manifest)
    return manifest


def sample_paths(root: str | Path) -> dict[str, Path]:
    sample_root = Path(root)
    return {
        "root": sample_root,
        "v1": sample_root / "workload.json",
        "v2": sample_root / "workload_v2.json",
        "v3": sample_root / "workload_v3.json",
        "r1": sample_root / "receipts" / "r1_B001.json",
        "r2": sample_root / "receipts" / "r2_B001.json",
        "r3": sample_root / "receipts" / "r3_B001.json",
        "manifest": sample_root / "manifest.json",
    }
