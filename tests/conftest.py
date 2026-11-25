from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


@pytest.fixture
def examples_dir() -> Path:
    return EXAMPLES


@pytest.fixture
def base_config() -> dict:
    return yaml.safe_load((EXAMPLES / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def base_edits() -> dict:
    return yaml.safe_load((EXAMPLES / "edits.yaml").read_text(encoding="utf-8"))


def write_variant(
    directory: Path,
    config: dict,
    edits: dict | None = None,
    *,
    lease_ttl_ms: int | None = None,
    max_workers: int | None = None,
    bandwidth: str | None = None,
    fractional_bits: int | None = None,
    quantization_units: int | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    value = copy.deepcopy(config)
    value["waveform_path"] = str((EXAMPLES / "waveform.yaml").resolve())
    value["lease_declaration_path"] = str((EXAMPLES / "lease.yaml").resolve())
    if edits is None:
        value.pop("edits_path", None)
    else:
        for command in edits.get("commands", []):
            if command.get("op") == "set_fractional_bits" and fractional_bits is not None:
                command["value"] = fractional_bits
            if command.get("op") == "set_quantization_units" and quantization_units is not None:
                command["value"] = quantization_units
            if command.get("op") == "set_kernel_bandwidth" and bandwidth is not None:
                command["value"] = bandwidth
        edits_path = directory / "edits.yaml"
        edits_path.write_text(yaml.safe_dump(edits, sort_keys=False), encoding="utf-8")
        value["edits_path"] = "edits.yaml"
    if lease_ttl_ms is not None:
        lease_path = directory / "lease.yaml"
        lease_path.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "worker_id": "ins-diag-test",
                    "ttl_ms": lease_ttl_ms,
                    "restore_generation": 0,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        value["lease_declaration_path"] = "lease.yaml"
    if max_workers is not None:
        value["runtime"] = dict(value.get("runtime", {}))
        value["runtime"]["max_workers"] = max_workers
    if bandwidth is not None:
        value.setdefault("kde", {})["bandwidth"] = bandwidth
    if fractional_bits is not None or quantization_units is not None:
        fixed = value.setdefault("clustering", {}).setdefault("fixed_point", {})
        if fractional_bits is not None:
            fixed["fractional_bits"] = fractional_bits
        if quantization_units is not None:
            fixed["quantization_units"] = quantization_units
    value.setdefault("runtime", {})["state_dir"] = "state"
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def cluster_projection(report: dict) -> dict:
    clusters = report.get("clusters") or {}
    density = report.get("density") or {}
    return {
        "windows": [
            {
                key: window.get(key)
                for key in (
                    "window_index",
                    "window_id",
                    "raw_features",
                    "fill_source",
                    "standardized_vector",
                    "standardized_fixed_units",
                    "cluster_id",
                    "cluster_distance_units",
                    "density",
                )
            }
            for window in report.get("windows", [])
        ],
        "standardization": report.get("standardization"),
        "clusters": {
            key: clusters.get(key)
            for key in (
                "fixed_point",
                "initial_centroid_window_indices",
                "centroids_units",
                "assignments",
                "cluster_members",
                "cluster_sse_units",
                "sse_units",
                "rounds",
                "result_fingerprint",
            )
        },
        "density": {
            key: density.get(key)
            for key in (
                "bandwidth",
                "maximum_log_density",
                "window_scores",
                "boundary_windows",
                "baseline_fingerprint",
            )
        },
        "boundary_windows": report.get("boundary_windows"),
        "failures": report.get("failures"),
    }


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

