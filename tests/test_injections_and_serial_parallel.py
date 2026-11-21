from __future__ import annotations

import json
import shutil
from pathlib import Path

from .conftest import Workspace


def test_all_injection_types_are_hit_and_mapped(workspace: Workspace) -> None:
    report = workspace.run(
        mode="full", output_name="injection-output", injections="injections.demo.json"
    )
    hits = report["injection_results"]["hits"]
    assert len(hits) == 5
    assert report["injection_results"]["missed_count"] == 0
    by_id = {item["injection_id"]: item for item in hits}
    assert by_id["INJ-PNG-TRUNCATED"]["code"] == "E_PNG_TRUNCATED"
    assert by_id["INJ-ASSERTION-FIELD"]["code"] == "E_TEXT_SCHEMA"
    assert by_id["INJ-FEATURE-NONFINITE"]["code"] == "E_FEATURE_NONFINITE"
    assert by_id["INJ-THRESHOLD-RANGE"]["code"] == "E_CONFIG_THRESHOLD"
    assert by_id["INJ-CHUNK-INDIVISIBLE"]["code"] == "E_CHUNK_MISMATCH"
    failed_samples = {item["sample"] for item in report["failures"]}
    assert failed_samples == {"sample-03", "sample-04", "sample-05", "sample-06"}
    assert report["threshold_injection"]["injected_value"] == 1.5
    assert report["threshold_injection"]["fallback_threshold"] == workspace.load_config().similarity_threshold
    assert report["retrieval_quality"]["queries"] == 4
    assert report["retrieval_quality"]["self_hit_at_k_rate"] == 1.0


def test_thread_count_does_not_change_semantic_output(
    workspace: Workspace, tmp_path: Path
) -> None:
    config = workspace.load_config()
    serial_config = type(config).from_mapping({**config.raw, "thread_pool_size": 1})
    parallel_config = type(config).from_mapping({**config.raw, "thread_pool_size": 8})
    serial = workspace.run(
        mode="full", output_name="serial-output", config=serial_config
    )
    parallel = workspace.run(
        mode="full", output_name="parallel-output", config=parallel_config
    )
    assert serial["reproducibility_sha256"] == parallel["reproducibility_sha256"]
    assert serial["retrieval_index"]["queries"] == parallel["retrieval_index"]["queries"]
    assert serial["failures"] == parallel["failures"]
    assert serial["samples"] == parallel["samples"]
