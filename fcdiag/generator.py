"""Deterministic fixture generation from fixed integer seeds.

The generator deliberately uses a small LCG instead of a global NumPy RNG so
fixture generation remains stable even when NumPy version/default RNG changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from .fingerprint import canonical_json
from .models import ReceiptRecord
from .pipeline import FixedPipeline, artifact_id, cache_key, raw_artifact_id
from .receipts import PRESENT_BY_VERSION, render_receipts
from .state import WorkState


class Lcg:
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFF
        if self.state == 0:
            self.state = 1

    def next_int(self, low: int, high: int) -> int:
        # Numerical Recipes constants; every operation is integer exact.
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        span = high - low + 1
        return low + (self.state % span)

    def next_float(self, low: float, high: float, scale: int = 1000) -> float:
        value = self.next_int(int(low * scale), int(high * scale))
        return value / scale


PIPELINE_YAML = """name: layered_feature_cache_demo
feature_count: 4
stages:
  - name: standardize
    kind: standardize
    upstream: []
    params: {}
  - name: bucketize
    kind: bucketize
    upstream:
      - standardize
    params:
      bucket_count: 5
  - name: interact
    kind: interact
    upstream:
      - standardize
    params:
      degree: 2
  - name: train_view
    kind: train_view
    upstream:
      - interact
    params:
      weights:
        - 1.0
        - 0.5
        - 0.25
      threshold: 0.0
"""


CONFIG_DOC = {
    "seed": 20261125,
    "fixed": {"frac_bits": 4, "gear": 1},
    "budget": {
        "batch_points": 50,
        "stage_quotas": {
            "standardize": 100,
            "bucketize": 100,
            "interact": 100,
            "train_view": 20,
        },
    },
    "policy": {"degraded_reuse": "param_version"},
}


def generate_samples(seed: int = 20261125, count: int = 8, feature_count: int = 4) -> Tuple[dict, ...]:
    rng = Lcg(seed)
    samples = []
    for index in range(count):
        sample_id = f"s{index:03d}"
        features = [rng.next_float(-3.0, 3.0) for _ in range(feature_count)]
        samples.append({"sample_id": sample_id, "features": features, "label": rng.next_int(0, 1)})
    return tuple(samples)


def generate_commands() -> Tuple[dict, ...]:
    return (
        {"op": "replace_sample", "sample": {"sample_id": "s002", "features": [1.25, -0.5, 0.75, 2.0], "label": 1}},
        {"op": "append_sample", "sample": {"sample_id": "s008", "features": [0.1, -0.2, 0.3, -0.4], "label": 0}},
        {"op": "set_bucket_count", "stage": "bucketize", "value": 7},
        {"op": "set_interaction_degree", "stage": "interact", "value": 3},
        {"op": "set_threshold", "stage": "train_view", "value": 0.25},
        {"op": "set_weight", "stage": "train_view", "index": 1, "value": 0.75},
        {"op": "set_gear", "stage": "*", "value": 5},
        {"op": "set_frac_bits", "stage": "*", "value": 6},
        {"op": "set_receipt_version", "stage": "*", "value": 2},
        {"op": "set_receipt_version", "stage": "*", "value": 1},
        {"op": "set_receipt_version", "stage": "*", "value": 3},
    )


def generate_injection_doc() -> dict:
    return {
        "injections": [
            {"id": "INF-001", "target": "pipeline.yaml:train_view.params.weights[0]", "payload": "NaN"},
            {"id": "INF-002", "target": "pipeline.yaml:bucketize.params.bucket_count", "payload": 0},
            {"id": "INF-003", "target": "pipeline.yaml:interact.upstream", "payload": "missing_stage"},
            {"id": "INF-004", "target": "receipts/v3.txt:L2", "payload": "remove-last-char"},
            {"id": "INF-005", "target": "receipts/v2.txt", "payload": "duplicate-artifact-id"},
        ]
    }


def build_baseline_receipts(state: WorkState) -> Tuple[ReceiptRecord, ...]:
    """Compute initial artifacts and create complete v3 receipts for them."""

    computer = FixedPipeline(state.pipeline, state.config.fixed)
    raw = computer.raw_artifacts(state.samples)
    artifacts_by_id = {item.artifact_id: item for item in raw}
    records: List[ReceiptRecord] = []
    line_no = 2
    for artifact in raw:
        records.append(
            ReceiptRecord(
                source_version=3,
                wire_version=3,
                line_no=line_no,
                artifact_id=artifact.artifact_id,
                hit_permille=900 + (line_no % 101),
                present=dict(PRESENT_BY_VERSION[3]),
                param_version=0,
                elapsed_ms=10 + line_no,
                digest=artifact.digest,
                duplicate_refs=(),
            )
        )
        line_no += 1
    for stage in state.pipeline.stages:
        result = computer.compute_stage(
            stage, state.samples, artifacts_by_id, state.param_versions
        )
        for artifact in result.artifacts:
            artifacts_by_id[artifact.artifact_id] = artifact
            refs = tuple(
                upstream for upstream in artifact.upstream if upstream.startswith(("int", "bkt"))
            )[:1]
            records.append(
                ReceiptRecord(
                    source_version=3,
                    wire_version=3,
                    line_no=line_no,
                    artifact_id=artifact.artifact_id,
                    hit_permille=700 + (line_no % 201),
                    present=dict(PRESENT_BY_VERSION[3]),
                    param_version=sum(artifact.param_versions.values()),
                    elapsed_ms=20 + line_no,
                    digest=artifact.digest,
                    duplicate_refs=refs,
                )
            )
            line_no += 1
    return tuple(records)


def write_fixtures(output_dir: str) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples.jsonl").write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in generate_samples()) + "\n",
        encoding="utf-8",
    )
    (out / "pipeline.yaml").write_text(PIPELINE_YAML, encoding="utf-8")
    (out / "config.json").write_text(canonical_json(CONFIG_DOC) + "\n", encoding="utf-8")
    (out / "commands.jsonl").write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in generate_commands()) + "\n",
        encoding="utf-8",
    )
    (out / "injections.json").write_text(
        canonical_json(generate_injection_doc()) + "\n", encoding="utf-8"
    )
    return {path.name: str(path) for path in sorted(out.iterdir()) if path.is_file()}
