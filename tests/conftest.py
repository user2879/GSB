from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest


DATA_SPEC = """
name: compact_probe
seed: 991
target: target
columns:
  - name: row_id
    type: id
    prefix: row
  - name: amount
    type: float
    distribution: normal
    mean: 1000.0
    std: 120.0
    missing_rate: 0.04
    decimals: 6
  - name: score
    type: float
    distribution: normal
    mean: 0.0
    std: 1.0
    missing_rate: 0.02
    decimals: 6
  - name: tenure
    type: int
    min: 1
    max: 48
    missing_rate: 0.0
  - name: region
    type: category_low
    values: [north, south, west]
    weights: [0.45, 0.35, 0.20]
    missing_rate: 0.02
  - name: channel
    type: category_low
    values: [web, store, app]
    weights: [0.5, 0.3, 0.2]
    missing_rate: 0.02
  - name: user_id
    type: category_high
    cardinality: 17
    prefix: u
    missing_rate: 0.03
  - name: amount_copy
    type: redundant
    source: amount
    noise_std: 0.0001
    missing_rate: 0.01
    decimals: 6
  - name: target
    type: target
    noise_std: 0.2
    formula:
      - column: amount
        weight: 0.001
      - column: score
        weight: 0.3
      - column: tenure
        weight: 0.01
      - column: region
        weight: 0.1
      - column: channel
        weight: 0.08
"""


@pytest.fixture
def make_experiment(tmp_path: Path) -> Callable[..., Path]:
    data_path = tmp_path / "data_spec.yaml"
    data_path.write_text(DATA_SPEC, encoding="utf-8")

    def factory(
        *,
        version: int = 1,
        pca_components: int = 3,
        max_workers: int = 3,
        cache_dir: str = "cache",
        snapshot_dir: str = "snapshot",
        sample_sizes: tuple[int, int] = (64, 96),
        decision_threshold_mode: str = "reference_median",
    ) -> Path:
        size_literal = ", ".join(str(value) for value in sample_sizes)
        config_path = tmp_path / f"experiment-v{version}-p{pca_components}-{snapshot_dir.replace('/', '_')}.yaml"
        config_path.write_text(
            f"""
version: {version}
engine:
  schema_version: 1
seed: 20261120
data_spec: {data_path.name}
tolerances:
  cross_encoding: 0.000001
  standardization: 0.00001
  projection: 0.0001
  final_metric: 0.00001
boundary:
  neighborhood: 0.05
  tie_tolerance: 0.0000001
cache_dir: {cache_dir}
snapshot_dir: {snapshot_dir}
max_workers: {max_workers}
scenarios:
  - id: compact
    sample_sizes: [{size_literal}]
    cross_orders: [1, 2]
    candidate_columns: [region, channel, user_id]
    cardinality_budget: 8
    overflow_policy: hash_bucket
    precision_tiers: [float32, float64]
    reductions:
      pca:
        components: {pca_components}
      random_projection:
        components: 3
      variance_filter:
        components: 3
        min_variance: 0.0
    decision:
      metric: first_coordinate
      threshold_mode: {decision_threshold_mode}
      threshold: 0.0
    max_workers: {max_workers}
""",
            encoding="utf-8",
        )
        return config_path

    return factory
