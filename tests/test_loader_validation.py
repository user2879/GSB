from __future__ import annotations

import pytest

from fcdiag.errors import InputInvalidError
from fcdiag.loader import load_config, load_pipeline, load_samples


def test_sample_validation_collects_all_violations():
    text = "\n".join(
        [
            '{"sample_id":"a","features":[1,"x",null],"label":2}',
            '{"sample_id":"a","features":[1,2]}',
            '{"sample_id":"b","features":[1,NaN]}',
        ]
    )
    with pytest.raises(InputInvalidError) as exc:
        load_samples(text)
    locators = " ".join(v.locator for v in exc.value.violations)
    messages = " ".join(v.message for v in exc.value.violations)
    assert len(exc.value.violations) >= 4
    assert "features[1]" in locators and "features[2]" in locators
    assert "duplicate sample_id" in messages and "finite" in messages


def test_pipeline_positive_integer_and_dag_validation():
    bad = """
name: bad
feature_count: 2
stages:
  - name: bucketize
    kind: bucketize
    upstream: [missing]
    params:
      bucket_count: 0
  - name: interact
    kind: interact
    upstream: [bucketize]
    params:
      degree: 0
"""
    with pytest.raises(InputInvalidError) as exc:
        load_pipeline(bad)
    messages = " ".join(v.message for v in exc.value.violations)
    assert "unknown upstream" in messages
    assert "bucket_count" in messages
    assert "degree" in messages


def test_pipeline_cycle_is_rejected():
    cyclic = """
name: cyclic
feature_count: 2
stages:
  - name: a
    kind: standardize
    upstream: []
    params: {}
  - name: b
    kind: bucketize
    upstream: [a, c]
    params:
      bucket_count: 2
  - name: c
    kind: interact
    upstream: [b]
    params:
      degree: 1
"""
    with pytest.raises(InputInvalidError) as exc:
        load_pipeline(cyclic)
    assert any("dependency cycle" in v.message for v in exc.value.violations)


def test_config_violations_are_reported():
    with pytest.raises(InputInvalidError) as exc:
        load_config(
            {
                "fixed": {"frac_bits": 0, "gear": 0},
                "budget": {"batch_points": 0, "stage_quotas": {"x": -1}},
                "policy": {"degraded_reuse": "unknown"},
                "seed": "nope",
            }
        )
    assert len(exc.value.violations) >= 6
