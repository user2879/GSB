import copy
import json

import pytest

from ckpt_diag.compute import compute_batch
from ckpt_diag.digest import canonical
from ckpt_diag.errors import DivergeError
from ckpt_diag.invalidation import (
    plan_invalidation, recompute_subset, verify_against_full_rebuild,
    verify_untouched_byte_stable)
from ckpt_diag.samplegen import build_spec_dict
from ckpt_diag.spec import load_spec


def _specs(tmp_path):
    old_dict = build_spec_dict()
    old_path = tmp_path / "old.json"
    old_path.write_text(json.dumps(old_dict))
    old = load_spec(old_path)
    new_dict = copy.deepcopy(old_dict)
    layer = next(l for l in new_dict["shards"][0]["layers"]
                 if l["layer_id"] == "L1")
    layer["scale"] += 1
    new_path = tmp_path / "new.json"
    new_path.write_text(json.dumps(new_dict))
    new = load_spec(new_path)
    return old, new


def _inputs(spec):
    return {
        "shard-alpha": {
            "batch-a-1": {"param_version": 1, "sample_ids": ["s-0", "s-1"],
                          "scope_layers": ["L1"]},
            "batch-a-2": {"param_version": 1, "sample_ids": ["s-2", "s-3"],
                          "scope_layers": ["L2"]},
        },
        "shard-beta": {
            "batch-b-1": {"param_version": 1, "sample_ids": ["s-0"],
                          "scope_layers": ["M0"]},
        },
    }


def test_invalidation_propagates_to_derived_layers_batches_and_cps(tmp_path):
    old, new = _specs(tmp_path)
    inputs = _inputs(old)
    scopes = {sh: {b: tuple(e["scope_layers"]) for b, e in batches.items()}
              for sh, batches in inputs.items()}
    pvs = {sh: {b: e["param_version"] for b, e in batches.items()}
           for sh, batches in inputs.items()}
    report = plan_invalidation(old, new, scopes, pvs)
    assert report.changed_layers == {"shard-alpha": ["L1"]}
    assert set(report.affected_layers["shard-alpha"]) == {"L1", "L2"}
    assert set(report.invalid_batches["shard-alpha"]) == {"batch-a-1", "batch-a-2"}
    assert "batch-b-1" not in report.invalid_batches.get("shard-beta", [])
    kinds = [step.kind for step in report.steps]
    assert "checkpoint" in kinds


def test_untouched_batches_byte_stable_and_subset_matches_full(tmp_path):
    old, new = _specs(tmp_path)
    inputs = _inputs(old)
    scopes = {sh: {b: tuple(e["scope_layers"]) for b, e in batches.items()}
              for sh, batches in inputs.items()}
    pvs = {sh: {b: e["param_version"] for b, e in batches.items()}
           for sh, batches in inputs.items()}
    report = plan_invalidation(old, new, scopes, pvs)
    before = {f"{sh}:{b}": compute_batch(
        old.shard(sh), b, e["param_version"], tuple(e["sample_ids"]),
        tuple(e["scope_layers"]))
        for sh, batches in inputs.items() for b, e in batches.items()}
    subset = recompute_subset(new, report, inputs)
    verify_against_full_rebuild(new, subset, inputs)
    invalid = {b for ids in report.invalid_batches.values() for b in ids}
    after = dict(before)
    for sh, batches in subset.items():
        for b, result in batches.items():
            after[f"{sh}:{b}"] = result
    verify_untouched_byte_stable(
        {b.split(":", 1)[1]: r for b, r in before.items() if b.startswith("shard-beta:")},
        {b.split(":", 1)[1]: r for b, r in after.items() if b.startswith("shard-beta:")},
        invalid)


def test_diverge_is_reported_with_location(tmp_path):
    old, new = _specs(tmp_path)
    inputs = _inputs(old)
    report = plan_invalidation(
        old, new,
        {sh: {b: tuple(e["scope_layers"]) for b, e in batches.items()}
         for sh, batches in inputs.items()},
        {sh: {b: e["param_version"] for b, e in batches.items()}
         for sh, batches in inputs.items()})
    subset = recompute_subset(new, report, inputs)
    # corrupt one subset result to force E_DIVERGE
    shard_id = next(iter(subset))
    batch_id = next(iter(subset[shard_id]))
    corrupted = copy.deepcopy(subset)
    object.__setattr__(corrupted[shard_id][batch_id], "total_loss",
                       corrupted[shard_id][batch_id].total_loss + 1)
    with pytest.raises(DivergeError) as exc:
        verify_against_full_rebuild(new, corrupted, inputs)
    assert exc.value.code == "E_DIVERGE"
