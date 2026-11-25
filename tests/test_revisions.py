from __future__ import annotations

import copy

from gsb_index.codes import (
    E_RECEIPT_FUTURE_REVISION,
    E_RECEIPT_STALE_COMPATIBLE,
    E_RECEIPT_STALE_INCOMPATIBLE,
)
from gsb_index.config import parse_project
from gsb_index.edits import project_to_config
from gsb_index.revisions import RevisionChain


def _project_with_revisions(sample_project, revisions, current):
    raw = copy.deepcopy(project_to_config(sample_project))
    for shard in raw["layout"]["shards"]:
        shard["term_bucket_capacity"] = 2520
    raw["revisions"] = {"revisions": revisions, "current_revision": current}
    for revision in revisions:
        if revision["version"] == current:
            raw["compression"] = {
                "bucket_count": revision.get("bucket_count", 4),
                "bucket_weights": revision.get("bucket_weights", [1.0, 2.0, 3.0, 4.0]),
                "encoding": revision.get("encoding", "plain-int-v1"),
            }
    project, issues = parse_project(raw)
    assert issues == []
    return project


def test_revision_chain_applies_weight_bucket_and_encoding_changes(sample_project):
    revisions = [
        {"version": 1, "parent": None, "bucket_count": 4, "bucket_weights": [1.0, 2.0, 3.0, 4.0], "encoding": "plain-int-v1"},
        {"version": 2, "parent": 1, "weight_updates": {"0": 5.0}},
        {"version": 3, "parent": 2, "add_buckets": [{"weight": 6.0}, 7.0]},
        {"version": 4, "parent": 3, "remove_buckets": [1], "encoding": "delta-int-v1"},
    ]
    project = _project_with_revisions(sample_project, revisions, 4)
    chain = RevisionChain(project.revisions, project.current_revision)
    profile = chain.profile(4)
    assert profile.bucket_count == 5
    assert profile.encoding == "delta-int-v1"
    assert profile.bucket_weights[0] == 5.0
    assert chain.ancestry(4) == (1, 2, 3, 4)


def test_stale_receipt_is_accepted_when_referenced_domain_is_unchanged(sample_project):
    revisions = [
        {"version": 1, "parent": None, "bucket_count": 4, "bucket_weights": [1.0, 2.0, 3.0, 4.0], "encoding": "plain-int-v1"},
        {"version": 2, "parent": 1, "changes": {"reason": "metadata-only"}},
    ]
    project = _project_with_revisions(sample_project, revisions, 2)
    chain = RevisionChain(project.revisions, project.current_revision)
    result = chain.arbitrate(1, [0, 1, 2, 3], receipt_profile=chain.profile(1))
    assert result.decision == "accepted_compatible"
    assert result.code == E_RECEIPT_STALE_COMPATIBLE
    assert result.covered_by == (2,)


def test_stale_receipt_is_retained_but_rejected_when_bucket_domain_changes(sample_project):
    revisions = [
        {"version": 1, "parent": None, "bucket_count": 4, "bucket_weights": [1.0, 2.0, 3.0, 4.0], "encoding": "plain-int-v1"},
        {"version": 2, "parent": 1, "weight_updates": {"1": 9.0}},
    ]
    project = _project_with_revisions(sample_project, revisions, 2)
    chain = RevisionChain(project.revisions, project.current_revision)
    result = chain.arbitrate(1, [0, 1, 2, 3], receipt_profile=chain.profile(1))
    assert result.decision == "rejected"
    assert result.code == E_RECEIPT_STALE_INCOMPATIBLE
    assert "referenced_bucket_weight_changed" in result.basis
    assert result.covered_by == (2,)


def test_rollback_makes_a_higher_receipt_a_future_revision(sample_project):
    revisions = [
        {"version": 1, "parent": None, "bucket_count": 4, "bucket_weights": [1.0, 2.0, 3.0, 4.0], "encoding": "plain-int-v1"},
        {"version": 2, "parent": 1, "encoding": "delta-int-v1"},
        {"version": 3, "parent": 2, "bucket_count": 8, "bucket_weights": [1.0] * 8, "encoding": "bucket-fixed-v1"},
    ]
    project = _project_with_revisions(sample_project, revisions, 1)
    chain = RevisionChain(project.revisions, project.current_revision)
    result = chain.arbitrate(3, [0], receipt_profile=chain.profile(3))
    assert result.decision == "rejected"
    assert result.code == E_RECEIPT_FUTURE_REVISION
    assert result.covered_by == (2, 3)
