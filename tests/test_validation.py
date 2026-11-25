from __future__ import annotations

import copy
import shutil

from gsb_index.codes import (
    E_BUCKET_CAPACITY_INDIVISIBLE,
    E_DOC_DUP_ID,
    E_DUP_PORT,
    E_DUP_SHARD_ID,
    E_INVALID_FIELD,
    E_INVALID_PORT,
    E_INVALID_STOPWORD_FLAG,
    E_NON_FINITE_WEIGHT,
    E_NON_POSITIVE_BUCKET_CAPACITY,
    E_NON_POSITIVE_CONCURRENCY,
    E_REVISION_NON_MONOTONIC,
    E_UNKNOWN_SHARD,
    E_UNKNOWN_TERM,
)
from gsb_index.config import load_project, parse_project
from gsb_index.edits import project_to_config


def test_validation_accumulates_all_issue_families(sample_project):
    raw = copy.deepcopy(project_to_config(sample_project))

    raw["documents"][1]["doc_id"] = raw["documents"][0]["doc_id"]
    raw["documents"][2]["fields"]["title"][0] = "outside-vocabulary"
    raw["documents"][3]["fields"]["unexpected"] = ["index"]
    raw["documents"][4]["field_stopwords"]["title"] = "yes"
    raw["documents"][5]["shard_id"] = "shard-missing"

    raw["layout"]["shards"].append(copy.deepcopy(raw["layout"]["shards"][0]))
    duplicate_port_shard = copy.deepcopy(raw["layout"]["shards"][1])
    duplicate_port_shard["shard_id"] = "shard-port-clone"
    duplicate_port_shard["port"] = raw["layout"]["shards"][0]["port"]
    raw["layout"]["shards"].append(duplicate_port_shard)
    extra_shards = [
        {"shard_id": "shard-badcap", "port": 19303, "term_bucket_capacity": 0, "concurrency_limit": 1, "seed": 1, "revision": 2},
        {"shard_id": "shard-badconc", "port": 19304, "term_bucket_capacity": 192, "concurrency_limit": -1, "seed": 2, "revision": 2},
        {"shard_id": "shard-indivisible", "port": 19305, "term_bucket_capacity": 193, "concurrency_limit": 1, "seed": 3, "revision": 2},
        {"shard_id": "shard-badport", "port": 70000, "term_bucket_capacity": 192, "concurrency_limit": 1, "seed": 4, "revision": 1},
        {"shard_id": "shard-rev-low", "port": 19306, "term_bucket_capacity": 192, "concurrency_limit": 1, "seed": 5, "revision": 1},
    ]
    raw["layout"]["shards"].extend(extra_shards)

    raw["compression"]["bucket_weights"][0] = float("nan")

    project, issues = parse_project(raw)
    codes = {issue["code"] for issue in issues}
    assert E_DOC_DUP_ID in codes
    assert E_UNKNOWN_TERM in codes
    assert E_INVALID_FIELD in codes
    assert E_INVALID_STOPWORD_FLAG in codes
    assert E_UNKNOWN_SHARD in codes
    assert E_DUP_SHARD_ID in codes
    assert E_DUP_PORT in codes
    assert E_NON_POSITIVE_BUCKET_CAPACITY in codes
    assert E_NON_POSITIVE_CONCURRENCY in codes
    assert E_BUCKET_CAPACITY_INDIVISIBLE in codes
    assert E_INVALID_PORT in codes
    assert E_REVISION_NON_MONOTONIC in codes
    assert E_NON_FINITE_WEIGHT in codes
    assert len(project.documents) > 0
    assert issues == sorted(
        issues,
        key=lambda item: (
            item["code"],
            item.get("object_type", ""),
            item.get("object_id", ""),
            item.get("field", ""),
            item.get("message", ""),
        ),
    )


def test_invalid_shard_does_not_abort_other_shards(sample_project):
    raw = copy.deepcopy(project_to_config(sample_project))
    raw["layout"]["shards"][0]["term_bucket_capacity"] = 193
    project, issues = parse_project(raw)
    assert [shard.shard_id for shard in project.shards] == ["shard-b", "shard-c"]
    assert any(issue["code"] == E_BUCKET_CAPACITY_INDIVISIBLE for issue in issues)
    assert any(issue["code"] == E_UNKNOWN_SHARD for issue in issues)
    assert len(project.documents) > 0


def test_component_directory_can_be_loaded_without_bundle(sample_dir, tmp_path):
    component_dir = tmp_path / "components"
    component_dir.mkdir()
    for name in (
        "vocabulary.json",
        "documents.json",
        "layout.json",
        "compression.json",
        "budget.json",
        "revisions.json",
        "queries.json",
    ):
        shutil.copy2(sample_dir / name, component_dir / name)
    project, issues = load_project(component_dir)
    assert issues == []
    assert len(project.documents) == 18
    assert [shard.shard_id for shard in project.shards] == ["shard-a", "shard-b", "shard-c"]
