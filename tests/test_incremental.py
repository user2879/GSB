from __future__ import annotations

import copy

import pytest

from vscan.errors import DivergenceError
from vscan.session import Session
from vscan.spec import parse_spec


def _two_segment_spec_dict(base):
    data = copy.deepcopy(base)
    second = copy.deepcopy(data["segments"][0])
    second["id"] = "s2"
    second["seed"] = 99
    data["segments"].append(second)
    return data


def _dictionary_by_batch(report):
    return {
        (item["query_id"], item["segment_id"], item["batch_no"]): item
        for item in report["batch_results"]
    }


def test_single_value_edit_recomputes_only_containing_batch(tmp_path, spec_dict):
    data = _two_segment_spec_dict(spec_dict)
    data["queries"].append(
        {
            "id": "q_group_only",
            "project": ["g"],
            "predicate": None,
            "group_by": ["g"],
            "aggregates": [{"name": "n", "op": "count"}],
        }
    )
    spec = parse_spec(data)
    with Session(spec, tmp_path / "data", tmp_path / "ledger.sqlite") as session:
        first = session.run()
        before = _dictionary_by_batch(first)
        old_group_query = first["queries"][1]
        old_other_batches = {
            key: value["aggregate_digest"]
            for key, value in before.items()
            if key[0] == "q1" and not (key[1] == "s1" and key[2] == 1)
        }
        old_columns = {
            (segment.id, column.name): column.column_digest
            for segment in session.store.segments
            for column in segment.columns
        }

        report = session.edit_column_value("s1", "v", 5, 777)
        after = _dictionary_by_batch(report)
        assert report["invalidation"]["invalid_batch_count"] == 1
        assert report["incremental"]["recomputed_batch_count"] == 1
        assert report["incremental"]["recomputed_batches"][0] == {
            "query_id": "q1",
            "segment_id": "s1",
            "batch_no": 1,
            "start_row": 4,
            "end_row": 8,
        }
        assert report["queries"][1]["aggregate_digest"] == old_group_query["aggregate_digest"]
        for key, digest in old_other_batches.items():
            assert after[key]["aggregate_digest"] == digest
        new_columns = {
            (segment.id, column.name): column.column_digest
            for segment in session.store.segments
            for column in segment.columns
        }
        for key, digest in old_columns.items():
            if key != ("s1", "v"):
                assert new_columns[key] == digest
        assert new_columns[("s1", "v")] != old_columns[("s1", "v")]
        assert session.verify_against_full_rebuild()["consistent"] is True


def test_unreferenced_column_edit_recomputes_no_batches(tmp_path, spec_dict):
    spec = parse_spec(copy.deepcopy(spec_dict))
    with Session(spec, tmp_path / "data", tmp_path / "ledger.sqlite") as session:
        session.run()
        report = session.edit_column_value("s1", "unused", 0, 12345)
        assert report["invalidation"]["column_invalidations"]
        assert report["incremental"]["recomputed_batch_count"] == 0
        assert report["incremental"]["reused_batch_count"] == 3
        assert session.verify_against_full_rebuild()["consistent"] is True


def test_projection_and_predicate_edits_affect_only_target_query(tmp_path, spec_dict):
    data = _two_segment_spec_dict(spec_dict)
    data["queries"].append(
        {
            "id": "q_other",
            "project": ["r"],
            "predicate": None,
            "group_by": ["r"],
            "aggregates": [{"name": "n", "op": "count"}],
        }
    )
    spec = parse_spec(data)
    with Session(spec, tmp_path / "data", tmp_path / "ledger.sqlite") as session:
        baseline = session.run()
        other_digest = {query["query_id"]: query["aggregate_digest"] for query in baseline["queries"]}
        projection = session.edit_projection("q1", ["g", "v", "r"])
        # Same sorted set is a no-op, so no batches should be invalidated.
        assert projection["invalidation"]["invalid_batch_count"] == 0
        projection = session.edit_projection("q1", ["g", "v"])
        assert all(
            item["query_id"] == "q1" for item in projection["incremental"]["recomputed_batches"]
        )
        assert projection["queries"][1]["aggregate_digest"] == other_digest["q_other"]
        predicate = session.edit_predicate(
            "q1", {"op": "gt", "column": "v", "value": 10}
        )
        assert all(
            item["query_id"] == "q1" for item in predicate["incremental"]["recomputed_batches"]
        )
        assert predicate["queries"][1]["aggregate_digest"] == other_digest["q_other"]
        assert session.verify_against_full_rebuild()["consistent"] is True


def test_batch_rows_edit_invalidates_all_batches(tmp_path, spec_dict):
    spec = parse_spec(copy.deepcopy(spec_dict))
    with Session(spec, tmp_path / "data", tmp_path / "ledger.sqlite") as session:
        session.run()
        report = session.set_batch_rows(5)
        assert report["incremental"]["recomputed_batch_count"] == 3
        assert report["incremental"]["reused_batch_count"] == 0
        assert [item["start_row"] for item in report["batch_results"]] == [0, 5, 10]
        assert session.verify_against_full_rebuild()["consistent"] is True


def test_divergence_reports_segment_column_and_batch(tmp_path, spec_dict):
    spec = parse_spec(copy.deepcopy(spec_dict))
    with Session(spec, tmp_path / "data", tmp_path / "ledger.sqlite") as session:
        session.run()
        first = session.results[sorted(session.results)[0]]
        first.computation.passed_rows += 1
        with pytest.raises(DivergenceError) as caught:
            session.verify_against_full_rebuild()
        assert caught.value.code == "E_DIVERGE"
        assert caught.value.detail["segment_id"] is not None
        assert caught.value.detail["column"] is not None
        assert caught.value.detail["batch_no"] is not None
