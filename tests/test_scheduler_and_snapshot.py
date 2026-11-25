from __future__ import annotations

from pathlib import Path

from search_proto.engine import SearchEngine
from search_proto.errors import E_POINT_BUDGET_PENDING, E_QUERY_BUDGET_EXCEEDED
from search_proto.model import Document, QuerySpec
from search_proto.snapshot import SnapshotStore

from .conftest import make_config, make_stopwords, make_vocab


def _docs(count: int = 4) -> list[Document]:
    return [
        Document.create(f"D{index}", 0, {"title": ["alpha"], "body": ["beta"]})
        for index in range(1, count + 1)
    ]


def test_batch_budget_pending_and_next_batch_progress() -> None:
    config = make_config(
        top_k=2,
        batch_point_budget=3,
        per_query_point_quota=100,
    )
    queries = [
        QuerySpec.create("Q_A", ["alpha", "beta", "gamma"], ["title", "body"]),
        QuerySpec.create("Q_B", ["alpha"], ["title", "body"]),
    ]
    engine = SearchEngine(config, make_stopwords(), make_vocab(), _docs(), queries)
    engine.materialize()
    engine.run_batches(max_batches=1)
    assert engine.pending_query_ids == ["Q_A", "Q_B"]
    assert engine.query_progress["Q_A"]["consumed_points"] == 3
    assert any(
        failure.code == E_POINT_BUDGET_PENDING for failure in engine.failures
    )
    first_batch = engine.batch_consumed_points[0]
    assert first_batch == 3

    engine.run_batches()
    assert engine.completed_query_ids == ["Q_A", "Q_B"]
    assert engine.pending_query_ids == []
    assert all(
        points <= config.batch_point_budget
        for points in engine.batch_consumed_points.values()
    )
    assert engine.query_progress["Q_A"]["consumed_points"] == 12
    assert engine.query_progress["Q_B"]["consumed_points"] == 4


def test_per_query_quota_aborts_with_business_code() -> None:
    config = make_config(
        top_k=2,
        batch_point_budget=100,
        per_query_point_quota=5,
    )
    query = QuerySpec.create("Q", ["alpha", "beta"], ["title", "body"])
    engine = SearchEngine(
        config, make_stopwords(), make_vocab(), _docs(), [query]
    )
    engine.materialize()
    engine.run_batches()
    assert engine.query_progress["Q"]["consumed_points"] == 4
    assert engine.aborted_query_ids == ["Q"]
    assert "Q" not in engine.query_results
    assert any(
        failure.code == E_QUERY_BUDGET_EXCEEDED for failure in engine.failures
    )
    assert engine.report()["business_code_counts"][E_QUERY_BUDGET_EXCEEDED] == 1


def test_impossible_batch_budget_stalls_without_spinning() -> None:
    config = make_config(
        top_k=2,
        batch_point_budget=2,
        per_query_point_quota=100,
    )
    query = QuerySpec.create("Q", ["alpha", "beta", "gamma"], ["title", "body"])
    engine = SearchEngine(
        config, make_stopwords(), make_vocab(), _docs(), [query]
    )
    engine.materialize()
    engine.run_batches()
    assert engine.scheduler_stalled is True
    assert engine.pending_query_ids == ["Q"]
    assert engine.batch_index == 1
    assert engine.batch_consumed_points == {0: 0}
    assert engine.report()["scheduler_stall_reason"] == (
        "batch_point_budget_smaller_than_remaining_query_work"
    )


def test_snapshot_corruption_falls_back_to_last_good(workspace_tmp: Path) -> None:
    config = make_config(top_k=2)
    engine = SearchEngine(
        config,
        make_stopwords(),
        make_vocab(),
        _docs(2),
        [QuerySpec.create("Q", ["alpha"], ["title", "body"])],
    )
    engine.materialize()
    engine.run_batches()
    engine.global_version = 1
    store = SnapshotStore(workspace_tmp / "snapshots")
    first = store.commit(engine)
    engine.global_version = 2
    second = store.commit(engine)
    assert first["version_name"] != second["version_name"]

    current = store._read_pointer(store.current_path)
    assert current == second["version_name"]
    (store.versions_dir / current / "state.json").write_text(
        "deliberately corrupt\n", encoding="utf-8"
    )
    recovered = store.load_best()
    assert recovered["status"] == "recovered_previous"
    assert recovered["version_name"] == first["version_name"]
    assert recovered["manifest"]["global_version"] == 1
