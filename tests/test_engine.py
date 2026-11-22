from __future__ import annotations

import json
from pathlib import Path

from iidx.config import load_config
from iidx.editing import update_document
from iidx.engine import Engine
from iidx.indexer import canonical_projection
from iidx.loader import discover_inputs
from iidx.util import first_difference


def _run(data: Path, state: Path, config, failures=None):
    return Engine(data_dir=data, state_dir=state, config=config, config_failures=failures).run()


def test_repeat_run_is_deterministic_and_reuses_every_block(sample_dir: Path, tmp_path: Path, sample_config) -> None:
    state = tmp_path / "state"
    first = _run(sample_dir, state, sample_config.config, sample_config.failures)
    second = _run(sample_dir, state, sample_config.config, sample_config.failures)
    assert first.report["canonical_digest"] == second.report["canonical_digest"]
    assert second.report["reuse"]["counts"]["reused_blocks"] == len(second.report["blocks"])
    assert second.report["diagnostics"]["cache"]["encoding_calls"] == 0
    assert all(second.report["diagnostics"]["stage_reuse"].values())
    assert all(block["compression"]["lengths_differ"] for block in second.report["blocks"])
    assert all(block["compression"]["decode_ok"] for block in second.report["blocks"])
    assert all(query["result_checksum"] for query in second.report["queries"])
    assert second.snapshot_path is not None and second.snapshot_path.is_dir()
    manifest = json.loads((second.snapshot_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["global_version"] == second.report["run"]["global_version"]
    assert manifest["config_hash"] == second.report["run"]["config_hash"]
    assert manifest["completed_shards"] == second.report["summary"]["completed_shards"]
    assert manifest["consumed_token_budget"] == second.report["summary"]["consumed_token_budget"]
    assert set(manifest["stage_versions"]) == {"loading", "build", "compress", "query"}
    for stage in ("stage-00-loading", "stage-01-build", "stage-02-compress", "stage-03-query"):
        assert (second.snapshot_path / stage / "stage-manifest.json").is_file()


def test_single_document_edit_reuses_unaffected_blocks(sample_dir: Path, tmp_path: Path, sample_config) -> None:
    state = tmp_path / "state"
    _run(sample_dir, state, sample_config.config, sample_config.failures)
    update_document(
        sample_dir / "documents.jsonl",
        "doc-0002",
        {"title_terms": ["alpha", "beta", "gamma"], "body_terms": ["alpha", "index", "query"]},
    )
    result = _run(sample_dir, state, sample_config.config, sample_config.failures)
    report = result.report
    assert report["reuse"]["counts"]["reused_blocks"] > 0
    assert report["reuse"]["counts"]["recomputed_blocks"] > 0
    assert report["summary"]["affected_shards"] == ["shard-b"]
    recomputed = report["reuse"]["counts"]["recomputed_blocks"]
    assert report["diagnostics"]["cache"]["encoding_calls"] == recomputed * 2
    assert report["reuse"]["recomputed_blocks"] == sorted(
        report["reuse"]["recomputed_blocks"], key=lambda item: item["block_no"]
    )


def test_incremental_final_state_matches_clean_full_rebuild(sample_dir: Path, tmp_path: Path) -> None:
    state = tmp_path / "incremental-state"
    paths = discover_inputs(sample_dir)
    initial_config = load_config(paths.config)
    _run(sample_dir, state, initial_config.config, initial_config.failures)
    update_document(
        sample_dir / "documents.jsonl",
        "doc-0003",
        {"title_terms": ["search", "vector"], "body_terms": ["cache", "index", "alpha"]},
    )
    config_path = sample_dir / "config.json"
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config_payload.update({"compression_tier": 2, "chunk_size": 4})
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    config_result = load_config(config_path)
    incremental = _run(sample_dir, state, config_result.config, config_result.failures)

    full_data = tmp_path / "full-data"
    full_data.mkdir()
    for name in ["documents.jsonl", "queries.jsonl", "vocabulary.txt", "query_versions.json", "config.json"]:
        (full_data / name).write_bytes((sample_dir / name).read_bytes())
    full = _run(full_data, tmp_path / "full-state", config_result.config, config_result.failures)
    assert first_difference(canonical_projection(incremental.report), canonical_projection(full.report)) is None
    assert incremental.report["canonical_digest"] == full.report["canonical_digest"]


def test_query_cache_tracks_document_length_even_when_positions_do_not_move(
    sample_dir: Path, tmp_path: Path, sample_config
) -> None:
    state = tmp_path / "state"
    first = _run(sample_dir, state, sample_config.config, sample_config.failures)
    query = next(item for item in first.report["queries"] if item["query_id"] == "q-001")
    target_doc_id = query["results"][0]["doc_id"]
    vocabulary = (sample_dir / "vocabulary.txt").read_text(encoding="utf-8").splitlines()
    appended_term = next(term for term in vocabulary if term not in query["query_terms"])
    rows = [json.loads(line) for line in (sample_dir / "documents.jsonl").read_text(encoding="utf-8").splitlines()]
    target = next(row for row in rows if row["doc_id"] == target_doc_id)
    update_document(
        sample_dir / "documents.jsonl",
        target_doc_id,
        {"body_terms": list(target["body_terms"]) + [appended_term]},
    )
    incremental = _run(sample_dir, state, sample_config.config, sample_config.failures)
    updated_query = next(item for item in incremental.report["queries"] if item["query_id"] == "q-001")
    assert updated_query["reused"] is False

    full_data = tmp_path / "full-data"
    full_data.mkdir()
    for name in ["documents.jsonl", "queries.jsonl", "vocabulary.txt", "query_versions.json", "config.json"]:
        (full_data / name).write_bytes((sample_dir / name).read_bytes())
    full = _run(full_data, tmp_path / "full-state", sample_config.config, sample_config.failures)
    assert incremental.report["canonical_digest"] == full.report["canonical_digest"]


def test_compression_tier_switch_reuses_precomputed_tier_artifacts(
    sample_dir: Path, tmp_path: Path, sample_config
) -> None:
    state = tmp_path / "state"
    _run(sample_dir, state, sample_config.config, sample_config.failures)
    tier2_config, failures = sample_config.config.with_overrides({"compression_tier": 2})
    assert failures == []
    switched = _run(sample_dir, state, tier2_config)
    assert switched.report["diagnostics"]["cache"]["encoding_calls"] == 0
    assert switched.report["diagnostics"]["cache"]["encoding_hits"] >= len(switched.report["blocks"]) * 2
    assert all(block["compression"]["selected_tier"] == 2 for block in switched.report["blocks"])
    assert switched.report["reuse"]["counts"]["reused_blocks"] == len(switched.report["blocks"])
    assert switched.report["reuse"]["counts"]["recomputed_queries"] == len(switched.report["queries"])
