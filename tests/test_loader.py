from __future__ import annotations

from pathlib import Path

import json

from iidx.demo import generate_sample
from iidx.loader import load_inputs, normalize_term


def test_normalization_folds_case_width_and_whitespace() -> None:
    assert normalize_term("  ＡＬＰＨＡ\t") == "alpha"
    assert normalize_term("Straße") == "strasse"
    assert normalize_term("A   B") == "a b"


def test_loader_isolates_invalid_rows_and_keeps_valid_entries(tmp_path: Path, sample_config) -> None:
    data = tmp_path / "invalid-sample"
    generate_sample(data, force=True, include_invalid=True)
    loaded = load_inputs(data, sample_config.config, None)
    codes = {item["code"] for item in loaded.failures}
    assert loaded.stats["accepted_documents"] == 36
    assert loaded.stats["accepted_queries"] == 12
    assert "E_DOCUMENT_TERM_OUT_OF_VOCAB" in codes
    assert "E_DOCUMENT_ID_DUPLICATE" in codes
    assert "E_QUERY_VERSION_UNSUPPORTED" in codes
    assert "E_QUERY_ID_DUPLICATE" in codes
    assert loaded.failures == sorted(
        loaded.failures,
        key=lambda item: (item["stage"], item["stable_key"], item["code"], item["message"]),
    )
    for docs in loaded.documents_by_shard.values():
        assert [doc.doc_id for doc in docs] == sorted(doc.doc_id for doc in docs)


def test_loader_is_deterministic(sample_dir: Path, sample_config) -> None:
    first = load_inputs(sample_dir, sample_config.config, None)
    second = load_inputs(sample_dir, sample_config.config, None)
    assert first.source_fingerprint == second.source_fingerprint
    assert first.shard_fingerprints == second.shard_fingerprints
    assert first.stats == second.stats
    assert first.failures == second.failures


def test_version_table_declares_mapping_keys_only(tmp_path: Path, sample_config) -> None:
    data = tmp_path / "version-sample"
    generate_sample(data, force=True)
    (data / "query_versions.json").write_text(
        json.dumps({"versions": {"v1": {"status": "stable"}, "not-a-version": True}}),
        encoding="utf-8",
    )
    loaded = load_inputs(data, sample_config.config, None)
    assert loaded.declared_query_versions == ("not-a-version", "v1")
    assert "stable" not in loaded.declared_query_versions
    assert "status" not in loaded.declared_query_versions
