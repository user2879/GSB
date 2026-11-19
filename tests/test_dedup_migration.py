from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from audiofp.config import DEFAULT_CONFIG, LEGACY_CONFIG
from audiofp.errors import DedupMergeError, IndexVersionError
from audiofp.library import Library


def _truth(manifest: dict) -> dict[str, str]:
    ids = {track["truth_id"] for track in manifest["tracks"]}
    return {
        path: truth_id
        for path, truth_id in manifest["truth_by_relative_path"].items()
        if truth_id in ids
    }


def test_multiformat_groups_are_deduplicated_without_false_merges(
    fixture_root: Path, manifest: dict, main_index: Path
) -> None:
    with Library(main_index, DEFAULT_CONFIG) as library:
        groups = library.store.list_groups()
        assert len(groups) == 3
        assert all(len(group["sources"]) == 3 for group in groups)
        report = library.validate_dedup_truth(_truth(manifest))
        assert report["ok"] is True
        fake = dict(_truth(manifest))
        fake[groups[0]["sources"][0]["relative_path"]] = "deliberate_distinct_truth"
        with pytest.raises(DedupMergeError) as error:
            library.validate_dedup_truth(fake)
        assert error.value.details["merged_groups"][0]["fingerprint_evidence"]


def _rows(store: object) -> list[tuple]:
    return [
        tuple(row)
        for row in store.connection.execute(
            "SELECT s.relative_path, f.fingerprint_hash, f.anchor_frame, f.target_frame, "
            "f.band1, f.band2, f.delta_frames FROM fingerprints f "
            "JOIN sources s ON s.source_id=f.source_id "
            "ORDER BY s.relative_path, f.anchor_frame, f.target_frame, f.band1, f.band2, f.fingerprint_hash"
        )
    ]


def test_migration_is_idempotent_and_matches_full_rebuild(
    fixture_root: Path,
    manifest: dict,
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy.db"
    with Library(legacy, LEGACY_CONFIG) as library:
        for track in manifest["tracks"]:
            for entry in track["formats"]:
                library.import_file(fixture_root / entry["relative_path"], root=fixture_root)
    unmigratable = tmp_path / "unmigratable.db"
    shutil.copy2(legacy, unmigratable)

    with Library(legacy, DEFAULT_CONFIG) as migrated:
        migrated_rows = _rows(migrated.store)
        migrated_stats = migrated.store.stats()
        migrated_query = migrated.query_file(
            fixture_root / manifest["query"]["formats"][1]["relative_path"],
            top_k=3,
        )
        first_source_path = migrated.store.list_groups()[0]["sources"][0]["relative_path"]
        reimport = migrated.import_file(fixture_root / first_source_path, root=fixture_root)
        assert reimport.outcome.exact_duplicate is True
        assert migrated.store.stats()["revision"] == migrated_stats["revision"]
    with Library(legacy, DEFAULT_CONFIG) as idempotent:
        assert idempotent.store.stats()["revision"] == migrated_stats["revision"]
        assert idempotent.store.stats()["content_hash"] == migrated_stats["content_hash"]

    rebuilt = tmp_path / "rebuilt.db"
    with Library(rebuilt, DEFAULT_CONFIG) as library:
        for track in manifest["tracks"]:
            for entry in track["formats"]:
                library.import_file(fixture_root / entry["relative_path"], root=fixture_root)
        rebuilt_rows = _rows(library.store)
        rebuilt_stats = library.store.stats()
        rebuilt_query = library.query_file(
            fixture_root / manifest["query"]["formats"][1]["relative_path"],
            top_k=3,
        )
    assert sorted(migrated_rows) == sorted(rebuilt_rows)
    assert migrated_stats["content_hash"] == rebuilt_stats["content_hash"]
    assert migrated_query["results"][0]["display_label"] == rebuilt_query["results"][0]["display_label"]
    assert migrated_query["results"][0]["score"] == rebuilt_query["results"][0]["score"]

    incompatible = replace(
        LEGACY_CONFIG,
        version="fp-v9",
        n_fft=1024,
        encoding_space="incompatible",
    )
    with pytest.raises(IndexVersionError) as error:
        Library(unmigratable, incompatible, auto_migrate=True).close()
    assert error.value.code == "E_INDEX_VERSION"
