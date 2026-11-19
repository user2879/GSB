from __future__ import annotations

import json
from pathlib import Path

import pytest

from fingerprint_workbench.config import BudgetConfig, FingerprintConfig
from fingerprint_workbench.errors import WorkbenchError
from fingerprint_workbench.search import compare_retrieval_candidates, search
from fingerprint_workbench.workbench import Workbench


def _import_many(workbench: Workbench, generated: dict, track_ids: list[str]) -> None:
    for track_id in track_ids:
        workbench.import_track(track_id, generated["tracks"][track_id])


def test_incremental_shards_and_exact_undo(generated: dict, tmp_path: Path) -> None:
    workbench = Workbench(tmp_path / "wb")
    workbench.import_track("track_01", generated["tracks"]["track_01"])
    first_version = workbench.current.version
    first_hash = workbench.current.manifest_hash
    first_shards = workbench.shard_file_hashes()
    workbench.import_track("track_02", generated["tracks"]["track_02"])
    second_shards = workbench.shard_file_hashes()
    assert workbench.current.manifest_hash != first_hash
    assert any(first_shards[key] != second_shards[key] for key in first_shards)
    threshold_state = workbench.update_config(similarity_threshold=0.2)
    assert workbench.shard_file_hashes() == second_shards
    assert threshold_state.manifest_hash != workbench.state_at(2).manifest_hash
    reconstruction = workbench.reconstruct_version(3)
    assert reconstruction["index_hash_match"] is True
    assert reconstruction["all_shards_match"] is True
    workbench.delete_track("track_02")
    restored = workbench.undo()
    assert restored.version == 5
    assert restored.manifest_hash == workbench.state_at(3).manifest_hash
    assert restored.manifest_hash != first_hash
    restored_first = workbench.revert(first_version)
    assert restored_first.manifest_hash == first_hash
    assert restored_first.state_digest == workbench.state_at(first_version).state_digest
    workbench.close()


def test_rollback_restores_retrieval_candidates(generated: dict, tmp_path: Path) -> None:
    workbench = Workbench(tmp_path / "wb")
    _import_many(
        workbench,
        generated,
        ["track_01", "track_02", "track_03", "track_04", "track_05", "track_06"],
    )
    baseline_version = workbench.current.version
    baseline_hash = workbench.current.manifest_hash
    query = generated["queries"]["q_01_exact_cross"]
    before = search(workbench, query, query_id="q_01_exact_cross")
    changed = workbench.update_config(
        fingerprint_config=FingerprintConfig(
            window_samples=1024,
            hop_samples=256,
            peak_density=18.0,
        ),
        similarity_threshold=0.15,
    )
    assert changed.manifest_hash != baseline_hash
    after = workbench.undo()
    assert after.version == baseline_version + 2
    assert after.manifest_hash == baseline_hash
    restored = search(workbench, query, query_id="q_01_exact_cross")
    compare_retrieval_candidates(before, restored)
    workbench.close()


def test_partition_and_version_error_codes(generated: dict, tmp_path: Path) -> None:
    workbench = Workbench(tmp_path / "wb")
    workbench.import_track("track_01", generated["tracks"]["track_01"])
    manifest = workbench.index.load_manifest(workbench.current.manifest_hash)
    shard, reference = sorted(manifest["shards"].items())[0]
    path = workbench.root / "indexes" / "partitions" / reference["file"]
    original = path.read_bytes()
    document = json.loads(original)
    document["schema"] = 99
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(WorkbenchError) as caught:
        workbench.index.load_shard(reference, shard)
    assert caught.value.code == "E_INDEX_VERSION"
    path.write_bytes(original + b"tamper")
    with pytest.raises(WorkbenchError) as caught:
        workbench.index.load_shard(reference, shard)
    assert caught.value.code == "E_PARTITION_HASH"
    workbench.close()
    (tmp_path / "wb" / "CURRENT.json").unlink()
    with pytest.raises(WorkbenchError) as caught:
        Workbench(tmp_path / "wb")
    assert caught.value.code == "E_SNAPSHOT_STALE"


def test_query_drift_diagnostic() -> None:
    expected = {
        "candidates": [
            {
                "track_id": "track_01",
                "score": 1.0,
                "score_components": {"histogram_peak_weight": 10},
            }
        ]
    }
    actual = {
        "candidates": [
            {
                "track_id": "track_02",
                "score": 0.5,
                "score_components": {"histogram_peak_weight": 4},
            }
        ]
    }
    with pytest.raises(WorkbenchError) as caught:
        compare_retrieval_candidates(expected, actual)
    assert caught.value.code == "E_QUERY_DRIFT"
    assert caught.value.detail["first_divergence_index"] == 0
    assert caught.value.detail["expected_score_components"] == {"histogram_peak_weight": 10}
    assert caught.value.detail["actual_score_components"] == {"histogram_peak_weight": 4}


def test_budget_and_audio_errors(generated: dict, tmp_path: Path) -> None:
    with pytest.raises(WorkbenchError) as caught:
        BudgetConfig(query_threads=0)
    assert caught.value.code == "E_BUDGET_DENIED"
    bad_audio = tmp_path / "bad.wav"
    bad_audio.write_bytes(b"not a wave")
    workbench = Workbench(tmp_path / "wb")
    with pytest.raises(WorkbenchError) as caught:
        workbench.import_track("bad", bad_audio)
    assert caught.value.code == "E_AUDIO_FORMAT"
    workbench.close()
