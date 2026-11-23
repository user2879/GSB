from __future__ import annotations

import json
from pathlib import Path

from ancient_vlayout.engine import RecognitionEngine
from ancient_vlayout.util import atomic_write_json, read_json


def test_injections_hit_and_are_isolated(generated_samples: Path, tmp_path: Path) -> None:
    result = RecognitionEngine(
        generated_samples,
        state_dir=tmp_path / "fault-state",
        injection_mode=True,
    ).settle()
    injection_items = {item["id"]: item for item in result.diagnostics["injections"]["items"]}
    assert set(injection_items) == {"INJ-001", "INJ-002", "INJ-003", "INJ-004"}
    assert all(item["hit"] for item in injection_items.values())
    assert injection_items["INJ-001"]["observed_codes"] == ["E_IMAGE_INVALID"]
    assert injection_items["INJ-002"]["observed_codes"] == ["E_TRUTH_INVALID"]
    assert injection_items["INJ-003"]["observed_codes"] == ["E_TEMPLATE_MISSING"]
    assert injection_items["INJ-004"]["observed_codes"] == ["E_CONFIG_INVALID"]
    assert [item["file"] for item in result.recognition["failures"]] == sorted(
        item["file"] for item in result.recognition["failures"]
    )
    assert [item["file"] for item in result.recognition["failures"]] == [
        "page_01.png",
        "page_03.png",
        "page_04.png",
    ]
    assert [image["file"] for image in result.recognition["images"]] == ["page_02.png"]
    assert sum(
        len(column["chars"])
        for image in result.recognition["images"]
        for column in image["columns"]
    ) == 20


def test_snapshot_pointer_recovery(generated_samples: Path, tmp_path: Path) -> None:
    state = tmp_path / "state"
    first = RecognitionEngine(generated_samples, state_dir=state).settle()
    pointer = state / "current.json"
    pointer.write_text("{broken", encoding="utf-8")
    recovered = RecognitionEngine(generated_samples, state_dir=state).run()
    assert recovered.report["snapshot"]["pointer_recovered"] is True
    assert recovered.recognition == first.recognition
    assert json.loads(pointer.read_text(encoding="utf-8"))["generation"] == first.report["snapshot"]["generation"]


def test_corrupt_stage_artifact_is_recomputed_without_changing_result(
    generated_samples: Path, tmp_path: Path
) -> None:
    state = tmp_path / "state"
    first = RecognitionEngine(generated_samples, state_dir=state).settle()
    engine = RecognitionEngine(generated_samples, state_dir=state)
    assert engine.current is not None
    manifest = engine.current.manifest
    match_key, relative = next(iter(manifest["stage_files"]["match"].items()))
    target = engine.current.path / relative
    target.write_text("{corrupt", encoding="utf-8")
    repaired = RecognitionEngine(generated_samples, state_dir=state).run()
    assert repaired.recognition == first.recognition
    assert repaired.report["cache"]["recomputed_count"] == 1
    assert repaired.report["cache"]["recomputed"][0]["stage"] == "match"
    assert repaired.report["snapshot"]["generation"] == first.report["snapshot"]["generation"] + 1
    assert (
        repaired.snapshot_manifest["stage_versions"]["match"]
        == first.report["snapshot"]["stage_versions"]["match"]
    )


def test_config_change_updates_snapshot_versions(generated_samples: Path, tmp_path: Path) -> None:
    state = tmp_path / "state"
    first = RecognitionEngine(generated_samples, state_dir=state).settle()
    config = read_json(generated_samples / "config.json")
    first_config_hash = first.report["snapshot"]["config_hash"]
    # Use an engine-local config override; do not mutate the shared fixture.
    changed = dict(config)
    changed["match_metric"] = "l1"
    result = RecognitionEngine(
        generated_samples,
        config=changed,
        state_dir=state,
    ).settle()
    assert result.report["snapshot"]["config_changed"] is True
    assert result.report["snapshot"]["version_changed"] is True
    assert result.report["snapshot"]["generation"] > first.report["snapshot"]["generation"]
    assert result.report["snapshot"]["config_hash"] != first_config_hash
    assert result.report["snapshot"]["stage_versions"]["match"] != first.report["snapshot"]["stage_versions"]["match"]
