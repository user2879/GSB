from __future__ import annotations

import json

import pytest

from rsseg.edits import copy_sample_set
from rsseg.engine import IncrementalEngine
from rsseg.errors import ErrorCode
from rsseg.runner import build_inputs, validate_index_or_report


def _single_injection(sources: Path, injection: dict) -> Path:
    path = sources / "single_injection.json"
    path.write_text(json.dumps({"injections": [injection]}), encoding="utf-8")
    return path


def test_truncated_png_is_caught_and_recorded(sample_set, tmp_path):
    sources = tmp_path / "sources"
    work = tmp_path / "work"
    copy_sample_set(sample_set, sources)
    injection = {"id": "INJ-001", "kind": "truncated_png", "target": [0, 1],
                 "payload": {"keep_ratio": 0.4}}
    inputs = build_inputs(sources, work, injections_path=_single_injection(sources, injection))
    report = IncrementalEngine(inputs).run()
    codes = {(f.injection_id, f.code) for f in report.failures}
    assert ("INJ-001", ErrorCode.E_TILE_READ_FAILED) in codes
    assert "INJ-001" in report.hits
    assert (0, 1) in report.pending_tiles


def test_non_finite_band_patch_rejected(sample_set, tmp_path):
    sources = tmp_path / "sources"
    work = tmp_path / "work"
    copy_sample_set(sample_set, sources)
    injection = {"id": "INJ-002", "kind": "non_finite_band", "target": [1, 0]}
    inputs = build_inputs(sources, work, injections_path=_single_injection(sources, injection))
    report = IncrementalEngine(inputs).run()
    codes = {(f.injection_id, f.code) for f in report.failures}
    assert ("INJ-002", ErrorCode.E_NON_FINITE_BAND) in codes
    assert "INJ-002" in report.hits


def test_asymmetric_adjacency_raises_index_error(sample_set, tmp_path):
    sources = tmp_path / "sources"
    copy_sample_set(sample_set, sources)
    record = validate_index_or_report(sources, sources / "injections.json")
    assert record is not None
    assert record.injection_id == "INJ-003"
    assert record.code == ErrorCode.E_INDEX_INVALID


def test_full_injection_plan_report_contains_all_hit_ids(sample_set, tmp_path):
    sources = tmp_path / "sources"
    work = tmp_path / "work"
    copy_sample_set(sample_set, sources)
    # The bundled plan contains an index-level fault which is fatal at load.
    with pytest.raises(Exception) as exc:
        build_inputs(sources, work, injections_path=sources / "injections.json")
    assert exc.value.code == ErrorCode.E_INDEX_INVALID
    assert exc.value.injection_id == "INJ-003"


def test_tile_faults_do_not_poison_other_tiles(sample_set, tmp_path):
    sources = tmp_path / "sources"
    work = tmp_path / "work"
    copy_sample_set(sample_set, sources)
    injection = {"id": "INJ-001", "kind": "truncated_png", "target": [0, 1],
                 "payload": {"keep_ratio": 0.4}}
    inputs = build_inputs(sources, work, injections_path=_single_injection(sources, injection))
    report = IncrementalEngine(inputs).run()
    failed_targets = {f.target for f in report.failures if f.injection_id == "INJ-001"}
    assert failed_targets == {"tile:0,1"}
    assert len(report.recomputed_tiles) == 8
