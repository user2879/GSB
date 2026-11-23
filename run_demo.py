"""End-to-end acceptance harness for the deterministic VC pipeline.

Validates:
  1. v1<->v2 conversion, sidecar restoration and precision-loss accounting;
  2. per-segment synthesis plus segment replacement isolation;
  3. an edit-command sequence where every incremental run equals a clean
     full rebuild of the same final state, field by field;
  4. serial vs multi-threaded execution producing identical PCM and keys;
  5. snapshot mismatch detection forcing a full rebuild.
"""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np

from voice_conversion.audio import read_wav
from voice_conversion.config import PipelineConfig
from voice_conversion.convert import (downgrade_v2_to_v1, precision_loss,
                                      restore_v1_with_sidecar,
                                      upgrade_v1_to_v2)
from voice_conversion.editor import Workspace
from voice_conversion.errors import FailureLog
from voice_conversion.pipeline import run_pipeline
from voice_conversion.v1 import parse_v1
from voice_conversion.v2 import parse_v2

ROOT = Path(__file__).parent
SAMPLES = ROOT / "samples"
ART = ROOT / "artifacts"


def _normalize(report: dict) -> dict:
    """Drop run-relative reuse counters before state-equality comparison."""
    report = copy.deepcopy(report)
    report.pop("reused_segments", None)
    report.pop("recomputed_segments", None)
    report.pop("reuse_invalidation", None)
    for row in report.get("segments", []):
        row.pop("reused", None)
        row.get("stage_reuse", {}).clear()
    return report


def check_conversion() -> dict:
    config = PipelineConfig.load(SAMPLES / "config" / "acoustic.json")
    failures = FailureLog()
    v1_segments = parse_v1(SAMPLES / "exchange" / "segments.v1.txt",
                           config, failures)
    v2_segments = parse_v2(SAMPLES / "exchange" / "segments.v2.json",
                           config, FailureLog())
    upgraded = upgrade_v1_to_v2(v1_segments, config)
    assert all(s.source_version == "v2" for s in upgraded)
    assert [s.intensity for s in upgraded] == \
        [s.intensity_permille / 1000.0 for s in v1_segments]

    downgraded, sidecar = downgrade_v2_to_v1(v2_segments, config)
    restored = restore_v1_with_sidecar(downgraded, sidecar)
    for before, after in zip(v2_segments, restored):
        assert before.start_frame == after.start_frame
        assert before.end_frame == after.end_frame
        assert before.intensity == after.intensity
        assert before.timbre == after.timbre
        assert before.passthrough == after.passthrough
        assert before.precision_tag == after.precision_tag
    loss = precision_loss(v2_segments, downgraded)
    return {
        "v1_segments": len(v1_segments),
        "v2_segments": len(v2_segments),
        "sidecar_restored": len(restored),
        "max_intensity_abs_error": loss["intensity"]["max_abs"],
        "max_start_frame_abs_error": loss["start_frame"]["max_abs"],
        "max_end_frame_abs_error": loss["end_frame"]["max_abs"],
        "sidecar_residuals_preserved": all(
            abs(e["start_residual"]) > 0 for e in sidecar["segments"]),
    }


def _run(workdir: Path, workspace: Workspace, *, full_rebuild: bool,
         workers: int | None = None) -> tuple[dict, np.ndarray]:
    config = workspace.config
    snap = workdir / "snapshot"
    out = workdir / "output.wav"
    report = run_pipeline(
        exchange_path=workspace.exchange_path,
        version=workspace.version,
        source_wav=SAMPLES / "wav" / "source.wav",
        config=config,
        snapshot_dir=snap,
        output_wav=out,
        full_rebuild=full_rebuild,
        max_workers=workers,
    )
    pcm, _ = read_wav(out)
    return report, pcm


def check_edit_sequence() -> dict:
    base = ART / "edit_sequence"
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True)
    config = PipelineConfig.load(SAMPLES / "config" / "acoustic.json")

    commands = [
        {"op": "switch_version", "version": "v2"},
        {"op": "set_boundary", "segment": 1, "start_frame": 62.0,
         "end_frame": 128.0},
        {"op": "set_intensity", "segment": 3, "intensity": 0.612},
        {"op": "delete_segment", "segment": 2},
        {"op": "append_segment", "segment": 6, "start_frame": 340.0,
         "end_frame": 380.0, "intensity": 0.555, "timbre": "soprano"},
        {"op": "switch_version", "version": "v1"},
        {"op": "switch_version", "version": "v2"},
        {"op": "set_quant_levels", "levels": 8},
        {"op": "set_boundary", "segment": 0, "start_frame": 3.0,
         "end_frame": 58.0},
    ]

    inc_dir = base / "incremental"
    inc_ws = Workspace(inc_dir / "ws", config)
    inc_ws.reset_to(SAMPLES / "exchange" / "segments.v1.txt",
                    SAMPLES / "config" / "acoustic.json", "v1")
    persistent = inc_dir / "persistent_snapshot"
    step_results = []
    previous_pcm = None
    for step, command in enumerate(commands):
        inc_ws.apply(command)
        report, pcm = _run(persistent, inc_ws, full_rebuild=False)
        if previous_pcm is not None and command["op"] in (
                "set_quant_levels",):
            assert not np.array_equal(previous_pcm, pcm), \
                "quant change must alter output"
        previous_pcm = pcm
        step_results.append({
            "step": step, "op": command["op"],
            "recomputed": report["recomputed_segments"],
            "reused": report["reused_segments"],
        })

    # Clean full rebuild of the identical final state.
    full_dir = base / "full"
    full_ws = Workspace(full_dir / "ws", inc_ws.config)
    full_ws.directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(inc_ws.exchange_path,
                full_ws.directory / inc_ws.exchange_path.name)
    full_ws.version = inc_ws.version
    full_report, full_pcm = _run(full_dir / "run", full_ws,
                                 full_rebuild=True)
    inc_report, inc_pcm = _run(persistent, inc_ws, full_rebuild=False)

    assert np.array_equal(full_pcm, inc_pcm), "assembled PCM differs"
    assert full_report["output_fingerprint"] == inc_report["output_fingerprint"]
    assert full_report["assembly_key"] == inc_report["assembly_key"]
    assert _normalize(full_report) == _normalize(inc_report)
    assert [f["segment"] for f in full_report["failures"]] == \
        [f["segment"] for f in inc_report["failures"]]
    total_reused = sum(step["reused"] for step in step_results)
    assert total_reused > 0, "incremental run never reused a segment"
    return {
        "commands": len(commands),
        "incremental_steps": step_results,
        "final_segment_count": full_report["valid_segment_count"],
        "full_equals_incremental": True,
        "output_fingerprint": full_report["output_fingerprint"],
    }


def check_concurrency() -> dict:
    base = ART / "concurrency"
    shutil.rmtree(base, ignore_errors=True)
    config = PipelineConfig.load(SAMPLES / "config" / "acoustic.json")
    ws = Workspace(base / "ws", config)
    ws.reset_to(SAMPLES / "exchange" / "segments.v2.json",
                SAMPLES / "config" / "acoustic.json", "v2")
    serial_report, serial_pcm = _run(base / "serial", ws,
                                     full_rebuild=True, workers=1)
    parallel_report, parallel_pcm = _run(base / "parallel", ws,
                                         full_rebuild=True, workers=4)
    assert np.array_equal(serial_pcm, parallel_pcm)
    assert serial_report["assembly_key"] == parallel_report["assembly_key"]
    for left, right in zip(serial_report["segments"],
                           parallel_report["segments"]):
        assert left["cache_keys"] == right["cache_keys"]
        assert left["mapping"] == right["mapping"]
    assert serial_report["failures"] == parallel_report["failures"]
    return {
        "serial_workers": 1,
        "parallel_workers": 4,
        "pcm_identical": True,
        "cache_keys_identical": True,
        "failure_order_identical": True,
    }


def check_segment_isolation() -> dict:
    base = ART / "isolation"
    shutil.rmtree(base, ignore_errors=True)
    config = PipelineConfig.load(SAMPLES / "config" / "acoustic.json")
    ws = Workspace(base / "ws", config)
    ws.reset_to(SAMPLES / "exchange" / "segments.v1.txt",
                SAMPLES / "config" / "acoustic.json", "v1")
    report_before, _ = _run(base / "before", ws, full_rebuild=True)
    keys_before = {row["segment"]: row["cache_keys"]
                   for row in report_before["segments"]}
    ws.apply({"op": "set_boundary", "segment": 2,
              "start_frame": 134.0, "end_frame": 198.0})
    report_after, _ = _run(base / "after", ws, full_rebuild=False)
    keys_after = {row["segment"]: row["cache_keys"]
                  for row in report_after["segments"]}
    untouched = [idx for idx in keys_before if idx != 2]
    unchanged = all(keys_before[idx] == keys_after[idx] for idx in untouched)
    changed = keys_before[2] != keys_after[2]
    assert unchanged and changed
    return {"untouched_segments_reused": untouched,
            "changed_segment_invalidated": 2}


def check_snapshot_mismatch() -> dict:
    base = ART / "snapshot"
    shutil.rmtree(base, ignore_errors=True)
    config = PipelineConfig.load(SAMPLES / "config" / "acoustic.json")
    ws = Workspace(base / "ws", config)
    ws.reset_to(SAMPLES / "exchange" / "segments.v1.txt",
                SAMPLES / "config" / "acoustic.json", "v1")
    _run(base / "run", ws, full_rebuild=True)
    # A structural config change is a hard resume-gate mismatch.
    altered = PipelineConfig.from_dict(
        {**config.to_dict(), "mel_bands": 24})
    ws.config = altered
    ws._write_config()
    report, _ = _run(base / "run", ws, full_rebuild=False)
    codes = [f["code"] for f in report["failures"]]
    assert "E_SNAPSHOT_MISMATCH" in codes
    assert report["recomputed_segments"] == report["valid_segment_count"]

    # A quant-only change is not a hard mismatch: keys change, gate stays valid.
    quant_base = ART / "snapshot_quant"
    shutil.rmtree(quant_base, ignore_errors=True)
    q_ws = Workspace(quant_base / "ws", config)
    q_ws.reset_to(SAMPLES / "exchange" / "segments.v1.txt",
                  SAMPLES / "config" / "acoustic.json", "v1")
    _run(quant_base / "run", q_ws, full_rebuild=True)
    q_ws.config = PipelineConfig.from_dict(
        {**config.to_dict(), "quant_levels": 16})
    q_ws._write_config()
    quant_report, _ = _run(quant_base / "run", q_ws, full_rebuild=False)
    assert "E_SNAPSHOT_MISMATCH" not in [
        f["code"] for f in quant_report["failures"]]
    assert quant_report["recomputed_segments"] == \
        quant_report["valid_segment_count"]
    return {"mismatch_detected": True,
            "rebuilt_segments": report["recomputed_segments"],
            "quant_change_without_mismatch": True}


def check_parser_failures() -> dict:
    config = PipelineConfig.load(SAMPLES / "config" / "acoustic.json")
    failures = FailureLog()
    segments = parse_v1(SAMPLES / "exchange" / "segments_broken.v1.txt",
                        config, failures)
    rows = failures.to_list()
    assert any(f["code"] == "E_FIELD_TRUNCATED" for f in rows)
    bad = next(f for f in rows if f["code"] == "E_FIELD_TRUNCATED")
    assert bad["line"] >= 1 and bad["field"]
    return {"parsed_segments": len(segments),
            "truncation_failures": sum(
                1 for f in rows if f["code"] == "E_FIELD_TRUNCATED"),
            "fields": sorted({f["field"] for f in rows})}


def main() -> dict:
    ART.mkdir(parents=True, exist_ok=True)
    summary = {
        "conversion": check_conversion(),
        "edit_sequence": check_edit_sequence(),
        "concurrency": check_concurrency(),
        "segment_isolation": check_segment_isolation(),
        "snapshot_mismatch": check_snapshot_mismatch(),
        "parser_failures": check_parser_failures(),
        "status": "PASS",
    }
    out = ART / "validation_report.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
