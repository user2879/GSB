from __future__ import annotations

import json
from pathlib import Path

from mesh_repair.pipeline import ARTIFACT_ORDER, run_fixed_pipeline

from .conftest import PROJECT_ROOT


def test_fixed_sequence_passes_and_preserves_legal_geometry(default_run):
    report = default_run.report
    assert default_run.status == "pass"
    assert default_run.business_code == "OK"
    assert default_run.exit_code == 0
    assert report["quality"]["status"] == "pass"
    assert report["verifier"]["status"] == "pass"
    assert report["quality_verifier_cross_check"]["status"] == "pass"
    assert report["quality"]["misdeleted_legal_faces"] == []
    assert report["quality"]["legal_face_geometry_preserved"] is True
    assert all(record["processed"] for record in report["repair"]["defect_alignment"])
    assert report["repair"]["scheduler"]["block_count"] == len(report["repair"]["block_summaries"])

    baseline_hash = report["step_hashes"]["baseline.ply"]
    assert report["step_hashes"]["merged.ply"] == baseline_hash
    assert set(report["step_order"]) == set(ARTIFACT_ORDER)


def test_two_runs_are_byte_identical(default_run, tmp_path):
    second = run_fixed_pipeline(PROJECT_ROOT / "configs" / "default.json", tmp_path / "again")
    assert second.status == "pass"
    for filename in (*ARTIFACT_ORDER, "report.json", "hashes.json"):
        assert (default_run.output_dir / filename).read_bytes() == (
            second.output_dir / filename
        ).read_bytes(), filename


def test_merged_hash_is_thread_count_invariant(default_run, tmp_path):
    hashes = [default_run.report["step_hashes"]["merged.ply"]]
    for threads in (1, 2, 8):
        result = run_fixed_pipeline(
            PROJECT_ROOT / "configs" / "default.json",
            tmp_path / f"threads_{threads}",
            threads=threads,
        )
        assert result.status == "pass"
        assert result.report["repair"]["scheduler"]["effective_threads"] == threads
        hashes.append(result.report["step_hashes"]["merged.ply"])
    assert len(set(hashes)) == 1


def test_budget_degradation_keeps_every_block(tmp_path):
    result = run_fixed_pipeline(PROJECT_ROOT / "configs" / "low_budget.json", tmp_path / "low")
    assert result.status == "pass"
    scheduler = result.report["repair"]["scheduler"]
    assert scheduler["degraded"] is True
    assert scheduler["effective_threads"] == 3
    reasons = {step["reason"] for step in scheduler["fallback_steps"]}
    assert reasons == {"time_budget", "memory_budget"}
    assert len(result.report["repair"]["block_summaries"]) == scheduler["block_count"]
    assert result.report["repair"]["boundary"]["position_conflict_count"] == 0
    assert result.report["step_hashes"]["merged.ply"] == result.report["step_hashes"]["baseline.ply"]


def test_budget_denial_returns_business_code(tmp_path):
    result = run_fixed_pipeline(PROJECT_ROOT / "configs" / "denied_budget.json", tmp_path / "denied")
    assert result.status == "fail"
    assert result.business_code == "E_BUDGET_DENIED"
    assert result.exit_code == 13
    assert result.report["error"]["details"]["largest_block_bytes"] > result.report["error"]["details"]["budget_bytes"]
    assert "merged.ply" not in result.report["step_order"]
    hashes = json.loads((result.output_dir / "hashes.json").read_text(encoding="utf-8"))
    assert hashes["business_code"] == "E_BUDGET_DENIED"


def test_cli_validate_consumes_only_files(default_run, tmp_path):
    import subprocess
    import sys

    report_path = tmp_path / "verifier_cli.json"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "mesh_repair",
            "validate",
            "--mesh",
            str(default_run.output_dir / "merged.ply"),
            "--truth",
            str(default_run.output_dir / "ground_truth.json"),
            "--out",
            str(report_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["verifier"] == "independent_file_level_triangle_mesh"
