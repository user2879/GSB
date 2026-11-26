from __future__ import annotations

import json
import subprocess
import sys

from gsb.trace import analyze_attribution
from gsb.trace import _stage_error


def test_reference_trace_reports_every_character_and_all_four_stages(project, baseline):
    attribution = analyze_attribution(
        project.dataset,
        config=project.config,
        templates=project.require_templates(),
        base_result=baseline,
        alternate_levels=("Q4", "Q6", "Q10"),
    )

    assert len(attribution["per_character"]) == 36
    assert attribution["reference"]["reference_bits"] == project.config.fixed_bits + 1
    assert attribution["reference"]["first_exceeding_counts"]["none"] == 36
    assert attribution["flip_count"] == 0
    for record in attribution["per_character"]:
        assert [stage["stage"] for stage in record["stages"]] == [
            "keypoint_coordinate_quantization",
            "template_match_score_rounding",
            "confidence_composition",
            "reading_order_adjudication",
        ]
        assert record["first_exceeding_stage"] is None


def test_zero_reference_uses_absolute_error_branch():
    stage = _stage_error(0, 17, stage="zero-check", note="unit-test")

    assert stage["truth_zero"] is True
    assert stage["error_branch"] == "absolute"
    assert stage["relative_error_ppm"] is None
    assert stage["error"] == 17


def test_cli_validate_returns_machine_readable_success(project_root):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "gsb",
            "--root",
            str(project_root),
            "validate",
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ok"
    assert payload["template_issues"] == []
    assert payload["dataset_issues"] == []
    assert payload["png_issues"] == []
    assert payload["samples"] == ["page_a", "page_b"]


def test_packaging_metadata_and_entry_point_declared(project_root):
    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (project_root / "requirements.txt").read_text(encoding="utf-8")
    assert 'name = "gsb-ancient-page"' in pyproject
    assert 'gsb = "gsb.__main__:main"' in pyproject
    assert "packages.find" in pyproject
    for dependency in ("numpy", "Pillow", "PyYAML", "pytest"):
        assert dependency.lower() in requirements.lower()
