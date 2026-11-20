from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from parseapp import pipeline as pipeline_module
from parseapp.config import ConfigError, load_config
from parseapp.detection import detect_lines, enumerate_candidates
from parseapp.loader import load_page
from parseapp.png import decode_png, encode_png_rgb
from parseapp.pipeline import SimulatedTermination, run_pipeline
from parseapp.validation import PageInvalid, normalize_truth


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
CONFIG_PATH = ROOT / "config" / "parse_config.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _copy_samples(tmp_path: Path) -> Path:
    import shutil

    destination = tmp_path / "samples"
    shutil.copytree(SAMPLES, destination)
    return destination


def _business_view(report: dict) -> dict:
    def strip_status(value):
        if isinstance(value, list):
            return [strip_status(item) for item in value]
        if isinstance(value, dict):
            return {
                key: strip_status(item)
                for key, item in value.items()
                if key != "cache_status"
            }
        return value

    return {
        "page_order": report["page_order"],
        "budget_ledger": report["budget_ledger"]["entries"],
        "pages": [
            {
                "file": page["file"],
                "status": page["status"],
                "tables": strip_status(page["tables"]),
                "charts": strip_status(page["charts"]),
                "line_detection": {
                    "segments": strip_status(page["line_detection"]["segments"])
                    if page["line_detection"]
                    else None,
                    "drop_counts": page["line_detection"]["drop_counts"]
                    if page["line_detection"]
                    else None,
                },
            }
            for page in report["pages"]
        ],
        "pending_queue": report["pending_queue"],
        "failures": report["failures"],
    }


def test_parallel_and_serial_reports_are_identical(tmp_path: Path):
    config = _config()
    parallel = run_pipeline(SAMPLES, config, tmp_path / "parallel", serial=False)
    serial = run_pipeline(SAMPLES, config, tmp_path / "serial", serial=True)
    assert _canonical(parallel) == _canonical(serial)
    assert parallel["status"] == "completed"
    assert parallel["counts"] == {"pages": 4, "ok": 4, "failed": 0, "pending": 0}
    assert parallel["budget_ledger"]["entries"] == [
        {
            "sequence": 1,
            "file": "page_01.png",
            "sort_key": "page_01.png|line",
            "points": 1289,
            "remaining_before": 5000,
            "remaining_after": 3711,
        },
        {
            "sequence": 2,
            "file": "page_02.png",
            "sort_key": "page_02.png|line",
            "points": 60,
            "remaining_before": 3711,
            "remaining_after": 3651,
        },
        {
            "sequence": 3,
            "file": "page_03.png",
            "sort_key": "page_03.png|line",
            "points": 1083,
            "remaining_before": 3651,
            "remaining_after": 2568,
        },
        {
            "sequence": 4,
            "file": "page_04.png",
            "sort_key": "page_04.png|line",
            "points": 87,
            "remaining_before": 2568,
            "remaining_after": 2481,
        },
    ]


def test_tables_and_chart_points_match_page_truth(tmp_path: Path):
    report = run_pipeline(SAMPLES, CONFIG_PATH, tmp_path / "state")
    assert report["status"] == "completed"
    for page_report in report["pages"]:
        truth = json.loads((SAMPLES / page_report["file"].replace(".png", ".json")).read_text())
        assert page_report["status"] == "ok"
        assert len(page_report["tables"]) == len(truth["tables"])
        for parsed, expected in zip(page_report["tables"], truth["tables"]):
            assert [line["coord"] for line in parsed["row_lines"]] == expected["row_lines"]
            assert [line["coord"] for line in parsed["col_lines"]] == expected["col_lines"]
            parsed_cells = sorted(
                (
                    cell["row_start"],
                    cell["row_end"],
                    cell["col_start"],
                    cell["col_end"],
                    tuple(cell["bbox"]),
                )
                for cell in parsed["cells"]
            )
            expected_cells = sorted(
                (
                    cell["row"],
                    cell["row"] + cell["row_span"] - 1,
                    cell["col"],
                    cell["col"] + cell["col_span"] - 1,
                    tuple(cell["bbox"]),
                )
                for cell in expected["cells"]
            )
            assert parsed_cells == expected_cells
            for cell in parsed["cells"]:
                assert cell["row_line_ids"]
                assert cell["col_line_ids"]
                assert cell["line_refs"]["top_row_line_ids"]
                assert cell["line_refs"]["left_col_line_ids"]
                assert cell["cache_key"]
            assert parsed["classification"]["classified_role"] == "table"

        assert len(page_report["charts"]) == len(truth["charts"])
        for parsed, expected in zip(page_report["charts"], truth["charts"]):
            assert parsed["kind"] == expected["kind"]
            assert parsed["classification"]["classified_role"] == "chart"
            assert len(parsed["points"]) == len(expected["points"])
            for got_point, truth_point in zip(parsed["points"], expected["points"]):
                assert got_point["pixel"] == truth_point["pixel"]
                assert got_point["value"]["x"] == pytest.approx(truth_point["x"], abs=1e-6)
                assert got_point["value"]["y"] == pytest.approx(truth_point["y"], abs=1e-6)
                assert got_point["normalized"]["x"] == pytest.approx(truth_point["norm_x"], abs=1e-6)
                assert got_point["normalized"]["y"] == pytest.approx(truth_point["norm_y"], abs=1e-6)
                assert got_point["normalization"]["method"] == "minmax"
                assert got_point["series"] == truth_point["series"]
                assert got_point["truth_series_index"] == truth_point["index"]
                assert got_point["axis_source"]["calibration"] == "page_truth_axis_calibration"
                assert got_point["axis_source"]["x_lines"]
                assert got_point["axis_source"]["y_lines"]
                assert got_point["cache_key"]

        line_detection = page_report["line_detection"]
        assert line_detection["candidate_count"] == page_report["detection_point_cost"]
        assert line_detection["points_consumed"] == page_report["detection_point_cost"]
        assert "E_LINE_TOO_SHORT" in line_detection["drop_counts"]
        merged = [
            segment
            for segment in line_detection["segments"]
            if segment["merge"]["source_count"] > 1
        ]
        assert merged
        assert all(segment["source_ids"] for segment in merged)


def test_invalid_pages_are_ordered_and_do_not_stop_batch(tmp_path: Path):
    samples = _copy_samples(tmp_path)
    truth = json.loads((samples / "page_02.json").read_text())
    truth["tables"][0]["row_lines"][2] = truth["tables"][0]["row_lines"][1]
    (samples / "page_02.json").write_text(json.dumps(truth), encoding="utf-8")
    png = samples / "page_03.png"
    png.write_bytes(b"not a png" + png.read_bytes()[9:])
    (samples / "page_04.json").unlink()

    report = run_pipeline(samples, CONFIG_PATH, tmp_path / "state")
    assert [page["file"] for page in report["pages"]] == [
        "page_01.png",
        "page_02.png",
        "page_03.png",
        "page_04.png",
    ]
    assert [page["status"] for page in report["pages"]] == [
        "ok",
        "failed",
        "failed",
        "failed",
    ]
    assert [failure["detail_code"] for failure in report["failures"]] == [
        "E_TRUTH_LINE_ORDER",
        "E_PNG_CONTAINER",
        "E_TRUTH_READ",
    ]
    assert all(failure["code"] == "E_PAGE_INVALID" for failure in report["failures"])
    assert len(report["pages"][0]["tables"][0]["cells"]) == 20


def test_detection_is_deterministic_and_records_merge_and_drop_reasons():
    image = np.full((100, 120), 255, dtype=np.uint8)
    image[10:12, 0:80] = 0
    image[30, 0:20] = 0
    image[40:100, 50:70] = 0
    binary = image < 160
    config = load_config(
        {
            "binary_threshold": 160,
            "min_line_length": 40,
            "line_merge_tolerance": 2,
        }
    )
    candidates = enumerate_candidates(binary)
    first = detect_lines(binary, config, candidates)
    second = detect_lines(binary, config, candidates)
    assert _canonical(first) == _canonical(second)
    assert first["points_consumed"] == len(candidates)
    horizontal = [segment for segment in first["segments"] if segment["axis"] == "h"]
    assert len(horizontal) == 1
    assert horizontal[0]["coord"] == 10
    assert horizontal[0]["thickness"] == 2
    assert horizontal[0]["merge"]["source_count"] == 2
    assert first["drop_counts"]["E_LINE_TOO_SHORT"] >= 1
    assert first["drop_counts"]["E_LINE_WIDTH_EXCEEDS_TOLERANCE"] >= 1
    assert any(
        drop["reason"] == "E_LINE_WIDTH_EXCEEDS_TOLERANCE"
        and drop["source_ids"]
        for drop in first["dropped_candidates"]
    )


def test_budget_pending_queue_resumes_in_filename_order(tmp_path: Path):
    config = _config()
    config["batch_detection_point_budget"] = 1500
    first = run_pipeline(SAMPLES, config, tmp_path / "state")
    assert first["status"] == "budget_incomplete"
    assert [page["status"] for page in first["pages"]] == ["ok", "ok", "pending", "pending"]
    assert [(entry["file"], entry["points"]) for entry in first["budget_ledger"]["entries"]] == [
        ("page_01.png", 1289),
        ("page_02.png", 60),
    ]
    expected_pending = [
        {
            "file": "page_03.png",
            "sort_key": "page_03.png|line",
            "code": "E_POINT_BUDGET_PENDING",
            "points_required": 1083,
        },
        {
            "file": "page_04.png",
            "sort_key": "page_04.png|line",
            "code": "E_POINT_BUDGET_PENDING",
            "points_required": 87,
        },
    ]
    assert first["pending_queue"] == expected_pending

    second = run_pipeline(SAMPLES, config, tmp_path / "state")
    assert second["status"] == "completed"
    assert second["pending_queue"] == []
    assert [page["status"] for page in second["pages"]] == ["ok", "ok", "ok", "ok"]
    assert second["recovery"]["restored_pending_queue"] == expected_pending
    assert [(entry["file"], entry["points"]) for entry in second["budget_ledger"]["entries"]] == [
        ("page_03.png", 1083),
        ("page_04.png", 87),
    ]


def test_stage_scoped_cache_invalidation(tmp_path: Path):
    base = _config()
    run_pipeline(SAMPLES, base, tmp_path / "state")

    chart_only = dict(base)
    chart_only["chart_decision_threshold"] = 0.60
    report_chart = run_pipeline(SAMPLES, chart_only, tmp_path / "state")
    for page in report_chart["pages"]:
        assert page["cache"]["line"]["status"] == "reused"
        assert page["cache"]["grid"]["status"] == "reused"
        assert page["cache"]["chart"]["status"] == "recomputed"
        assert page["cache"]["cells"]["recomputed"] == 0
        assert page["cache"]["points"]["recomputed"] > 0

    grid_only = dict(chart_only)
    grid_only["cell_min_area"] = 150
    report_grid = run_pipeline(SAMPLES, grid_only, tmp_path / "state")
    for page in report_grid["pages"]:
        assert page["cache"]["line"]["status"] == "reused"
        assert page["cache"]["grid"]["status"] == "recomputed"
        assert page["cache"]["chart"]["status"] == "reused"
        assert page["cache"]["cells"]["reused"] > 0
        assert page["cache"]["points"]["recomputed"] == 0

    normalization_only = dict(grid_only)
    normalization_only["coordinate_normalization"] = "pixel_relative"
    report_norm = run_pipeline(SAMPLES, normalization_only, tmp_path / "state")
    for page in report_norm["pages"]:
        assert page["cache"]["line"]["status"] == "reused"
        assert page["cache"]["grid"]["status"] == "reused"
        assert page["cache"]["chart"]["status"] == "recomputed"
        assert page["charts"][0]["normalization"]["method"] == "pixel_relative"


def test_fully_cached_run_does_not_enumerate_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _config()
    state = tmp_path / "state"
    first = run_pipeline(SAMPLES, config, state)
    original_costs = {page["file"]: page["detection_point_cost"] for page in first["pages"]}

    def forbidden(*args, **kwargs):
        raise AssertionError("candidate enumeration must not run for cached pages")

    monkeypatch.setattr(pipeline_module, "enumerate_candidates", forbidden)
    cached = run_pipeline(SAMPLES, config, state)
    assert cached["status"] == "completed"
    assert cached["budget_ledger"]["entries"] == []
    for page in cached["pages"]:
        assert page["detection_point_cost"] == original_costs[page["file"]]
        assert page["detection_points_consumed"] == 0
        assert page["cache"]["line"]["status"] == "reused"
        assert page["cache"]["grid"]["status"] == "reused"
        assert page["cache"]["chart"]["status"] == "reused"


def test_single_page_image_replacement_reuses_unchanged_cells_and_points(tmp_path: Path):
    samples = _copy_samples(tmp_path)
    config = _config()
    state = tmp_path / "state"
    run_pipeline(samples, config, state)

    gray, _ = decode_png((samples / "page_04.png").read_bytes())
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    rgb[540:548, 400:408, :] = 0
    (samples / "page_04.png").write_bytes(encode_png_rgb(rgb))
    report = run_pipeline(samples, config, state)

    for page in report["pages"][:3]:
        assert page["cache"]["line"]["status"] == "reused"
        assert page["cache"]["grid"]["status"] == "reused"
        assert page["cache"]["chart"]["status"] == "reused"
    changed = report["pages"][3]
    assert changed["cache"]["line"]["status"] == "recomputed"
    assert changed["cache"]["grid"]["status"] == "recomputed"
    assert changed["cache"]["chart"]["status"] == "recomputed"
    assert changed["cache"]["cells"] == {"reused": 14, "recomputed": 0}
    assert changed["cache"]["points"] == {"reused": 7, "recomputed": 0}


def test_forced_termination_recovers_only_committed_generation(tmp_path: Path):
    config = _config()
    baseline = run_pipeline(SAMPLES, config, tmp_path / "baseline")
    crash_state = tmp_path / "crash"
    with pytest.raises(SimulatedTermination):
        run_pipeline(
            SAMPLES,
            config,
            crash_state,
            fail_after_commits=1,
        )
    assert not (crash_state / "report.json").exists()
    pointer = json.loads((crash_state / "snapshots" / "CURRENT.json").read_text())
    manifest = json.loads(
        (
            crash_state
            / "snapshots"
            / "generations"
            / pointer["generation"]
            / "manifest.json"
        ).read_text()
    )
    assert manifest["status"] == "active"
    assert manifest["completed_pages"] == ["page_01.png", "page_02.png"]
    assert [entry["file"] for entry in manifest["budget_plan"]["entries"]] == [
        "page_01.png",
        "page_02.png",
        "page_03.png",
        "page_04.png",
    ]

    resumed = run_pipeline(SAMPLES, config, crash_state)
    assert resumed["status"] == "completed"
    assert resumed["recovery"]["resumed_active"] is True
    assert resumed["recovery"]["snapshot_version"] == pointer["global_version"]
    assert resumed["recovery"]["recovery_point_pages"] == [
        "page_01.png",
        "page_02.png",
    ]
    assert _canonical(_business_view(resumed)) == _canonical(_business_view(baseline))


def test_chart_quota_and_overlap_priority(tmp_path: Path):
    quota_config = _config()
    quota_config["page_chart_point_quota"] = 3
    quota_report = run_pipeline(SAMPLES, quota_config, tmp_path / "quota")
    for page in quota_report["pages"]:
        chart = page["charts"][0]
        assert len(chart["points"]) == 3
        assert chart["quota"]["truncated"] is True
        assert page["warnings"][0]["code"] == "E_CHART_POINT_QUOTA_TRUNCATED"

    samples = _copy_samples(tmp_path)
    truth_path = samples / "page_01.json"
    truth = json.loads(truth_path.read_text())
    truth["charts"][0]["region"][0] = truth["tables"][0]["region"][2] - 20
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    overlap_report = run_pipeline(samples, _config(), tmp_path / "overlap")
    page = overlap_report["pages"][0]
    assert page["status"] == "ok"
    overlap = page["charts"][0]["classification"]["overlap"]
    assert overlap
    assert overlap[0]["priority"] == "table"
    assert overlap[0]["rationale"] == "TABLE_PRIORITY_ON_OVERLAP"
    assert page["charts"][0]["classification"]["excluded_boxes"]


def test_concurrency_limit_is_respected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = _config()
    config["thread_pool_concurrency"] = 3
    state = tmp_path / "state"
    original = pipeline_module._process_valid_page
    active = 0
    maximum = 0
    guard = threading.Lock()

    def wrapped(*args, **kwargs):
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        try:
            time.sleep(0.15)
            return original(*args, **kwargs)
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(pipeline_module, "_process_valid_page", wrapped)
    report = run_pipeline(SAMPLES, config, state)
    assert report["status"] == "completed"
    assert maximum <= 3
    assert maximum >= 2


def test_atomic_pointer_and_snapshot_artifact_integrity(tmp_path: Path):
    config = _config()
    report = run_pipeline(SAMPLES, config, tmp_path / "state")
    assert report["status"] == "completed"
    pointer = json.loads((tmp_path / "state" / "snapshots" / "CURRENT.json").read_text())
    generation = tmp_path / "state" / "snapshots" / "generations" / pointer["generation"]
    manifest_bytes = (generation / "manifest.json").read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == pointer["manifest_sha256"]
    manifest = json.loads(manifest_bytes)
    for relative, expected_hash in manifest["artifact_files"].items():
        payload = (generation / relative).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected_hash
    assert manifest["global_version"] == pointer["global_version"]
    assert manifest["config_hash"] == report["config_hash"]
    assert manifest["completed_pages"] == report["page_order"]

    partial = tmp_path / "state" / "snapshots" / "generations" / ".tmp-interrupted"
    partial.mkdir()
    (partial / "manifest.json").write_text("{}", encoding="utf-8")
    assert json.loads((tmp_path / "state" / "snapshots" / "CURRENT.json").read_text()) == pointer

    check = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_windows_atomic.py")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0
    result = json.loads(check.stdout)
    assert result["temp_dir_created"] is True
    assert result["atomic_replace_supported"] is True


def test_config_and_truth_validation_errors():
    with pytest.raises(ConfigError):
        load_config({"binary_threshold": 999})
    truth = json.loads((SAMPLES / "page_01.json").read_text())
    truth["tables"][0]["col_lines"].reverse()
    with pytest.raises(PageInvalid) as error:
        normalize_truth(truth, 960, 600)
    assert error.value.detail_code == "E_TRUTH_LINE_ORDER"

    truth = json.loads((SAMPLES / "page_01.json").read_text())
    truth["tables"][0]["cells"][0]["text"] = ""
    with pytest.raises(PageInvalid) as error:
        normalize_truth(truth, 960, 600)
    assert error.value.detail_code == "E_TRUTH_CELL_TEXT"

    page = load_page(SAMPLES / "page_01.png", SAMPLES / "page_01.json", load_config(CONFIG_PATH))
    assert page.gray.dtype == np.uint8
    assert page.gray.ndim == 2
