from __future__ import annotations

import copy
import functools
import os
import shutil
import signal
import threading
import time
from pathlib import Path

import pytest

from layout_reader import cli
from layout_reader.columns import infer_columns
from layout_reader.config import load_config
from layout_reader.faults import load_fault_registry
from layout_reader.models import BBox, Block
from layout_reader.pipeline import run_pipeline
from layout_reader.roles import assign_roles


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return load_config(ROOT / "configs" / "default.json")


def _run(sample_dir: Path, state_dir: Path, config=None, faults=None, **kwargs):
    return run_pipeline(
        input_dir=sample_dir,
        config=config or _config(),
        fault_registry=load_fault_registry(faults),
        state_dir=state_dir,
        **kwargs,
    )


def test_full_run_matches_truth_and_layout_expectations(sample_dir: Path, tmp_path: Path):
    report = _run(sample_dir, tmp_path / "state")
    assert report["status"] == "completed"
    assert report["summary"]["completed_page_count"] == 6
    assert report["summary"]["failed_page_count"] == 0
    assert report["truth_validation"]["passed"] is True
    assert report["truth_validation"]["pages_passed"] == 6

    pages = {page["page"]: page for page in report["pages"]}
    assert list(pages) == [f"page_{index:03d}.png" for index in range(1, 7)]
    assert pages["page_001.png"]["diagnostics"]["summary"]["column_layout"] == "single"
    assert pages["page_002.png"]["diagnostics"]["summary"]["column_count"] == 2
    assert pages["page_003.png"]["diagnostics"]["summary"]["column_count"] == 3
    assert pages["page_004.png"]["diagnostics"]["summary"]["spanning_block_count"] >= 2
    assert len(pages["page_006.png"]["reading_order"]) == 30

    for page in report["pages"]:
        blocks = page["blocks"]
        assert [block["block_id"] for block in blocks] == sorted(
            block["block_id"] for block in blocks
        )
        orders = sorted(block["reading_order"] for block in blocks)
        assert orders == list(range(1, len(blocks) + 1))
        assert {block["role"] for block in blocks} <= {
            "image",
            "title",
            "body",
            "header_footer",
        }
        assert all(block["role_evidence"] for block in blocks)


def test_serial_and_parallel_results_are_field_identical(sample_dir: Path, tmp_path: Path):
    serial = _run(sample_dir, tmp_path / "serial", workers=1)
    parallel = _run(sample_dir, tmp_path / "parallel", workers=4)
    assert serial["execution"]["mode"] == "serial"
    assert parallel["execution"]["mode"] == "parallel"
    assert serial["pages"] == parallel["pages"]
    assert serial["failures"] == parallel["failures"]
    assert serial["injections"] == parallel["injections"]
    assert serial["page_cache"] == parallel["page_cache"]
    assert serial["result_digest"] == parallel["result_digest"]


def test_incremental_consistency_and_selective_invalidation(
    sample_dir: Path, tmp_path: Path
):
    input_dir = tmp_path / "samples"
    shutil.copytree(sample_dir, input_dir)
    incremental_state = tmp_path / "incremental"
    initial = _config()
    first = _run(input_dir, incremental_state, initial)
    assert first["summary"]["recomputed_stage_count"] == 30

    threshold_config = copy.deepcopy(initial)
    threshold_config["binarization_threshold"] = 170
    threshold_report = _run(input_dir, incremental_state, threshold_config)
    assert threshold_report["summary"]["reused_stage_count"] == 6
    assert threshold_report["summary"]["recomputed_stage_count"] == 24
    by_page = {item["page"]: item["stages"] for item in threshold_report["page_cache"]}
    assert by_page["page_001.png"]["image"]["status"] == "reused"
    assert by_page["page_001.png"]["detect"]["status"] == "recomputed"

    full_threshold = _run(input_dir, tmp_path / "full-threshold", threshold_config)
    assert threshold_report["result_digest"] == full_threshold["result_digest"]
    assert threshold_report["pages"] == full_threshold["pages"]

    order_config = copy.deepcopy(threshold_config)
    order_config["reading_order_rule"] = "region_first"
    order_report = _run(input_dir, incremental_state, order_config)
    assert order_report["summary"]["reused_stage_count"] == 24
    assert order_report["summary"]["recomputed_stage_count"] == 6
    by_page = {item["page"]: item["stages"] for item in order_report["page_cache"]}
    assert by_page["page_001.png"]["columns"]["status"] == "reused"
    assert by_page["page_001.png"]["order"]["status"] == "recomputed"

    shutil.copyfile(input_dir / "page_002.png", input_dir / "page_006.png")
    replaced = _run(input_dir, incremental_state, order_config)
    by_page = {item["page"]: item["stages"] for item in replaced["page_cache"]}
    assert all(stage["status"] == "reused" for stage in by_page["page_001.png"].values())
    assert all(
        stage["status"] == "recomputed" for stage in by_page["page_006.png"].values()
    )

    shutil.copyfile(input_dir / "page_001.png", input_dir / "page_007.png")
    (input_dir / "page_005.png").unlink()
    final_incremental = _run(input_dir, incremental_state, order_config)
    final_full = _run(input_dir, tmp_path / "full-final", order_config)
    assert final_incremental["pages"] == final_full["pages"]
    assert final_incremental["failures"] == final_full["failures"]
    assert final_incremental["result_digest"] == final_full["result_digest"]
    assert [page["page"] for page in final_incremental["pages"]] == [
        "page_001.png",
        "page_002.png",
        "page_003.png",
        "page_004.png",
        "page_006.png",
        "page_007.png",
    ]
    assert final_incremental["summary"]["truth_validation_passed"] is False


def test_fault_injection_hits_map_codes_and_stay_page_local(
    sample_dir: Path, tmp_path: Path
):
    clean = _run(sample_dir, tmp_path / "clean")
    injected = _run(
        sample_dir,
        tmp_path / "injected",
        faults=ROOT / "faults" / "sample_injections.json",
    )
    assert injected["summary"]["failed_page_count"] == 4
    assert injected["summary"]["injection_hit_count"] == 4
    failures = {failure["page"]: failure for failure in injected["failures"]}
    assert failures["page_002.png"]["code"] == "E_IMAGE_INVALID"
    assert failures["page_003.png"]["code"] == "E_BLOCK_BOUNDS"
    assert failures["page_004.png"]["code"] == "E_COLUMN_GAP_INVALID"
    assert failures["page_005.png"]["code"] == "E_ORDER_DUPLICATE"
    assert all(record["hit"] for record in injected["injections"])
    assert [record["id"] for record in injected["injections"]] == [
        "F001",
        "F002",
        "F003",
        "F004",
    ]
    assert [record["business_code"] for record in injected["injections"]] == [
        "E_IMAGE_INVALID",
        "E_BLOCK_BOUNDS",
        "E_COLUMN_GAP_INVALID",
        "E_ORDER_DUPLICATE",
    ]

    clean_pages = {page["page"]: page for page in clean["pages"]}
    injected_pages = {page["page"]: page for page in injected["pages"]}
    assert set(injected_pages) == {"page_001.png", "page_006.png"}
    for page_name in injected_pages:
        assert injected_pages[page_name] == clean_pages[page_name]


def test_column_boundary_conflict_uses_stability_then_count():
    config = _config()
    config["column_gap_threshold"] = 24
    config["column_candidate_max_width_ratio"] = 0.34
    config["column_span_ratio"] = 0.62
    config["column_span_min_overlap_ratio"] = 0.15
    blocks = [
        Block("b1", BBox(0, 10, 100, 26), 0.2, 0.1, role="body"),
        Block("b2", BBox(200, 10, 300, 26), 0.2, 0.1, role="body"),
        Block("b3", BBox(0, 50, 100, 66), 0.2, 0.1, role="body"),
        Block("b4", BBox(200, 50, 300, 66), 0.2, 0.1, role="body"),
        Block("b5", BBox(90, 100, 210, 116), 0.2, 0.1, role="body"),
    ]
    _, diagnostics = infer_columns(blocks, 400, 600, config)
    assert diagnostics["column_count"] == 2
    assert diagnostics["conflict_count"] >= 1
    equal_conflicts = [
        conflict
        for conflict in diagnostics["conflicts"]
        if conflict["reason"] == "equal_overlap_between_columns"
    ]
    assert equal_conflicts
    assert equal_conflicts[0]["decision_order"] == [
        "greater_width_stability",
        "greater_existing_block_count",
        "leftmost_tie_break",
    ]


def test_role_boundaries_are_inclusive_and_priority_breaks_ties():
    config = _config()
    role = config["role_detection"]
    image_density = float(role["image_min_ink_density"])
    image_periodicity = float(role["image_max_projection_periodicity"])
    ratio_min = float(role["title_body_height_ratio"]["min"])
    blocks = [
        Block("b1", BBox(40, 300, 300, 316), 0.2, 0.1),
        Block(
            "b2",
            BBox(40, 360, 300, 376),
            image_density,
            image_periodicity,
        ),
        Block("b3", BBox(40, 420, 300, 420 + int(16 * ratio_min)), 0.2, 0.1),
    ]
    blocks, diagnostics = assign_roles(blocks, 960, config)
    assert [block.role for block in blocks] == ["body", "image", "title"]
    assert blocks[1].role_evidence["decision_rule"] == "image_density_boundary_inclusive"
    assert blocks[1].role_evidence["tie_break_applied"] is True
    assert blocks[2].role_evidence["decision_rule"] == "title_height_ratio_boundary_inclusive"
    assert diagnostics["body_median_height"] == 16.0


def test_graceful_stop_commits_progress_and_resume_matches_full(
    sample_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import layout_reader.pipeline as pipeline_module

    config = _config()
    config["thread_pool_workers"] = 1
    config["pages_per_batch"] = 2
    config["snapshot_interval_pages"] = 2
    stop_event = threading.Event()
    original = pipeline_module.detect_blocks
    calls = {"count": 0}

    @functools.wraps(original)
    def stopping_detect(*args, **kwargs):
        result = original(*args, **kwargs)
        calls["count"] += 1
        if calls["count"] >= 2:
            stop_event.set()
        return result

    monkeypatch.setattr(pipeline_module, "detect_blocks", stopping_detect)
    interrupted = run_pipeline(
        input_dir=sample_dir,
        config=config,
        fault_registry=load_fault_registry(None),
        state_dir=tmp_path / "interrupted",
        stop_event=stop_event,
    )
    assert interrupted["status"] == "interrupted"
    assert interrupted["summary"]["processed_page_count"] == 2
    assert interrupted["pending_pages"]
    snapshot = (tmp_path / "interrupted" / "manifest.json").read_text(encoding="utf-8")
    assert '"interrupted":true' in snapshot.replace("\n", "").replace(" ", "")

    monkeypatch.setattr(pipeline_module, "detect_blocks", original)
    resumed = run_pipeline(
        input_dir=sample_dir,
        config=config,
        fault_registry=load_fault_registry(None),
        state_dir=tmp_path / "interrupted",
    )
    full = run_pipeline(
        input_dir=sample_dir,
        config=config,
        fault_registry=load_fault_registry(None),
        state_dir=tmp_path / "full",
    )
    assert resumed["recovery"]["resumed"] is True
    assert resumed["recovery"]["base_was_interrupted"] is True
    assert resumed["recovery"]["recovered_pages"] == ["page_001.png", "page_002.png"]
    assert resumed["summary"]["reused_stage_count"] >= 10
    assert resumed["result_digest"] == full["result_digest"]


def test_cli_sigterm_returns_zero_and_writes_interrupted_report(
    sample_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import layout_reader.pipeline as pipeline_module

    config_path = tmp_path / "slow_config.json"
    config = _config()
    config.update(
        {
            "thread_pool_workers": 1,
            "pages_per_batch": 1,
            "snapshot_interval_pages": 1,
        }
    )
    config_path.write_text(__import__("json").dumps(config), encoding="utf-8")
    original = pipeline_module.detect_blocks

    @functools.wraps(original)
    def slow_detect(*args, **kwargs):
        time.sleep(0.12)
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "detect_blocks", slow_detect)

    def send_term() -> None:
        time.sleep(0.32)
        os.kill(os.getpid(), signal.SIGTERM)

    timer = threading.Thread(target=send_term, daemon=True)
    timer.start()
    report_path = tmp_path / "cli-report.json"
    exit_code = cli.main(
        [
            "analyze",
            "--input",
            str(sample_dir),
            "--config",
            str(config_path),
            "--state",
            str(tmp_path / "cli-state"),
            "--report",
            str(report_path),
            "--serial",
        ]
    )
    timer.join(timeout=2)
    assert exit_code == 0
    import json

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "interrupted"
    assert report["summary"]["processed_page_count"] >= 1
