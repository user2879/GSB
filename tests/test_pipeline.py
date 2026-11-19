from __future__ import annotations

from pathlib import Path

from hls_workbench.config import (
    BudgetConfig,
    ClientBufferConfig,
    DecisionConfig,
    EncoderConfig,
    FaultConfig,
    MediaConfig,
    RetryConfig,
    WorkbenchConfig,
)
from hls_workbench.demo import _hash_tree, _tree_hashes
from hls_workbench.errors import ERROR_CODES
from hls_workbench.pipeline import run_pipeline


def test_end_to_end_outputs_are_byte_identical(tmp_path: Path) -> None:
    config = WorkbenchConfig(
        media=MediaConfig(duration_seconds=2.0, slice_seconds=1.0, width=320, height=180),
        encoder=EncoderConfig(),
        budget=BudgetConfig(max_workers=3, memory_budget_mb=180, cpu_budget_ms=180),
        retry=RetryConfig(max_retries=2, base_delay_ms=1),
        fault=FaultConfig(task_key="720p:00001", failures=3),
        client_buffer=ClientBufferConfig(capacity_ms=6000, threshold_ms=1000, step_ms=100),
        decision=DecisionConfig(bandwidth_kbps=2600),
    )
    first = run_pipeline(config, tmp_path / "first", sleep_fn=lambda _: None)
    second = run_pipeline(config, tmp_path / "second", sleep_fn=lambda _: None)

    assert first.report_path.read_bytes() == second.report_path.read_bytes()
    assert first.report_sha256 == second.report_sha256
    hashes_a = _tree_hashes(first.root)
    hashes_b = _tree_hashes(second.root)
    assert hashes_a == hashes_b
    assert _hash_tree(hashes_a) == _hash_tree(hashes_b)
    assert len(first.artifacts) == 4
    assert first.report["steps"]["transcode"]["timeline_check_count"] == 4
    assert (first.root / "hls" / "master.m3u8").exists()
    assert (first.root / "hls" / "360p" / "index.m3u8").read_text(encoding="utf-8").count("#EXTINF") == 2
    assert first.report["steps"]["transcode"]["isolation_count"] == 1
    assert first.report["steps"]["transcode"]["budget_denial_count"] > 0


def test_required_error_codes_are_defined() -> None:
    required = {
        "E_BUDGET_DENIED",
        "E_PTS_GAP",
        "E_TASK_DUPLICATE",
        "E_RETRY_EXHAUSTED",
        "E_BUFFER_UNDERRUN",
        "E_MEDIA_FORMAT",
    }
    assert required <= set(ERROR_CODES)
