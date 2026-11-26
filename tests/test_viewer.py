from __future__ import annotations

from pathlib import Path

from voice_diag.pipeline import DiagnosticPipeline, PipelineOptions
from voice_diag.synthesize import generate_demo
from voice_diag.viewer import cache_rows, interval_rows, load_view_data


def test_viewer_loads_committed_review_data(tmp_path: Path) -> None:
    sample = tmp_path / "sample"
    output = tmp_path / "out"
    generate_demo(sample)
    result = DiagnosticPipeline(
        sample / "config.json",
        sample / "manifest.json",
        output,
        options=PipelineOptions(max_batches=32),
    ).run()
    assert result["status"] == "complete"

    data = load_view_data(output)
    report = data["report"]
    assert data["final"] is not None
    assert data["state"] is not None
    assert set(data["receipts"]) == {"evidence.a1", "evidence.a2.json"}
    rows = interval_rows(report)
    assert len(rows) == 4
    assert all(row[6] == "yes" for row in rows)
    assert cache_rows(report, limit=3) == []
