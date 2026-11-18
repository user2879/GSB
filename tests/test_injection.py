from __future__ import annotations

from pathlib import Path

from remix_workbench.engine import RemixEngine, compare_signatures, render_signature
from remix_workbench.injections import load_injections


def test_all_injections_hit_expected_business_codes(
    config: dict, samples_root: Path, injections_path: Path, tmp_path: Path
) -> None:
    injections = load_injections(injections_path)
    workdir = tmp_path / "injected"
    first = RemixEngine(
        samples_root=samples_root,
        config=config,
        workdir=workdir,
        injections=injections,
        force_workers=4,
    ).execute()
    assert first.report is not None
    summary = {item["id"]: item for item in first.report["injection_summary"]}
    assert set(summary) == {"INJ001", "INJ002", "INJ003", "INJ004", "INJ005"}
    for item in summary.values():
        assert item["hit"] is True
        assert item["expected_code_matched"] is True
        assert item["expected_code"] in item["observed_codes"]
    codes = {failure["code"] for failure in first.report["failures"]}
    assert codes == {"E_TRACK_FORMAT", "E_MIX_CONFIG"}
    assert first.report["failure_order"] == sorted(
        first.report["failure_order"], key=lambda value: tuple(value.split("|"))
    )
    # Reusing parse artifacts must retain per-track business failures.
    second = RemixEngine(
        samples_root=samples_root,
        config=config,
        workdir=workdir,
        injections=injections,
        force_workers=4,
    ).execute()
    assert second.report is not None
    assert second.report["cache_summary"]["reuse_count"] > 0
    for item in second.report["injection_summary"]:
        assert item["hit"] is True
    assert compare_signatures(
        render_signature(first.report, first.workdir),
        render_signature(second.report, second.workdir),
    )["matches"] is True

