import pytest

from workbench.pipeline import EXIT_FORCED_STOP, ForcedStop, Pipeline
from workbench.verify import fusion_artifacts_equal, reports_equal


def test_forced_stop_exit_code_and_resume(small_samples, tmp_path):
    work = tmp_path / "w"
    with pytest.raises(ForcedStop) as exc:
        Pipeline(small_samples, work, stop_after={"tiles": 2}).run()
    assert exc.value.code == EXIT_FORCED_STOP
    # snapshot holds exactly the completed tiles; no report promise
    assert not (work / "snapshot" / "global" / "report.json").exists()
    report = Pipeline(small_samples, work).run()
    full = Pipeline(small_samples, tmp_path / "wf", full=True).run()
    ok, diffs = reports_equal(report, full)
    assert ok, diffs
    ok, diffs = fusion_artifacts_equal(work, tmp_path / "wf")
    assert ok, diffs


def test_forced_stop_during_fusion(small_samples, tmp_path):
    work = tmp_path / "w"
    with pytest.raises(ForcedStop):
        Pipeline(small_samples, work, stop_after={"fusions": 1}).run()
    report = Pipeline(small_samples, work).run()
    full = Pipeline(small_samples, tmp_path / "wf", full=True).run()
    ok, diffs = reports_equal(report, full)
    assert ok, diffs
