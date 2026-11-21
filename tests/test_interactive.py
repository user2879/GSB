import copy
import time

import workbench.interactive
from workbench import codes
from workbench.interactive import Workbench
from workbench.pipeline import Pipeline


def make_workbench(dataset, tmp_path, **config_overrides):
    config = copy.deepcopy(dataset["config"])
    config.update(config_overrides)
    return Workbench(dataset["turns"], dataset["acoustics"], dataset["tools"],
                     dataset["lexicon"], config, dataset["load_failures"],
                     str(tmp_path))


def test_command_run_returns_summary(dataset, tmp_path):
    wb = make_workbench(dataset, tmp_path)
    response = wb.apply("run")
    assert response["ok"]
    summary = response["summary"]
    assert summary["completed_turns"] == summary["total_turns"] == 24
    assert "confidence_distribution" in summary
    assert summary["version"] == 1


def test_preview_budget_returns_stale_summary(dataset, tmp_path, monkeypatch):
    wb = make_workbench(dataset, tmp_path)
    wb.apply("run")  # warm summary at version 1
    wb.config["interactive_latency_ms"] = 1  # force budget exhaustion

    original_run = Pipeline.run

    def slow_run(self):
        time.sleep(0.05)
        return original_run(self)

    monkeypatch.setattr(Pipeline, "run", slow_run)
    response = wb.apply("set-weight text 0.65")
    preview = response["preview"]
    assert preview["code"] == codes.E_PREVIEW_BUDGET
    assert preview["version"] == 1            # explicitly versioned stale result
    assert preview["requested_version"] == 2
    wb._worker.join(timeout=5)                # let the worker finish


def test_startup_stale_snapshot_flagged(dataset, tmp_path):
    wb = make_workbench(dataset, tmp_path)
    wb.apply("run")
    # a new workbench over the same run dir with a changed config
    wb2 = make_workbench(dataset, tmp_path)
    wb2.config["weights"]["text"] = 0.9
    wb2.startup_findings = wb2.snapshotter.check_stale(
        workbench.interactive.fingerprint(wb2.config))
    assert any(f["code"] == codes.E_SNAPSHOT_STALE for f in wb2.startup_findings)


def test_invalid_precision_command_rejected(dataset, tmp_path):
    wb = make_workbench(dataset, tmp_path)
    response = wb.apply("set-precision 7 2")
    assert response["ok"] is False
    assert response["code"] == codes.E_LOAD_PRECISION


def test_snapshot_committed_each_interval(dataset, tmp_path):
    import os
    wb = make_workbench(dataset, tmp_path)
    wb.apply("run")
    wb.apply("set-weight text 0.65")
    manifest = wb.snapshotter.load_manifest()
    assert manifest["global_version"] == 2
    assert os.path.exists(os.path.join(wb.snapshotter.snap_dir, "clip.json"))
