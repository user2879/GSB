import json
import os

from workbench import codes
from workbench.snapshot import Snapshotter, snapshot_artifacts


def test_commit_writes_manifest_and_artifacts(tmp_path):
    snap = Snapshotter(str(tmp_path))
    manifest = snap.commit(3, "cfg-hash", {"report": {"v": 1}, "clip": {"T001": 5}})
    assert manifest["global_version"] == 3
    assert manifest["config_hash"] == "cfg-hash"
    assert set(manifest["artifacts"]) == {"report", "clip"}
    assert os.path.exists(snap.manifest_path)
    assert os.path.exists(os.path.join(snap.snap_dir, "report.json"))
    # tmp dir must be drained after wholesale replacement
    assert os.listdir(snap.tmp_dir) == []


def test_wholesale_replacement_on_second_commit(tmp_path):
    snap = Snapshotter(str(tmp_path))
    snap.commit(1, "hash-a", {"report": {"v": 1}})
    snap.commit(2, "hash-b", {"report": {"v": 2}})
    with open(os.path.join(snap.snap_dir, "report.json")) as handle:
        assert json.load(handle)["v"] == 2
    assert snap.load_manifest()["global_version"] == 2


def test_stale_detection(tmp_path):
    snap = Snapshotter(str(tmp_path))
    snap.commit(1, "hash-a", {"report": {"v": 1}})
    findings = snap.check_stale("hash-b")
    assert len(findings) == 1
    assert findings[0]["code"] == codes.E_SNAPSHOT_STALE
    assert findings[0]["snapshot_version"] == 1
    assert snap.check_stale("hash-a") == []


def test_snapshot_artifacts_from_report(dataset):
    from workbench.cache import StageCache
    from workbench.pipeline import Pipeline
    report = Pipeline(dataset["turns"], dataset["acoustics"], dataset["tools"],
                      dataset["lexicon"], dataset["config"],
                      cache=StageCache()).run()
    artifacts = snapshot_artifacts(report)
    assert set(artifacts) == {"normalize_stats", "quantized", "weighted_sum",
                              "clip", "report"}
    assert set(artifacts["quantized"]) == {t["turn_id"] for t in dataset["turns"]}
