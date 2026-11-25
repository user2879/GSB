import pytest

from qcalib.hashing import stable_json
from qcalib.pipeline import config_fingerprint
from qcalib.snapshot import ForcedAbort, SnapshotStore

from tests._helpers import build_inputs, comparable, execute


def test_snapshot_stage_files_are_committed(tmp_path):
    result = execute(tmp_path / "state", None)
    store = SnapshotStore(tmp_path / "state")
    manifest = store.load_manifest()
    assert manifest["committed"] == [
        "parse_validate", "quantile_calibration",
        "fixed_point_requantization", "distillation_correction",
        "ledger_aggregation"]
    for stage in manifest["committed"]:
        assert (store.stage_root / stage / "data.json").exists()
    assert not list(store.tmp_root.glob("*"))


def test_forced_abort_recovery_matches_full_run(tmp_path):
    inputs, _ = build_inputs(None)
    fault_state = tmp_path / "fault_state"
    store = SnapshotStore(fault_state,
                          fault_point="before_commit:ledger_aggregation")
    from qcalib.runner import run_pipeline
    with pytest.raises(ForcedAbort):
        run_pipeline(inputs, store)

    # Earlier stages remain durable; the final ledger was never swapped in.
    partial = SnapshotStore(fault_state).load_manifest()
    assert "ledger_aggregation" not in partial["committed"]
    assert "distillation_correction" in partial["committed"]

    full = execute(tmp_path / "clean_state", None)
    config_hash = config_fingerprint(inputs.config)
    content_versions = {layer.name: layer.content_version
                        for layer in inputs.layers}
    recoverable = SnapshotStore(fault_state).recoverable_stages(
        config_hash, content_versions)
    assert "distillation_correction" in recoverable

    # Resume without the fault and confirm the rebuilt ledger matches full.
    resumed_store = SnapshotStore(tmp_path / "resumed_state")
    # Copy durable stage state into the resumed store location.
    import shutil
    shutil.copytree(fault_state, tmp_path / "resumed_state",
                    dirs_exist_ok=True)
    from qcalib.runner import run_pipeline as run
    resumed = run(inputs, SnapshotStore(tmp_path / "resumed_state"))
    assert stable_json(comparable(resumed)) == stable_json(comparable(full))


def test_recovery_rejects_stale_config_hash(tmp_path):
    result = execute(tmp_path / "state", None)
    inputs, _ = build_inputs(None)
    store = SnapshotStore(tmp_path / "state")
    stale = store.recoverable_stages("cfg-does-not-exist",
                                     result["content_versions"])
    assert stale == []
