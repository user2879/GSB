from qcalib import loaders, receipts
from qcalib.errors import FailureList
from qcalib.pipeline import PipelineInputs
from qcalib.runner import diagnostic_report, run_pipeline
from qcalib.snapshot import SnapshotStore
from tests._helpers import ROOT


def test_constrained_budget_aborts_layer_and_keeps_pending(tmp_path):
    failures = FailureList()
    layers = loaders.load_weight_package(
        ROOT / "samples" / "weights.jsonl", failures)
    calibration = loaders.load_calibration(
        ROOT / "samples" / "calibration.json", layers, failures)
    config = loaders.load_config(ROOT / "configs" / "pipeline_constrained.yaml")
    entries = receipts.parse_receipt_dir(config.receipts_dir, failures)
    inputs = PipelineInputs(
        layers=layers, calibration=calibration, config=config,
        receipt_entries=entries, input_failures=failures.to_list())
    result = run_pipeline(inputs, SnapshotStore(tmp_path / "state"))
    codes = {item["code"] for item in result["failures"]}
    assert "E_QUOTA_EXCEEDED" in codes
    assert "E_BUDGET_PENDING" in codes
    assert "fusion_block" in result["budget"]["quota_aborted_layers"]
    assert result["budget"]["pending_tasks"]
    report = diagnostic_report(result)
    assert report["pending_task_count"] >= 1
    assert report["quota_aborted_layers"] == ["fusion_block"]
