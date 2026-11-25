import numpy as np

from qcalib import loaders, receipts
from qcalib.errors import E_ZERO_SCALE, FailureList
from qcalib.models import WeightLayer
from qcalib.pipeline import PipelineInputs
from qcalib.runner import run_pipeline
from qcalib.snapshot import SnapshotStore
from tests._helpers import ROOT


def test_all_zero_layer_uses_fallback_without_crashing(tmp_path):
    failures = FailureList()
    layers = loaders.load_weight_package(
        ROOT / "samples" / "weights.jsonl", failures)
    zero_layer = WeightLayer("zero_layer", (4, 8),
                             np.zeros((4, 8)), 5, 8)
    layers.append(zero_layer)
    calibration = loaders.load_calibration(
        ROOT / "samples" / "calibration.json", layers, failures)
    config = loaders.load_config(ROOT / "configs" / "pipeline.yaml")
    entries = receipts.parse_receipt_dir(config.receipts_dir, failures)
    inputs = PipelineInputs(
        layers=layers, calibration=calibration, config=config,
        receipt_entries=entries, input_failures=failures.to_list())
    result = run_pipeline(inputs, SnapshotStore(tmp_path / "state"))
    zero_artifact = result["layers"]["zero_layer"]
    assert all(group["zero_scale_fallback"]
               for group in zero_artifact["group_calibration"])
    assert any(item["code"] == E_ZERO_SCALE for item in result["failures"])
    assert all(np.isfinite(group["scale"])
               for group in zero_artifact["group_calibration"])
