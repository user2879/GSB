"""Shared deterministic builders for end-to-end pipeline tests."""
from __future__ import annotations

from pathlib import Path

from qcalib import loaders, receipts
from qcalib.commands import apply_commands
from qcalib.errors import FailureList
from qcalib.pipeline import PipelineInputs
from qcalib.runner import run_pipeline
from qcalib.snapshot import SnapshotStore

ROOT = Path(__file__).resolve().parents[1]


def build_inputs(commands_path: str | Path | None = None,
                 config_path: str | Path | None = None) -> tuple[PipelineInputs, FailureList]:
    failures = FailureList()
    config_path = Path(config_path) if config_path else ROOT / "configs" / "pipeline.yaml"
    layers = loaders.load_weight_package(
        ROOT / "samples" / "weights.jsonl", failures)
    calibration = loaders.load_calibration(
        ROOT / "samples" / "calibration.json", layers, failures)
    config = loaders.load_config(config_path)
    if commands_path is not None:
        commands = loaders.load_command_stream(commands_path, failures)
        config, layers = apply_commands(config, layers, commands, failures)
    receipt_entries = receipts.parse_receipt_dir(config.receipts_dir, failures)
    return (PipelineInputs(layers=layers, calibration=calibration,
                           config=config, receipt_entries=receipt_entries,
                           input_failures=failures.to_list()),
            failures)


def execute(store_path: str | Path,
            commands_path: str | Path | None = None,
            previous_cache: dict | None = None,
            fault_point: str | None = None) -> dict:
    inputs, pre_failures = build_inputs(commands_path)
    store = SnapshotStore(store_path, fault_point=fault_point)
    result = run_pipeline(inputs, store, previous_cache=previous_cache)
    return result


def comparable(payload: dict) -> dict:
    # Cache internals are an implementation reuse mechanism, not ledger fields.
    return {key: value for key, value in payload.items() if key != "cache"}
