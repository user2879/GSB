from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import yaml

from motion_pipeline.config import load_config
from motion_pipeline.runner import PipelineRunner, RunRequest, RunResult
from motion_pipeline.samples import load_sequences


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CONFIG = ROOT / "samples" / "config.yaml"
SAMPLE_SEQUENCES = ROOT / "samples" / "sequences.jsonl"


def sample_config_dict() -> dict[str, Any]:
    return yaml.safe_load(SAMPLE_CONFIG.read_text(encoding="utf-8"))


def write_config(
    path: Path, mutate: Callable[[dict[str, Any]], None] | None = None
) -> Path:
    source = sample_config_dict()
    if mutate is not None:
        mutate(source)
    path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    return path


def run_pipeline(
    *,
    config_path: Path,
    sequence_path: Path,
    tier: int,
    state_dir: Path,
    output_dir: Path,
    fresh: bool = False,
    expected_version: int | None = None,
) -> RunResult:
    config_result = load_config(config_path)
    assert config_result.ok, config_result.failures
    sequence_result = load_sequences(sequence_path)
    assert sequence_result.sequences, sequence_result.failures
    return PipelineRunner(
        RunRequest(
            config=config_result.config,
            sequences=sequence_result.sequences,
            failures=sequence_result.failures,
            scale_tier=tier,
            state_dir=state_dir,
            output_dir=output_dir,
            fresh=fresh,
            expected_version=expected_version,
        )
    ).run()


def replace_sequence(
    source_path: Path,
    destination_path: Path,
    *,
    sequence_index: int,
    mutate: Callable[[dict[str, Any]], None],
) -> Path:
    lines = source_path.read_text(encoding="utf-8").splitlines()
    sequence = json.loads(lines[sequence_index])
    mutate(sequence)
    lines[sequence_index] = json.dumps(sequence, ensure_ascii=True, separators=(",", ":"))
    destination_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination_path


def event_actions(result: RunResult) -> dict[tuple[str, str], str]:
    return {
        (event["sequence_id"], event["stage"]): event["action"]
        for event in result.cache_execution["events"]
    }


def assert_canonical_equal(incremental: RunResult, full: RunResult) -> None:
    assert incremental.predictions == full.predictions
    assert incremental.decision_record == full.decision_record
    assert incremental.cache_ledger == full.cache_ledger
    assert incremental.failures == full.failures
    assert (
        incremental.diagnostic_report["per_sequence_prediction_error"]
        == full.diagnostic_report["per_sequence_prediction_error"]
    )

