#!/usr/bin/env python3
"""Run the local acceptance scenario and compare incremental/full outputs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_pipeline.config import load_config
from motion_pipeline.runner import PipelineRunner, RunRequest
from motion_pipeline.samples import load_sequences


def run(
    *,
    config_path: Path,
    sequence_path: Path,
    tier: int,
    state_dir: Path,
    output_dir: Path,
    fresh: bool = False,
) -> Any:
    config_result = load_config(config_path)
    if not config_result.ok:
        raise RuntimeError(config_result.failures)
    sequence_result = load_sequences(sequence_path)
    if not sequence_result.sequences:
        raise RuntimeError(sequence_result.failures)
    return PipelineRunner(
        RunRequest(
            config=config_result.config,
            sequences=sequence_result.sequences,
            failures=sequence_result.failures,
            scale_tier=tier,
            state_dir=state_dir,
            output_dir=output_dir,
            fresh=fresh,
        )
    ).run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "artifacts" / "demo"),
        help="Directory for copied inputs, stage states and result documents",
    )
    args = parser.parse_args()
    output_root = Path(args.output_dir)
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    config_path = output_root / "config.yaml"
    sequences_path = output_root / "sequences.jsonl"
    shutil.copy(ROOT / "samples" / "config.yaml", config_path)
    shutil.copy(ROOT / "samples" / "sequences.jsonl", sequences_path)

    incremental_state = output_root / "incremental-state"
    initial = run(
        config_path=config_path,
        sequence_path=sequences_path,
        tier=1,
        state_dir=incremental_state,
        output_dir=output_root / "tier-1-initial",
        fresh=True,
    )
    tier_two = run(
        config_path=config_path,
        sequence_path=sequences_path,
        tier=2,
        state_dir=incremental_state,
        output_dir=output_root / "tier-2-expanded",
    )

    source = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source["motion_model"]["damping"] = 0.74
    source["prediction_steps"] = 3
    edited_config = output_root / "config-edited.yaml"
    edited_config.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    lines = sequences_path.read_text(encoding="utf-8").splitlines()
    replacement = json.loads(lines[1])
    replacement["frames"][2]["trajectories"][0]["x"] += 2.5
    lines[1] = json.dumps(replacement, ensure_ascii=True, separators=(",", ":"))
    edited_sequences = output_root / "sequences-edited.jsonl"
    edited_sequences.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tier_four = run(
        config_path=edited_config,
        sequence_path=edited_sequences,
        tier=4,
        state_dir=incremental_state,
        output_dir=output_root / "tier-4-edited",
    )

    source["quantization_levels"] = [2, 4, 8]
    final_config = output_root / "config-final.yaml"
    final_config.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    incremental = run(
        config_path=final_config,
        sequence_path=edited_sequences,
        tier=2,
        state_dir=incremental_state,
        output_dir=output_root / "incremental-final",
    )
    full = run(
        config_path=final_config,
        sequence_path=edited_sequences,
        tier=2,
        state_dir=output_root / "blank-state",
        output_dir=output_root / "full-final",
        fresh=True,
    )

    equality = {
        "predictions": incremental.predictions == full.predictions,
        "decision_record": incremental.decision_record == full.decision_record,
        "cache_ledger": incremental.cache_ledger == full.cache_ledger,
        "failures": incremental.failures == full.failures,
        "prediction_error": (
            incremental.diagnostic_report["per_sequence_prediction_error"]
            == full.diagnostic_report["per_sequence_prediction_error"]
        ),
    }
    verification = {
        "status": "ok" if all(equality.values()) else "mismatch",
        "tier_strategies": {
            "1": initial.decision_record["selected"],
            "2": tier_two.decision_record["selected"],
            "4": tier_four.decision_record["selected"],
        },
        "incremental_final": {
            "strategy": incremental.decision_record["selected"],
            "stage_cache": incremental.diagnostic_report["stage_cache"],
            "stage_cache_totals": incremental.diagnostic_report["stage_cache_totals"],
            "snapshot_version": incremental.diagnostic_report["snapshot"]["global_version"],
            "stale_codes": incremental.diagnostic_report["snapshot"]["stale_business_codes"],
        },
        "blank_full_rebuild": {
            "stage_cache": full.diagnostic_report["stage_cache"],
            "stage_cache_totals": full.diagnostic_report["stage_cache_totals"],
            "snapshot_version": full.diagnostic_report["snapshot"]["global_version"],
        },
        "equality": equality,
        "failure_business_codes": incremental.diagnostic_report["failure_business_codes"],
        "sequence_count": len(incremental.predictions["sequences"]),
        "prediction_steps": incremental.decision_record["metrics"]["prediction_steps"],
        "quantization_level": incremental.diagnostic_report["quantization_level"],
        "incremental_output_dir": str((output_root / "incremental-final").resolve()),
        "full_output_dir": str((output_root / "full-final").resolve()),
    }
    verification_path = output_root / "verification.json"
    verification_path.write_text(
        json.dumps(verification, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, ensure_ascii=True, sort_keys=True))
    if verification["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
