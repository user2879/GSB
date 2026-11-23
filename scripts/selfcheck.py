from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rope_diag.config import load_config, validate_config_dict
from rope_diag.errors import ErrorCode
from rope_diag.injections import load_injections
from rope_diag.pruning import load_pruning_policies
from rope_diag.samples import load_samples, parse_sample_row


def main() -> int:
    parser = argparse.ArgumentParser(description="Import, sample-load and validation-code self-check.")
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, default=Path("sample_data/config.json"))
    parser.add_argument("--samples", type=Path, default=Path("sample_data/samples.jsonl"))
    parser.add_argument(
        "--pruning", type=Path, default=Path("sample_data/pruning_strategies.json")
    )
    parser.add_argument(
        "--injections", type=Path, default=Path("sample_data/injections.json")
    )
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    config, config_failures, config_hash = load_config(args.config)
    if config is None:
        result = {
            "ok": False,
            "numpy_version": np.__version__,
            "error_codes": {
                "config": ErrorCode.CONFIG_INVALID,
                "sample": ErrorCode.SAMPLE_INVALID,
            },
            "config_failures": [failure.to_dict() for failure in config_failures],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 2

    samples = load_samples(args.samples, dimension=config.dimension)
    policies, pruning_failures, pruning_hash = load_pruning_policies(args.pruning)
    injections, injection_failures, injection_hash = load_injections(args.injections)

    _, bad_config_failures = validate_config_dict(
        {"dimension": 3}, source_path=Path("<selfcheck-invalid-config>"), config_hash=""
    )
    _, bad_sample_failures = parse_sample_row(
        {
            "sample_id": "bad",
            "query_position": -1,
            "key_positions": [1, 1],
            "query_features": [0.0] * config.dimension,
            "key_features": [[0.0] * config.dimension, [0.0] * config.dimension],
        },
        dimension=config.dimension,
        line=1,
    )
    codes_available = (
        any(failure.code == ErrorCode.CONFIG_INVALID for failure in bad_config_failures)
        and any(failure.code == ErrorCode.SAMPLE_INVALID for failure in bad_sample_failures)
    )
    snapshot_root = workspace / ".rope_diag" / "snapshots"
    temp_root = workspace / ".rope_diag" / "tmp"
    result: dict[str, Any] = {
        "ok": not (
            config_failures
            or samples.failures
            or pruning_failures
            or injection_failures
            or not codes_available
        ),
        "python_import": True,
        "numpy_version": np.__version__,
        "config_hash": config_hash,
        "pruning_hash": pruning_hash,
        "injection_hash": injection_hash,
        "config_valid": not config_failures,
        "sample_total_lines": samples.total_lines,
        "sample_valid_count": samples.sample_count,
        "sample_failures": [failure.to_dict() for failure in samples.failures],
        "pruning_policy_count": len(policies),
        "pruning_failures": [failure.to_dict() for failure in pruning_failures],
        "injection_count": len(injections),
        "injection_failures": [failure.to_dict() for failure in injection_failures],
        "error_codes": {
            "config": ErrorCode.CONFIG_INVALID,
            "sample": ErrorCode.SAMPLE_INVALID,
            "no_extrapolation": ErrorCode.NO_EXTRAPOLATION_SAMPLE,
            "point_budget_pending": ErrorCode.POINT_BUDGET_PENDING,
            "injected_fault": ErrorCode.INJECTED_FAULT,
        },
        "error_code_self_test": codes_available,
        "workspace": workspace.as_posix(),
        "snapshot_path": snapshot_root.as_posix(),
        "temporary_path": temp_root.as_posix(),
        "workspace_relative_paths": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
