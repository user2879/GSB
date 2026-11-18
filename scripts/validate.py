from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from probcal.pipeline import core_digest, report_core  # noqa: E402
from probcal.service import edit_service, run_service  # noqa: E402
from tests.test_service import create_workspace, run_ok, write_edit  # noqa: E402


EDIT_SEQUENCE = [
    [
        {"op": "switch_calibration", "scheme": "raw", "enabled": False},
        {"op": "switch_calibration", "scheme": "platt", "enabled": True},
        {"op": "set_bins", "scheme": "histogram", "bins": 12},
        {"op": "set_bins", "target": "evaluation", "bins": 10},
    ],
    [
        {"op": "set_regularization", "scheme": "platt", "strength": "0.002"},
        {"op": "set_regularization", "scheme": "isotonic", "strength": "1.0"},
    ],
    [{"op": "set_threshold_step", "step": "0.025"}],
    [
        {
            "op": "replace_sample",
            "sample_id": 17,
            "values": {
                "age": 39.5,
                "income": 70123.25,
                "income_redundant": 72232.95,
                "region": "east",
                "segment": "premium",
                "risk_score": 1.75,
            },
            "target": 1,
        }
    ],
]


def stage_keys(report: dict) -> dict[str, str]:
    return {
        stage: report["snapshot"]["stage_versions"][stage]["cache_key"]
        for stage in ("data", "model", "calibration", "metrics")
    }


def main() -> int:
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-v", "tests"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    temporary_root = ROOT / "artifacts" / "tmp"
    temporary_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="acceptance-", dir=str(temporary_root)))
    summary: dict = {
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": _package_version("numpy"),
            "pyyaml": _package_version("yaml"),
        },
        "unit_tests": {
            "returncode": tests.returncode,
            "summary": (tests.stderr.strip().splitlines() or [""])[-1],
        },
    }
    try:
        paths = create_workspace(temporary, workers=4)
        incremental_state = temporary / "incremental-state"
        initial = run_ok(paths["config"], incremental_state)
        full_initial = run_ok(paths["config"], temporary / "full-0")
        summary["initial"] = {
            "core_sha256": core_digest(initial),
            "full_rebuild_equal": report_core(initial) == report_core(full_initial),
            "stage_keys": stage_keys(initial),
            "model_count": len(initial["models"]),
            "calibration_count": len(initial["calibrations"]),
            "inversion_count": len(initial["reverse_pairs"]),
            "scheme_metrics": [
                {
                    "model_name": item["model_name"],
                    "scheme": item["scheme"],
                    "ece": item["ece"],
                    "mce": item["mce"],
                }
                for item in initial["scheme_metrics"]
            ],
        }
        incremental_steps = []
        for index, operations in enumerate(EDIT_SEQUENCE, start=1):
            edit_path = write_edit(temporary / f"edit-{index}.yaml", operations)
            report = edit_service(paths["config"], incremental_state, edit_path)
            if report["status"] != "ok":
                raise RuntimeError(json.dumps(report, ensure_ascii=False, indent=2))
            full = run_ok(paths["config"], temporary / f"full-{index}")
            incremental_steps.append(
                {
                    "step": index,
                    "snapshot_version": report["snapshot"]["version"],
                    "reused_stages": report["execution"]["reused_stages"],
                    "recomputed_stages": report["execution"]["recomputed_stages"],
                    "invalidated_inputs": [
                        {"stage": item["stage"], "input": item["input_key"]}
                        for item in report["execution"]["invalidated_inputs"]
                    ],
                    "core_sha256": core_digest(report),
                    "full_rebuild_equal": report_core(report) == report_core(full),
                    "stage_keys_equal": stage_keys(report) == stage_keys(full),
                }
            )
        summary["incremental_steps"] = incremental_steps

        one_paths = create_workspace(temporary / "parallel", workers=1)
        four_paths = create_workspace(temporary / "parallel", workers=4)
        one = run_ok(one_paths["config"], temporary / "parallel-state-1")
        four = run_ok(four_paths["config"], temporary / "parallel-state-4")
        summary["concurrency"] = {
            "one_worker_core_sha256": core_digest(one),
            "four_worker_core_sha256": core_digest(four),
            "core_equal": report_core(one) == report_core(four),
            "stage_keys_equal": stage_keys(one) == stage_keys(four),
        }
        summary["all_checks_passed"] = bool(
            tests.returncode == 0
            and summary["initial"]["full_rebuild_equal"]
            and all(
                step["full_rebuild_equal"] and step["stage_keys_equal"]
                for step in incremental_steps
            )
            and summary["concurrency"]["core_equal"]
            and summary["concurrency"]["stage_keys_equal"]
        )
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    destination = ROOT / "artifacts" / "validation" / "acceptance_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("all_checks_passed") else 1


def _package_version(module_name: str) -> str:
    module = __import__(module_name)
    return str(getattr(module, "__version__", "unknown"))


if __name__ == "__main__":
    raise SystemExit(main())

