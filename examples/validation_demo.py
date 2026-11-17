"""One-command deterministic acceptance demonstration."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adw.commands import apply_command
from adw.config import load_config_bundle
from adw.data import generate_dataset
from adw.engine import WorkbenchEngine
from adw.injection import load_injection_bundle
from adw.pipeline import DetectionPipeline, canonical_result_payload, run_full
from adw.storage import ContentCache


def timed(operation):
    started = time.perf_counter()
    result = operation()
    return result, round((time.perf_counter() - started) * 1000.0, 3)


def main() -> int:
    bundle = load_config_bundle(ROOT / "config.yaml")
    if bundle.failures:
        print(json.dumps({"status": "failed", "failures": bundle.failures}, indent=2))
        return 2
    dataset = generate_dataset(bundle.description)
    injection = load_injection_bundle(ROOT / "injection.yaml")
    full, full_ms = timed(lambda: run_full(dataset, bundle.config))
    with tempfile.TemporaryDirectory(prefix="adw-validation-") as temp:
        temp_path = Path(temp)
        cache = ContentCache(temp_path / "cache", enabled=True)
        baseline, baseline_ms = timed(lambda: DetectionPipeline(dataset, bundle.config, cache=cache).run())
        repeated, repeated_ms = timed(lambda: DetectionPipeline(dataset, bundle.config, cache=cache).run())

        hist_changed = copy.deepcopy(bundle.config)
        hist_changed["algorithms"]["histogram_density"]["params"]["bins"] = 20
        hist_outcome, hist_ms = timed(lambda: DetectionPipeline(dataset, hist_changed, cache=cache).run())
        hist_full = run_full(dataset, hist_changed)

        threshold_changed = copy.deepcopy(hist_changed)
        threshold_changed["threshold"] = 0.61
        threshold_outcome, threshold_ms = timed(
            lambda: DetectionPipeline(dataset, threshold_changed, cache=cache).run()
        )
        threshold_full = run_full(dataset, threshold_changed)

        config = copy.deepcopy(bundle.config)
        sequence = [
            "algo disable random_partition_depth",
            "param set histogram_density bins 20",
            "preprocess set mode standardize",
            "threshold set 0.61",
            "algo enable random_partition_depth",
            "param set random_partition_depth trees 40",
        ]
        sequence_ms = 0.0
        for line in sequence:
            config = apply_command(config, bundle.description, line).config
            outcome, elapsed = timed(lambda c=config: DetectionPipeline(dataset, c, cache=cache).run())
            sequence_ms += elapsed
        sequence_full = run_full(dataset, config)

        config_one = copy.deepcopy(bundle.config)
        config_one["interaction"]["concurrency"] = 1
        config_four = copy.deepcopy(bundle.config)
        config_four["interaction"]["concurrency"] = 4
        parallel_one = DetectionPipeline(dataset, config_one, cache=ContentCache(None, enabled=False)).run()
        parallel_four = DetectionPipeline(dataset, config_four, cache=ContentCache(None, enabled=False)).run()

        budget_config = copy.deepcopy(bundle.config)
        budget_config["storage"] = {
            "snapshot_dir": str(temp_path / "budget-snapshots"),
            "cache_dir": str(temp_path / "budget-cache"),
            "report_path": str(temp_path / "report.json"),
            "injection_report_path": str(temp_path / "injection-report.json"),
        }
        engine = WorkbenchEngine(dataset, budget_config, root=ROOT)
        engine.start()
        pending_config = copy.deepcopy(budget_config)
        pending_config["threshold"] = 0.62
        budget_response = engine.submit_config(
            pending_config,
            affected_algorithms=[],
            source_command="threshold set 0.62",
        )
        engine.wait_current()
        engine.request_shutdown()
        engine.join_worker()

        injected = run_full(dataset, bundle.config, injection_items=injection.data["items"])

    result = {
        "status": "ok",
        "environment": {"python": __import__("sys").version.split()[0], "numpy": np.__version__},
        "sample_count": len(dataset.sample_ids),
        "ground_truth_anomaly_count": sum(dataset.target),
        "full_run": {
            "elapsed_ms": full_ms,
            "anomaly_count": full.payload["summary"]["anomaly_count"],
            "algorithm_distributions": full.diagnostics["distributions"]["raw"],
        },
        "incremental": {
            "baseline_elapsed_ms": baseline_ms,
            "same_config_elapsed_ms": repeated_ms,
            "same_config_cache": repeated.diagnostics["cache"],
            "same_config_recalculated": repeated.diagnostics["recalculated_algorithms"],
            "histogram_change": {
                "elapsed_ms": hist_ms,
                "recalculated": hist_outcome.diagnostics["recalculated_algorithms"],
                "reused": hist_outcome.diagnostics["reused_algorithms"],
                "equal_to_full": canonical_result_payload(hist_outcome.payload)
                == canonical_result_payload(hist_full.payload),
            },
            "threshold_change": {
                "elapsed_ms": threshold_ms,
                "recalculated": threshold_outcome.diagnostics["recalculated_algorithms"],
                "reused": threshold_outcome.diagnostics["reused_algorithms"],
                "equal_to_full": canonical_result_payload(threshold_outcome.payload)
                == canonical_result_payload(threshold_full.payload),
            },
            "interactive_sequence": {
                "commands": sequence,
                "elapsed_ms_sum": round(sequence_ms, 3),
                "equal_to_full": canonical_result_payload(outcome.payload)
                == canonical_result_payload(sequence_full.payload),
            },
        },
        "concurrency": {
            "one_equals_four": canonical_result_payload(parallel_one.payload)
            == canonical_result_payload(parallel_four.payload),
            "both_equal_serial": canonical_result_payload(parallel_four.payload)
            == canonical_result_payload(full.payload),
        },
        "budget_response": {
            "status": budget_response["status"],
            "code": budget_response["code"],
            "elapsed_ms": budget_response["interaction"]["elapsed_ms"],
            "budget_ms": budget_response["interaction"]["budget_ms"],
            "over_budget": budget_response["interaction"]["over_budget"],
        },
        "injection": {
            "hits": injected.diagnostics["injection"]["hits"],
            "total": injected.diagnostics["injection"]["total"],
            "business_codes": sorted({item["code"] for item in injected.payload["failures"]}),
            "failed_samples": sorted(
                {item["sample_id"] for item in injected.payload["failures"] if item["sample_id"]}
            ),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
