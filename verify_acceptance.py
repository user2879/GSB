from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from domain_pipeline.canonical import atomic_write_text  # noqa: E402
from domain_pipeline.config import (  # noqa: E402
    dump_config_yaml,
    load_config_yaml,
    validate_config,
)
from domain_pipeline.edits import load_edit_plan  # noqa: E402
from domain_pipeline.errors import SnapshotStaleError  # noqa: E402
from domain_pipeline.loader import (  # noqa: E402
    SampleStore,
    parse_samples,
    state_failures,
)
from domain_pipeline.pipeline import (  # noqa: E402
    PipelineRunner,
    _diff_paths,
    canonical_comparison_values,
)
from domain_pipeline.snapshot import SnapshotStore  # noqa: E402
from domain_pipeline.synthetic import (  # noqa: E402
    DEMO_CONFIG,
    generate_demo_workspace,
    make_sample,
)


def canonical_from_snapshot(store: SnapshotStore) -> tuple[dict[str, Any], Any]:
    data = store.load_current()
    if data is None:
        raise RuntimeError("snapshot is missing")
    config = validate_config(data.raw_config)
    parsed = parse_samples(data.sample_lines, config)
    failures = state_failures(config.failures, parsed.failures)
    return (
        canonical_comparison_values(
            data.stages,
            failures,
            data.ledger,
            pipeline_complete=data.pipeline_complete,
        ),
        data,
    )


def run_budget_probe(root: Path, workers: int) -> dict[str, Any]:
    if root.exists():
        shutil.rmtree(root)
    config = copy.deepcopy(DEMO_CONFIG)
    config["concurrency"]["max_workers"] = workers
    config["concurrency"]["batch_budget"] = 3
    lines = [
        json.dumps(make_sample(index), sort_keys=True, separators=(",", ":"))
        for index in range(8)
    ]
    store = SnapshotStore(root / "snapshots")
    runner = PipelineRunner(
        snapshot_store=store,
        sample_store=SampleStore.from_lines(lines),
        raw_config=config,
        pipeline_complete=False,
    )
    sequences: list[list[dict[str, Any]]] = []
    pending_steps: list[list[str]] = []
    failure_codes: set[str] = set()
    for step in range(10):
        outcome = runner.process_current(batch_id=f"budget-step-{step}")
        sequences.append(outcome.report["budget"]["sequence"])
        pending_steps.append(outcome.report["pending"])
        failure_codes.update(
            code
            for failure in outcome.report["failures"]
            for code in failure["codes"]
        )
        runner._commit()
        if outcome.complete:
            break
        reloaded = PipelineRunner.from_snapshot(store)
        if reloaded is None:
            raise RuntimeError("pending snapshot could not be resumed")
        runner = reloaded
    canonical, data = canonical_from_snapshot(store)
    return {
        "workers": workers,
        "steps": len(sequences),
        "sequences": sequences,
        "pending_steps": pending_steps,
        "failure_codes": sorted(failure_codes),
        "complete": runner.pipeline_complete,
        "canonical": canonical,
        "stage_versions": data.manifest.get("stage_versions"),
    }


def main() -> int:
    workspace = ROOT / "demo_workspace"
    full_workspace = ROOT / "full_rebuild_workspace"
    acceptance_dir = ROOT / "acceptance"
    for directory in (workspace, full_workspace, acceptance_dir):
        if directory.exists():
            shutil.rmtree(directory)

    paths = generate_demo_workspace(workspace, reset=True)
    incremental_store = SnapshotStore(workspace / "snapshots")
    runner = PipelineRunner(
        snapshot_store=incremental_store,
        sample_store=SampleStore.from_path(paths["samples"]),
        raw_config=load_config_yaml(str(paths["config"])),
        pipeline_complete=False,
    )
    plan = load_edit_plan(paths["plan"])
    result = runner.run_plan(plan, verify_full=True)
    incremental_report = {
        "batch_ids": [report["batch_id"] for report in result.reports],
        "invalidations": [report["invalidations"] for report in result.reports],
        "reuses": [report["reuses"] for report in result.reports],
        "budgets": [report["budget"] for report in result.reports],
        "full_rebuild_checks": [
            report["full_rebuild"] for report in result.reports
        ],
        "failure_codes": sorted(
            {
                code
                for report in result.reports
                for failure in report["failures"]
                for code in failure["codes"]
            }
        ),
    }

    incremental_canonical, incremental_data = canonical_from_snapshot(
        incremental_store
    )
    full_samples = full_workspace / "samples" / "samples.jsonl"
    full_config = full_workspace / "configs" / "pipeline.yaml"
    full_samples.parent.mkdir(parents=True, exist_ok=True)
    full_config.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        full_samples,
        "\n".join(incremental_data.sample_lines)
        + ("\n" if incremental_data.sample_lines else ""),
    )
    atomic_write_text(full_config, dump_config_yaml(incremental_data.raw_config))
    full_store = SnapshotStore(full_workspace / "snapshots")
    full_runner = PipelineRunner(
        snapshot_store=full_store,
        sample_store=SampleStore.from_path(full_samples),
        raw_config=incremental_data.raw_config,
        pipeline_complete=False,
    )
    full_outcome = full_runner.process_current(batch_id="full-rebuild")
    full_runner._commit()
    full_canonical, _ = canonical_from_snapshot(full_store)
    parity_diffs = _diff_paths(incremental_canonical, full_canonical)

    versions = incremental_store.list_versions()
    latest_pointer = incremental_store.current_pointer()
    restore_probe: dict[str, Any] = {}
    if versions:
        first = versions[0]
        restored = incremental_store.restore(int(first["global_version"]))
        restore_probe["restored_version"] = restored["global_version"]
        restore_probe["pointer_after_restore"] = incremental_store.current_pointer()
        incremental_store.restore(latest_pointer or str(restored["version_id"]))
        restore_probe["pointer_restored_to_latest"] = incremental_store.current_pointer()
    stale_probe: dict[str, Any] = {}
    try:
        incremental_store.validate_resume(expected_version=999999)
        stale_probe["version_mismatch"] = "NO_ERROR"
    except SnapshotStaleError as exc:
        stale_probe["version_mismatch"] = exc.code
    try:
        incremental_store.validate_resume(expected_config_hash="invalid-config-hash")
        stale_probe["config_mismatch"] = "NO_ERROR"
    except SnapshotStaleError as exc:
        stale_probe["config_mismatch"] = exc.code

    serial_probe = run_budget_probe(acceptance_dir / "budget-serial", 1)
    parallel_probe = run_budget_probe(acceptance_dir / "budget-parallel", 4)
    budget_parity = _diff_paths(
        serial_probe["canonical"], parallel_probe["canonical"]
    )
    budget_sequences_equal = (
        serial_probe["sequences"] == parallel_probe["sequences"]
    )
    budget_pending_equal = (
        serial_probe["pending_steps"] == parallel_probe["pending_steps"]
    )

    quant_base_config = copy.deepcopy(incremental_data.raw_config)
    quant_base_config["quantization"] = {
        "fractional_bits": 12,
        "scale_level": 4,
    }
    quant_runner = PipelineRunner(
        snapshot_store=SnapshotStore(acceptance_dir / "quant-probe-snapshots"),
        sample_store=SampleStore.from_lines(incremental_data.sample_lines),
        raw_config=quant_base_config,
        pipeline_complete=False,
    )
    quant_before = quant_runner.process_current(batch_id="quant-before")
    quant_runner.raw_config["quantization"] = {
        "fractional_bits": 10,
        "scale_level": 8,
    }
    quant_after = quant_runner.process_current(batch_id="quant-after")
    before_inference = quant_before.stages["inference"]
    after_inference = quant_after.stages["inference"]
    quant_probe = {
        "features_reused": "features" in quant_after.report["reuses"],
        "training_reused": "training" in quant_after.report["reuses"],
        "inference_invalidated": "inference" in quant_after.report["invalidations"],
        "summary_invalidated": "summary" in quant_after.report["invalidations"],
        "first_quantized_integer_before": before_inference["predictions"][0][
            "quantized_integer"
        ],
        "first_quantized_integer_after": after_inference["predictions"][0][
            "quantized_integer"
        ],
        "summary_log_loss_before": quant_before.stages["summary"]["log_loss"],
        "summary_log_loss_after": quant_after.stages["summary"]["log_loss"],
    }

    report = {
        "ok": (
            all(
                check.get("equal") is True
                for check in incremental_report["full_rebuild_checks"]
            )
            and not parity_diffs
            and not budget_parity
            and budget_sequences_equal
            and budget_pending_equal
            and serial_probe["complete"]
            and parallel_probe["complete"]
        ),
        "python_version": sys.version.split()[0],
        "incremental": incremental_report,
        "incremental_snapshot": {
            "global_version": incremental_data.manifest.get("global_version"),
            "config_hash": incremental_data.manifest.get("config_hash"),
            "data_fingerprint": incremental_data.manifest.get("data_fingerprint"),
            "stage_versions": incremental_data.manifest.get("stage_versions"),
        },
        "full_rebuild": {
            "complete": full_outcome.complete,
            "global_version": full_store.load_current().manifest.get("global_version"),
            "stage_versions": full_store.load_current().manifest.get("stage_versions"),
            "parity_equal": not parity_diffs,
            "parity_diffs": parity_diffs[:50],
        },
        "snapshot_restore": restore_probe,
        "snapshot_stale_codes": stale_probe,
        "budget_serial_parallel": {
            "serial": {
                key: value
                for key, value in serial_probe.items()
                if key != "canonical"
            },
            "parallel": {
                key: value
                for key, value in parallel_probe.items()
                if key != "canonical"
            },
            "canonical_equal": not budget_parity,
            "parity_diffs": budget_parity[:50],
            "budget_sequences_equal": budget_sequences_equal,
            "pending_queues_equal": budget_pending_equal,
        },
        "quantization_probe": quant_probe,
        "versions": [
            {
                "global_version": manifest.get("global_version"),
                "version_id": manifest.get("version_id"),
                "config_hash": manifest.get("config_hash"),
                "data_fingerprint": manifest.get("data_fingerprint"),
                "stage_versions": manifest.get("stage_versions"),
            }
            for manifest in versions
        ],
    }
    acceptance_dir.mkdir(parents=True, exist_ok=True)
    out_path = acceptance_dir / "acceptance_report.json"
    atomic_write_text(out_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"ok": report["ok"], "report": str(out_path)}, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
