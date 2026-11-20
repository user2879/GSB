"""End-to-end local verification: formats, kernels, incrementality and injections."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantcal.errors import QuantCalError  # noqa: E402
from quantcal.injections import (  # noqa: E402
    apply_input_injection,
    load_injection_manifest,
    tamper_snapshot_version,
)
from quantcal.pipeline import PipelineResult, run_pipeline  # noqa: E402
from quantcal.util import atomic_write_json, load_json, load_jsonl  # noqa: E402


DEMO_INPUT = ROOT / "sample_data"
DEMO_WORK = ROOT / "demo_work"
ARTIFACTS = ROOT / "demo_artifacts"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _edit_json(path: Path, edit: Any) -> None:
    document = load_json(path)
    edit(document)
    atomic_write_json(path, document)


def _edit_samples(path: Path, edit: Any) -> None:
    rows = load_jsonl(path)
    edit(rows)
    lines = [json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compare_runs(incremental: PipelineResult, full: PipelineResult) -> dict[str, Any]:
    inc_result = incremental.output_dir / "result.json"
    full_result = full.output_dir / "result.json"
    inc_sidecar = incremental.output_dir / "exchange_sidecar.json"
    full_sidecar = full.output_dir / "exchange_sidecar.json"
    exchange_name = "calibration_v1.txt" if incremental.result["exchange_version"] == "v1" else "calibration_v2.json"
    inc_exchange = incremental.output_dir / exchange_name
    full_exchange = full.output_dir / exchange_name
    result_match = inc_result.read_bytes() == full_result.read_bytes()
    sidecar_match = inc_sidecar.read_bytes() == full_sidecar.read_bytes()
    exchange_match = inc_exchange.read_bytes() == full_exchange.read_bytes()
    if not (result_match and sidecar_match and exchange_match):
        raise RuntimeError(
            f"incremental/full mismatch: result={result_match}, sidecar={sidecar_match}, exchange={exchange_match}"
        )
    return {
        "result_sha256": _sha(inc_result),
        "exchange_sha256": _sha(inc_exchange),
        "sidecar_sha256": _sha(inc_sidecar),
        "result_field_exact": result_match,
        "exchange_exact": exchange_match,
        "sidecar_exact": sidecar_match,
    }


def _fresh_full(input_dir: Path, tag: str, version: str, **kwargs: Any) -> PipelineResult:
    work = DEMO_WORK / "full-runs" / tag
    _reset_dir(work)
    return run_pipeline(
        input_dir=input_dir,
        work_dir=work,
        exchange_version=version,
        force_rebuild=True,
        **kwargs,
    )


def _step(
    steps: list[dict[str, Any]],
    input_dir: Path,
    incremental_work: Path,
    name: str,
    version: str,
    edit: Any = None,
    edit_target: str | None = None,
) -> PipelineResult:
    if edit is not None:
        if edit_target == "json":
            _edit_json(input_dir / "config.json", edit)
        elif edit_target == "jsonl":
            _edit_samples(input_dir / "samples.jsonl", edit)
        else:
            raise ValueError(edit_target)
    incremental = run_pipeline(
        input_dir=input_dir,
        work_dir=incremental_work,
        exchange_version=version,
        report_path=DEMO_WORK / "reports" / f"{name}.json",
    )
    full = _fresh_full(input_dir, name, version)
    comparison = _compare_runs(incremental, full)
    ledger = load_json(incremental.output_dir / "ledger.json")
    steps.append(
        {
            "step": name,
            "exchange_version": version,
            "exit_code": incremental.exit_code,
            "result_sha256": comparison["result_sha256"],
            "exchange_sha256": comparison["exchange_sha256"],
            "sidecar_sha256": comparison["sidecar_sha256"],
            "result_field_exact": comparison["result_field_exact"],
            "exchange_exact": comparison["exchange_exact"],
            "sidecar_exact": comparison["sidecar_exact"],
            "reused_tensors": ledger["reused_tensors"],
            "recomputed_tensors": ledger["recomputed_tensors"],
            "quantile_reused": ledger["stages"]["quantile"]["reused"],
            "quantile_computed": ledger["stages"]["quantile"]["computed"],
            "budget_deduction_sequence": ledger["budget"]["deduction_sequence"],
            "summary": incremental.result["summary"],
        }
    )
    return incremental


def run_incremental_sequence() -> list[dict[str, Any]]:
    input_dir = DEMO_WORK / "incremental-input"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    shutil.copytree(DEMO_INPUT, input_dir)
    incremental_work = DEMO_WORK / "incremental-work"
    if incremental_work.exists():
        shutil.rmtree(incremental_work)
    steps: list[dict[str, Any]] = []

    _step(steps, input_dir, incremental_work, "01-initial-v2", "v2")
    _step(steps, input_dir, incremental_work, "02-switch-v1", "v1")

    def change_one_sample(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row.get("sample_id") == "cal-0000":
                row["features"]["attn.q.weight"][0] = round(
                    row["features"]["attn.q.weight"][0] + 0.375, 10
                )
                return
        raise RuntimeError("sample cal-0000 not found")

    _step(
        steps,
        input_dir,
        incremental_work,
        "03-replace-sample-cal-0000",
        "v1",
        change_one_sample,
        "jsonl",
    )

    def delete_sample(rows: list[dict[str, Any]]) -> None:
        rows[:] = [row for row in rows if row.get("sample_id") != "cal-0001"]

    _step(
        steps,
        input_dir,
        incremental_work,
        "04-delete-sample-cal-0001",
        "v1",
        delete_sample,
        "jsonl",
    )

    def set_probe_range_quantiles(config: dict[str, Any]) -> None:
        config["tensor_overrides"].setdefault("probe.range", {}).update(
            {"quantile_lower_permille": 50, "quantile_upper_permille": 950}
        )

    _step(
        steps,
        input_dir,
        incremental_work,
        "05-probe-range-quantiles",
        "v1",
        set_probe_range_quantiles,
        "json",
    )

    def set_bits(config: dict[str, Any]) -> None:
        config["quantization_bits"] = 12

    _step(steps, input_dir, incremental_work, "06-bits-12", "v1", set_bits, "json")

    def set_floor(config: dict[str, Any]) -> None:
        config["rounding_mode"] = "floor"

    _step(steps, input_dir, incremental_work, "07-rounding-floor", "v1", set_floor, "json")

    def set_symmetric_saturation(config: dict[str, Any]) -> None:
        config["saturation_strategy"] = "saturate_symmetric"

    _step(
        steps,
        input_dir,
        incremental_work,
        "08-saturation-symmetric",
        "v1",
        set_symmetric_saturation,
        "json",
    )
    _step(steps, input_dir, incremental_work, "09-switch-v2", "v2")
    return steps


def run_budget_and_concurrency_checks() -> dict[str, Any]:
    work_one = DEMO_WORK / "budget-concurrency" / "serial"
    work_four = DEMO_WORK / "budget-concurrency" / "parallel"
    _reset_dir(work_one)
    _reset_dir(work_four)
    serial = run_pipeline(
        DEMO_INPUT,
        work_one,
        "v2",
        force_rebuild=True,
        max_concurrency=1,
        batch_point_budget=100,
    )
    parallel = run_pipeline(
        DEMO_INPUT,
        work_four,
        "v2",
        force_rebuild=True,
        max_concurrency=4,
        batch_point_budget=100,
    )
    if serial.result != parallel.result:
        raise RuntimeError("budget/parallel result differs from serial result")
    serial_seq = serial.diagnostics["schedule"]["budget_deduction_sequence"]
    parallel_seq = parallel.diagnostics["schedule"]["budget_deduction_sequence"]
    if serial_seq != parallel_seq:
        raise RuntimeError("budget deduction sequence differs by concurrency")
    if [x["tensor"] for x in serial.pending] != [x["tensor"] for x in parallel.pending]:
        raise RuntimeError("pending queue differs by concurrency")
    if any(item["code"] != "E_CALIB_BUDGET_PENDING" for item in serial.pending):
        raise RuntimeError("pending queue is missing E_CALIB_BUDGET_PENDING")

    work_full_serial = DEMO_WORK / "concurrency-full" / "serial"
    work_full_parallel = DEMO_WORK / "concurrency-full" / "parallel"
    _reset_dir(work_full_serial)
    _reset_dir(work_full_parallel)
    full_serial = run_pipeline(
        DEMO_INPUT, work_full_serial, "v2", force_rebuild=True, max_concurrency=1
    )
    full_parallel = run_pipeline(
        DEMO_INPUT, work_full_parallel, "v2", force_rebuild=True, max_concurrency=4
    )
    if (full_serial.output_dir / "result.json").read_bytes() != (
        full_parallel.output_dir / "result.json"
    ).read_bytes():
        raise RuntimeError("full-run result depends on max_concurrency")
    return {
        "budget_limit": 100,
        "serial_exit_code": serial.exit_code,
        "parallel_exit_code": parallel.exit_code,
        "serial_order": serial.result["tensor_order"],
        "parallel_order": parallel.result["tensor_order"],
        "budget_deduction_sequence": serial_seq,
        "pending": serial.pending,
        "result_exact": serial.result == parallel.result,
        "full_concurrency_result_exact": True,
    }


def run_injection_checks() -> list[dict[str, Any]]:
    manifest = load_injection_manifest(DEMO_INPUT / "injections.json")
    results: list[dict[str, Any]] = []
    for injection in manifest["injections"][:3]:
        injection_id = injection["id"]
        scenario_input = DEMO_WORK / "injection-inputs" / injection_id
        scenario_work = DEMO_WORK / "injection-runs" / injection_id
        hit = apply_input_injection(DEMO_INPUT, scenario_input, injection)
        hit_path = scenario_input / "injection_hit.json"
        atomic_write_json(hit_path, {"hits": [hit]})
        result = run_pipeline(
            scenario_input,
            scenario_work,
            "v2",
            force_rebuild=True,
            injection_hits=[hit],
            report_path=scenario_work / "diagnostics_report.json",
        )
        codes = [item["code"] for item in result.errors]
        expected = hit["expected_code"]
        if expected not in codes:
            raise RuntimeError(f"{injection_id} did not produce {expected}: {codes}")
        hit_tensor = hit.get("json_pointer")
        failed_tensors = {item.get("tensor") for item in result.errors}
        if injection.get("tensor") not in failed_tensors:
            raise RuntimeError(f"{injection_id} did not isolate target tensor")
        if injection.get("tensor") in result.result["tensor_order"]:
            raise RuntimeError(f"{injection_id} failed tensor entered final summary")
        diagnostics = load_json(scenario_work / "diagnostics_report.json")
        if injection_id not in {
            item.get("injection_id") for item in diagnostics.get("injection_hits", [])
        }:
            raise RuntimeError(f"{injection_id} missing from diagnostics injection_hits")
        results.append(
            {
                "injection_id": injection_id,
                "kind": injection["kind"],
                "dry_path": hit.get("path"),
                "json_pointer": hit_tensor,
                "expected_code": expected,
                "actual_codes": codes,
                "failed_tensor": injection.get("tensor"),
                "failed_tensor_excluded_from_summary": True,
                "exit_code": result.exit_code,
                "report": str((scenario_work / "diagnostics_report.json").resolve()),
                "hit_recorded": True,
            }
        )

    snapshot_injection = manifest["injections"][3]
    snapshot_work = DEMO_WORK / "injection-runs" / "INJ-004"
    _reset_dir(snapshot_work)
    run_pipeline(
        DEMO_INPUT,
        snapshot_work,
        "v2",
        force_rebuild=True,
        report_path=snapshot_work / "baseline_report.json",
    )
    tampered = tamper_snapshot_version(snapshot_work)
    hit = {
        "injection_id": snapshot_injection["id"],
        "kind": snapshot_injection["kind"],
        "path": tampered["path"],
        "json_pointer": tampered["json_pointer"],
        "expected_code": snapshot_injection["expected_code"],
        "applied": True,
        "original": tampered["original"],
        "tampered": tampered["tampered"],
    }
    caught: QuantCalError | None = None
    try:
        run_pipeline(
            DEMO_INPUT,
            snapshot_work,
            "v2",
            resume=True,
            injection_hits=[hit],
            report_path=snapshot_work / "stale_report.json",
        )
    except QuantCalError as exc:
        caught = exc
    if caught is None or caught.code != "E_SNAPSHOT_STALE":
        raise RuntimeError("snapshot tamper did not produce E_SNAPSHOT_STALE")
    fatal_report = {
        "format": "quantcal.fatal-report.v1",
        "injection_hits": [hit],
        "exit_code": 16,
        "error": caught.to_dict(),
    }
    atomic_write_json(snapshot_work / "stale_report.json", fatal_report)
    results.append(
        {
            "injection_id": snapshot_injection["id"],
            "kind": snapshot_injection["kind"],
            "path": tampered["path"],
            "json_pointer": tampered["json_pointer"],
            "expected_code": "E_SNAPSHOT_STALE",
            "actual_codes": [caught.code],
            "exit_code": 16,
            "hit_recorded": True,
            "report": str((snapshot_work / "stale_report.json").resolve()),
        }
    )
    return results


def main() -> int:
    if not (DEMO_INPUT / "tensors.json").exists():
        from scripts.generate_samples import generate

        generate(DEMO_INPUT)
    _reset_dir(DEMO_WORK)
    _reset_dir(ARTIFACTS)

    baseline = run_pipeline(
        DEMO_INPUT,
        DEMO_WORK / "baseline-v2",
        "v2",
        force_rebuild=True,
        report_path=DEMO_WORK / "reports" / "baseline-v2.json",
    )
    if baseline.exit_code != 0:
        raise RuntimeError(f"baseline failed with exit code {baseline.exit_code}")
    repeat = run_pipeline(
        DEMO_INPUT,
        DEMO_WORK / "baseline-v2-repeat",
        "v2",
        force_rebuild=True,
        report_path=DEMO_WORK / "reports" / "baseline-v2-repeat.json",
    )
    for relative in ("result.json", "calibration_v2.json", "exchange_sidecar.json", "diagnostics.json"):
        if (baseline.output_dir / relative).read_bytes() != (repeat.output_dir / relative).read_bytes():
            raise RuntimeError(f"repeated clean run differs in {relative}")
    interop = baseline.diagnostics["input_exchange_validation"]
    if not interop["v1_to_v2_to_v1_exact"]:
        raise RuntimeError("v1 -> v2 -> v1 round trip is not exact")

    incremental_steps = run_incremental_sequence()
    budget = run_budget_and_concurrency_checks()
    injections = run_injection_checks()

    report = {
        "format": "quantcal.demo-report.v1",
        "seed": 20261121,
        "status": "pass",
        "baseline": {
            "exit_code": baseline.exit_code,
            "summary": baseline.result["summary"],
            "result_sha256": _sha(baseline.output_dir / "result.json"),
            "diagnostics_sha256": _sha(baseline.output_dir / "diagnostics.json"),
        },
        "format_interoperability": {
            "v1_to_v2_to_v1_exact": interop["v1_to_v2_to_v1_exact"],
            "v1_tensor_order": interop["input_v1"]["tensor_order"],
            "v2_tensor_order": interop["input_v2"]["tensor_order"],
            "v2_downgrade_loss_codes": interop["input_v2"]["loss_codes"],
            "v2_downgrade_loss_tensor_order": [
                row["name"] for row in interop["input_v2"]["downgrade_loss_by_tensor"]
            ],
            "v2_downgrade_loss_order_stable": [
                row["name"] for row in interop["input_v2"]["downgrade_loss_by_tensor"]
            ]
            == sorted(row["name"] for row in interop["input_v2"]["downgrade_loss_by_tensor"]),
        },
        "incremental_steps": incremental_steps,
        "budget_and_concurrency": budget,
        "injections": injections,
        "verification": {
            "incremental_exact_all_steps": all(
                row["result_field_exact"] and row["exchange_exact"] and row["sidecar_exact"]
                for row in incremental_steps
            ),
            "injection_ids_recorded": sorted(row["injection_id"] for row in injections)
            == ["INJ-001", "INJ-002", "INJ-003", "INJ-004"],
            "deterministic_repeat_ready": True,
            "repeat_clean_run_exact": True,
        },
    }
    atomic_write_json(ARTIFACTS / "demo_report.json", report)
    print(json.dumps({"status": "pass", "report": str(ARTIFACTS / "demo_report.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
