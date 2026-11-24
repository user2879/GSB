"""Command-line interface for the calibration pipeline and receipt tools."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

from .errors import FailureList
from .pipeline import CalibrationPipeline, PipelineRequest
from .receipts import convert_receipt_file
from .samplegen import generate_samples
from .utils import canonical_json_text, write_json


def _run_command(args: argparse.Namespace) -> int:
    receipts: list[Path] = [Path(item) for item in (args.receipts or [])]
    if args.receipts_dir is not None:
        receipts.extend(sorted(Path(args.receipts_dir).glob("*.jsonl")))
    report_path = Path(args.report) if args.report else Path(args.state) / "latest-report.json"
    request = PipelineRequest(
        weights_path=Path(args.weights),
        calibration_path=Path(args.calibration),
        config_path=Path(args.config),
        state_dir=Path(args.state),
        receipt_paths=tuple(receipts),
        commands_path=Path(args.commands) if args.commands else None,
        faults_path=Path(args.faults) if args.faults else None,
        report_path=report_path,
        full_rebuild=args.full_rebuild,
    )
    report = CalibrationPipeline(request).run()
    if args.print_summary:
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "report": str(report_path),
                    "result_digest": report.get("result_digest"),
                    "business_code_counts": report["business_code_counts"],
                    "completed_layers": report["completed_layers"],
                    "partial_layers": report["partial_layers"],
                    "budget": report["budget"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0 if report["status"] in {"ok", "partial"} else 2


def _convert_command(args: argparse.Namespace) -> int:
    failures = FailureList()
    report = convert_receipt_file(
        input_path=Path(args.input),
        target_version=args.to,
        output_path=Path(args.output),
        sidecar_path=Path(args.sidecar) if args.sidecar else None,
        failures=failures,
    )
    report["failure_list"] = failures.to_list()
    report["business_code_counts"] = failures.counts()
    if args.report:
        write_json(Path(args.report), report)
    else:
        print(canonical_json_text(report), end="")
    return 0 if not failures.items else 2


def _core_projection(report: dict[str, Any]) -> dict[str, Any]:
    layers: list[dict[str, Any]] = []
    for layer in report.get("layers", []):
        groups = []
        for group in layer.get("groups", []):
            groups.append(
                {
                    key: value
                    for key, value in group.items()
                    if not key.endswith("_reused")
                }
            )
        layers.append(
            {
                "layer": layer.get("layer"),
                "content_version": layer.get("content_version"),
                "content_fingerprint": layer.get("content_fingerprint"),
                "status": layer.get("status"),
                "completed": layer.get("completed"),
                "calibration": layer.get("calibration"),
                "quantization": layer.get("quantization"),
                "error": layer.get("error"),
                "aggregate_outputs": layer.get("aggregate_outputs"),
                "groups": groups,
            }
        )
    return {
        "layers": layers,
        "receipt_ledger": report.get("receipt_ledger"),
        "budget": report.get("budget"),
        "role_fingerprints": report.get("role_fingerprints"),
        "completed_layers": report.get("completed_layers"),
        "partial_layers": report.get("partial_layers"),
        "business_code_counts": report.get("business_code_counts"),
    }


def _demo_command(args: argparse.Namespace) -> int:
    root = Path(args.workdir)
    samples = generate_samples(root / "samples")
    state = root / "state"
    full_state = root / "state_full"
    reports = root / "reports"
    for generated_dir in (state, full_state, reports, root / "receipts_roundtrip"):
        shutil.rmtree(generated_dir, ignore_errors=True)
    reports.mkdir(parents=True, exist_ok=True)
    receipt_paths = [samples["q1"], samples["q2"], samples["q3"]]

    def run(
        state_dir: Path,
        report_name: str,
        *,
        commands: Path | None = None,
        faults: Path | None = None,
        full_rebuild: bool = False,
    ) -> dict[str, Any]:
        request = PipelineRequest(
            weights_path=samples["weights"],
            calibration_path=samples["calibration"],
            config_path=samples["config"],
            state_dir=state_dir,
            receipt_paths=tuple(receipt_paths),
            commands_path=commands,
            faults_path=faults,
            report_path=reports / report_name,
            full_rebuild=full_rebuild,
        )
        return CalibrationPipeline(request).run()

    baseline = run(state, "baseline.json")
    incremental = run(state, "incremental.json", commands=samples["commands"])
    full = run(full_state, "full_rebuild.json", commands=samples["commands"], full_rebuild=True)
    faults = run(state, "fault_injection.json", faults=samples["faults"])

    roundtrip_dir = root / "receipts_roundtrip"
    roundtrip_dir.mkdir(parents=True, exist_ok=True)
    conversion_failures = FailureList()
    q3_to_q1 = convert_receipt_file(
        input_path=samples["q3"],
        target_version="q1",
        output_path=roundtrip_dir / "q3.as.q1.jsonl",
        sidecar_path=roundtrip_dir / "q3.sidecar.json",
        failures=conversion_failures,
    )
    q1_to_q3 = convert_receipt_file(
        input_path=roundtrip_dir / "q3.as.q1.jsonl",
        target_version="q3",
        output_path=roundtrip_dir / "q1.as.q3.jsonl",
        sidecar_path=roundtrip_dir / "q3.sidecar.json",
        failures=conversion_failures,
    )
    q3_to_q2 = convert_receipt_file(
        input_path=samples["q3"],
        target_version="q2",
        output_path=roundtrip_dir / "q3.as.q2.jsonl",
        sidecar_path=roundtrip_dir / "q3-via-q2.sidecar.json",
        failures=conversion_failures,
    )
    q2_to_q3 = convert_receipt_file(
        input_path=roundtrip_dir / "q3.as.q2.jsonl",
        target_version="q3",
        output_path=roundtrip_dir / "q2.as.q3.jsonl",
        sidecar_path=roundtrip_dir / "q3-via-q2.sidecar.json",
        failures=conversion_failures,
    )

    def read_jsonl(path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    q3_original = read_jsonl(samples["q3"])
    q3_restored = read_jsonl(roundtrip_dir / "q1.as.q3.jsonl")
    q3_restored_via_q2 = read_jsonl(roundtrip_dir / "q2.as.q3.jsonl")
    roundtrip_equal = q3_original == q3_restored == q3_restored_via_q2
    roundtrip_summary = {
        "schema": "quantcalib-receipt-roundtrip-summary-v1",
        "q3_to_q1": q3_to_q1,
        "q1_to_q3": q1_to_q3,
        "q3_to_q2": q3_to_q2,
        "q2_to_q3": q2_to_q3,
        "q3_fields_restored": bool(
            q3_to_q1["q3_bypass_fields_preserved"]
            and q3_to_q2["q3_bypass_fields_preserved"]
            and q1_to_q3["restored_from_sidecar"]
            and q2_to_q3["restored_from_sidecar"]
        ),
        "roundtrip_equal": roundtrip_equal,
        "conversion_failure_count": len(conversion_failures.items),
        "outputs": [
            str(roundtrip_dir / "q3.as.q1.jsonl"),
            str(roundtrip_dir / "q1.as.q3.jsonl"),
            str(roundtrip_dir / "q3.as.q2.jsonl"),
            str(roundtrip_dir / "q2.as.q3.jsonl"),
            str(roundtrip_dir / "q3.sidecar.json"),
            str(roundtrip_dir / "q3-via-q2.sidecar.json"),
        ],
    }
    write_json(roundtrip_dir / "roundtrip_summary.json", roundtrip_summary)

    projections = {
        "incremental": _core_projection(incremental),
        "full_rebuild": _core_projection(full),
    }
    incremental_full_equal = projections["incremental"] == projections["full_rebuild"]
    second_incremental = run(state, "incremental_second.json", commands=samples["commands"])
    second_projection = _core_projection(second_incremental)
    second_equal = second_projection == projections["incremental"]
    summary = {
        "schema": "quantcalib-demo-summary-v1",
        "seed": 20261125,
        "baseline": {
            "status": baseline["status"],
            "result_digest": baseline.get("result_digest"),
            "completed_layers": baseline["completed_layers"],
            "business_code_counts": baseline["business_code_counts"],
        },
        "incremental": {
            "status": incremental["status"],
            "result_digest": incremental.get("result_digest"),
            "completed_layers": incremental["completed_layers"],
            "changed_layer_invalidation": incremental.get("invalidation", {}).get("layers", {}).get(
                "retrieval.head"
            ),
            "reused_layers": {
                name: data.get("reused_stages", [])
                for name, data in incremental.get("invalidation", {}).get("layers", {}).items()
                if data.get("reused_stages")
            },
        },
        "full_rebuild": {
            "status": full["status"],
            "result_digest": full.get("result_digest"),
        },
        "fault_injection": {
            "status": faults["status"],
            "injections": faults["fault_injections"],
            "business_code_counts": faults["business_code_counts"],
        },
        "receipt_roundtrip": {
            "q3_fields_restored": roundtrip_summary["q3_fields_restored"],
            "roundtrip_equal": roundtrip_equal,
            "conversion_failure_count": roundtrip_summary["conversion_failure_count"],
        },
        "validation": {
            "incremental_equals_full_rebuild": incremental_full_equal,
            "second_incremental_equals_first": second_equal,
            "all_faults_hit": bool(faults["fault_injections"])
            and all(item["hit"] for item in faults["fault_injections"]),
            "receipt_roundtrip_equal": roundtrip_equal,
            "receipt_q3_fields_restored": roundtrip_summary["q3_fields_restored"],
            "reports": [
                str(reports / "baseline.json"),
                str(reports / "incremental.json"),
                str(reports / "full_rebuild.json"),
                str(reports / "incremental_second.json"),
                str(reports / "fault_injection.json"),
                str(roundtrip_dir / "roundtrip_summary.json"),
            ],
        },
    }
    summary_path = root / "demo_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    checks = summary["validation"]
    return 0 if all(
        checks[key]
        for key in (
            "incremental_equals_full_rebuild",
            "second_incremental_equals_first",
            "all_faults_hit",
            "receipt_roundtrip_equal",
            "receipt_q3_fields_restored",
        )
    ) else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m quantcalib",
        description="Deterministic per-layer weight calibration, quantization and receipt ledger tool",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run staged calibration and write a JSON diagnostic report")
    run.add_argument("--weights", required=True)
    run.add_argument("--calibration", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--receipts", nargs="*", default=[])
    run.add_argument("--receipts-dir")
    run.add_argument("--commands")
    run.add_argument("--faults")
    run.add_argument("--state", default=".quantcalib-state")
    run.add_argument("--report")
    run.add_argument("--full-rebuild", action="store_true")
    run.add_argument("--print-summary", action="store_true")
    run.set_defaults(handler=_run_command)

    convert = subparsers.add_parser("convert-receipts", help="convert q1/q2/q3 receipt JSONL losslessly")
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--to", choices=("q1", "q2", "q3"), required=True)
    convert.add_argument("--sidecar")
    convert.add_argument("--report")
    convert.set_defaults(handler=_convert_command)

    demo = subparsers.add_parser("demo", help="generate deterministic samples and run acceptance checks")
    demo.add_argument("--workdir", default="artifacts/demo")
    demo.set_defaults(handler=_demo_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))
