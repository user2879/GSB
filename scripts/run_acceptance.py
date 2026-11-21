#!/usr/bin/env python3
"""Run the public acceptance flow and write a reproducible summary JSON."""

from __future__ import annotations

import json
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy
import PIL
import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from workbench.config import AlignmentConfig
from workbench.generator import generate_dataset
from workbench.injections import InjectionSet
from workbench.pipeline import Workbench

def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(
    root: Path,
    *,
    samples: Path,
    config: AlignmentConfig | Path,
    output: Path,
    mode: str,
    injections: Path | None,
) -> dict[str, Any]:
    return Workbench(
        samples_dir=samples,
        config=config,
        output_dir=output,
        mode=mode,
        injections=InjectionSet.load(injections) if injections else InjectionSet.empty(),
    ).run()


def _replace_image(path: Path) -> None:
    image = Image.new("RGB", (256, 192), (245, 243, 237))
    draw = ImageDraw.Draw(image)
    draw.ellipse((28, 20, 112, 96), fill=(205, 62, 75), outline=(20, 20, 20), width=2)
    draw.rectangle((146, 86, 228, 168), fill=(50, 110, 205), outline=(20, 20, 20), width=2)
    image.save(path, format="PNG")


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report["status"],
        "reproducibility_sha256": report["reproducibility_sha256"],
        "business_codes": report["business_codes"],
        "retrieval_quality": report["retrieval_quality"],
        "failure_count": len(report["failures"]),
        "pending_task_count": len(report["pending_tasks"]),
        "snapshot": report["snapshot"],
    }


def main() -> int:
    samples = ROOT / "samples"
    config_dir = ROOT / "config"
    if not samples.exists() or not (config_dir / "alignment.json").exists():
        generate_dataset(samples, config_dir, seed=20261122, sample_count=8)

    output = ROOT / "output"
    config_path = config_dir / "alignment.json"
    config = AlignmentConfig.load(config_path)
    empty_injections = config_dir / "injections.json"
    demo_injections = config_dir / "injections.demo.json"

    baseline = _run(
        ROOT,
        samples=samples,
        config=config,
        output=output,
        mode="full",
        injections=empty_injections,
    )
    incremental = _run(
        ROOT,
        samples=samples,
        config=config,
        output=output,
        mode="incremental",
        injections=empty_injections,
    )
    baseline_incremental_equal = (
        baseline["reproducibility_sha256"] == incremental["reproducibility_sha256"]
    )
    reuse_counts = {
        stage: incremental["reuse_ledger"][stage]["reused_count"]
        for stage in ("visual", "text", "alignment", "retrieval")
    }

    with tempfile.TemporaryDirectory(prefix="cm-workbench-acceptance-") as temp_name:
        temp = Path(temp_name)
        edit_root = temp / "edit-sequence"
        shutil.copytree(samples, edit_root / "samples")
        shutil.copytree(config_dir, edit_root / "config")
        edited_samples = edit_root / "samples"
        edited_config_path = edit_root / "config" / "alignment.json"

        _run(
            edit_root,
            samples=edited_samples,
            config=AlignmentConfig.load(edited_config_path),
            output=edit_root / "incremental-output",
            mode="full",
            injections=edit_root / "config" / "injections.json",
        )
        _replace_image(edited_samples / "sample-01" / "image.png")
        _run(
            edit_root,
            samples=edited_samples,
            config=AlignmentConfig.load(edited_config_path),
            output=edit_root / "incremental-output",
            mode="incremental",
            injections=edit_root / "config" / "injections.json",
        )
        text_path = edited_samples / "sample-02" / "text.json"
        text_document = json.loads(text_path.read_text(encoding="utf-8"))
        text_document["assertions"][0]["value"] = "circle"
        text_path.write_text(json.dumps(text_document, ensure_ascii=False, indent=2), encoding="utf-8")
        _run(
            edit_root,
            samples=edited_samples,
            config=AlignmentConfig.load(edited_config_path),
            output=edit_root / "incremental-output",
            mode="incremental",
            injections=edit_root / "config" / "injections.json",
        )
        edited_raw = json.loads(edited_config_path.read_text(encoding="utf-8"))
        for update in (
            {"visual_weight": 0.5, "text_weight": 0.5},
            {"chunk_size": 16},
            {"quantization_level": 2},
        ):
            edited_raw.update(update)
            edited_config_path.write_text(
                json.dumps(edited_raw, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _run(
                edit_root,
                samples=edited_samples,
                config=AlignmentConfig.load(edited_config_path),
                output=edit_root / "incremental-output",
                mode="incremental",
                injections=edit_root / "config" / "injections.json",
            )
        edited_incremental = _run(
            edit_root,
            samples=edited_samples,
            config=AlignmentConfig.load(edited_config_path),
            output=edit_root / "incremental-output",
            mode="incremental",
            injections=edit_root / "config" / "injections.json",
        )

        clean_root = temp / "clean-full"
        shutil.copytree(edited_samples, clean_root / "samples")
        shutil.copytree(edit_root / "config", clean_root / "config")
        clean_full = _run(
            clean_root,
            samples=clean_root / "samples",
            config=AlignmentConfig.load(clean_root / "config" / "alignment.json"),
            output=clean_root / "output",
            mode="full",
            injections=clean_root / "config" / "injections.json",
        )
        edit_sequence_equal = (
            edited_incremental["reproducibility_sha256"]
            == clean_full["reproducibility_sha256"]
        )
        edit_sequence_checks = {
            "retrieval_equal": edited_incremental["retrieval_index"]["queries"]
            == clean_full["retrieval_index"]["queries"],
            "failures_equal": edited_incremental["failures"] == clean_full["failures"],
            "boundary_equal": edited_incremental["boundary_diagnostics"]
            == clean_full["boundary_diagnostics"],
            "cache_keys_equal": [
                (item["visual_cache_key"], item["text_cache_key"], item["alignment_cache_key"])
                for item in edited_incremental["samples"]
            ]
            == [
                (item["visual_cache_key"], item["text_cache_key"], item["alignment_cache_key"])
                for item in clean_full["samples"]
            ],
        }

        budget_config = AlignmentConfig.from_mapping({**config.raw, "feature_point_budget": 10})
        budget_report = _run(
            temp,
            samples=samples,
            config=budget_config,
            output=temp / "budget-output",
            mode="full",
            injections=empty_injections,
        )

        serial_config = AlignmentConfig.from_mapping({**config.raw, "thread_pool_size": 1})
        parallel_config = AlignmentConfig.from_mapping({**config.raw, "thread_pool_size": 8})
        serial_report = _run(
            temp,
            samples=samples,
            config=serial_config,
            output=temp / "serial-output",
            mode="full",
            injections=empty_injections,
        )
        parallel_report = _run(
            temp,
            samples=samples,
            config=parallel_config,
            output=temp / "parallel-output",
            mode="full",
            injections=empty_injections,
        )

    injection_output = ROOT / "output" / "injection_demo"
    injection_report = _run(
        ROOT,
        samples=samples,
        config=config,
        output=injection_output,
        mode="full",
        injections=demo_injections,
    )
    hits = injection_report["injection_results"]["hits"]
    injection_checks = {
        "configured": injection_report["injection_results"]["configured"],
        "hit_count": len(hits),
        "missed_count": injection_report["injection_results"]["missed_count"],
        "hit_codes": {item["injection_id"]: item["code"] for item in hits},
        "failed_samples": sorted({item["sample"] for item in injection_report["failures"]}),
        "threshold_fallback": injection_report["threshold_injection"],
    }

    checks = {
        "baseline_validation": baseline["status"] in {"ok", "degraded"}
        and not baseline["failures"],
        "incremental_equals_full": baseline_incremental_equal,
        "all_stages_reused_on_unchanged_incremental": all(
            count > 0 for count in reuse_counts.values()
        ),
        "edit_sequence_equals_clean_full": edit_sequence_equal
        and all(edit_sequence_checks.values()),
        "all_injections_hit": len(hits) == 5
        and injection_report["injection_results"]["missed_count"] == 0,
        "point_budget_pending": "E_POINT_BUDGET_PENDING"
        in budget_report["business_codes"]
        and budget_report["scheduler"]["consumed_points"] <= 10,
        "serial_parallel_equal": serial_report["reproducibility_sha256"]
        == parallel_report["reproducibility_sha256"],
        "boundary_diagnostic_present": "E_BOUNDARY_SENSITIVE"
        in baseline["business_codes"],
    }
    summary = {
        "status": "passed" if all(checks.values()) else "failed",
        "environment": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "pillow": PIL.__version__,
            "pytest": pytest.__version__,
        },
        "checks": checks,
        "baseline": _summary(baseline),
        "unchanged_incremental": {
            **_summary(incremental),
            "digest_equal_to_full": baseline_incremental_equal,
            "reuse_counts": reuse_counts,
        },
        "edit_sequence": {
            "digest_equal_to_clean_full": edit_sequence_equal,
            "checks": edit_sequence_checks,
            "incremental_digest": edited_incremental["reproducibility_sha256"],
            "clean_full_digest": clean_full["reproducibility_sha256"],
        },
        "injection_demo": injection_checks,
        "point_budget": {
            "business_codes": budget_report["business_codes"],
            "scheduler": budget_report["scheduler"],
            "pending_task_count": len(budget_report["pending_tasks"]),
            "retrieval_query_count": len(budget_report["retrieval_index"]["queries"]),
        },
        "serial_parallel": {
            "serial_digest": serial_report["reproducibility_sha256"],
            "parallel_digest": parallel_report["reproducibility_sha256"],
            "equal": serial_report["reproducibility_sha256"]
            == parallel_report["reproducibility_sha256"],
        },
    }
    _write_json(output / "acceptance.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
