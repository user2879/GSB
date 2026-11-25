#!/usr/bin/env python3
"""Run the deterministic acceptance matrix and emit a compact JSON record."""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw

from layout_diag.editing import apply_edit_stream
from layout_diag.engine import run_from_paths
from layout_diag.generator import generate_samples
from layout_diag.utils import read_json, stable_hash, write_json
from layout_diag.validation import validate_workspace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default="verification/results.json")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="layout-verify-") as temporary:
        root = Path(temporary)
        dataset = root / "samples"
        config_path = root / "config.json"
        generate_samples(dataset, config_path=config_path)
        config = read_json(config_path)
        config["point_budget"] = 200
        write_json(config_path, config)

        validated_dataset, _, validation_failures = validate_workspace(dataset, config_path)
        if validation_failures.items:
            raise SystemExit(f"validation unexpectedly failed: {validation_failures.as_dicts()}")

        baseline, baseline_failures = run_from_paths(dataset, config_path, root / "baseline")
        repeated, _ = run_from_paths(dataset, config_path, root / "baseline")
        baseline_summary = summarize(baseline, baseline_failures)
        repeat_summary = {
            "run_mode": repeated.manifest["run_mode"],
            "cache": repeated.documents["reuse"]["summary"],
            "keypoints_equal": repeated.documents["keypoints"] == baseline.documents["keypoints"],
            "reading_equal": repeated.documents["reading_order"] == baseline.documents["reading_order"],
            "failures_equal": repeated.documents["failures"] == baseline.documents["failures"],
        }

        tiers = {}
        for tier in ("small", "medium", "large"):
            tier_config = dict(config)
            tier_config["tier"] = tier
            tier_path = root / f"config-{tier}.json"
            write_json(tier_path, tier_config)
            result, failures = run_from_paths(dataset, tier_path, root / f"tier-{tier}", force_rebuild=True)
            points = result.documents["keypoints"]["keypoints"]
            tiers[tier] = {
                "candidate_count": result.documents["report"]["stages"]["candidate_count"],
                "keypoint_count": len(points),
                "first_keypoint": {
                    "x_q": points[0]["x_q"],
                    "y_q": points[0]["y_q"],
                    "fixed_bits": points[0]["fixed_bits"],
                    "quant_step": points[0]["quant_step"],
                    "residual_x_q": points[0]["residual_x_q"],
                    "residual_y_q": points[0]["residual_y_q"],
                },
                "conflict_counts": result.documents["report"]["conflict_counts"],
                "business_code_counts": result.documents["report"]["business_code_counts"],
                "failures": [failure.as_dict() for failure in failures.items],
            }
        if len({item["candidate_count"] for item in tiers.values()}) < 2:
            raise SystemExit("tier candidate counts did not change")
        if len({(item["first_keypoint"]["x_q"], item["first_keypoint"]["y_q"]) for item in tiers.values()}) < 2:
            raise SystemExit("tier keypoint coordinates did not change")

        budget_config = dict(config)
        budget_config["tier"] = "medium"
        budget_config["point_budget"] = 20
        budget_config["max_workers"] = 1
        budget_path = root / "config-budget.json"
        write_json(budget_path, budget_config)
        budget, budget_failures = run_from_paths(dataset, budget_path, root / "budget")
        budget_resume, _ = run_from_paths(dataset, budget_path, root / "budget")
        budget_summary = {
            "point_budget": budget_config["point_budget"],
            "detected_points": len(budget.documents["keypoints"]["keypoints"]),
            "pending_points": budget.manifest["pending_points"],
            "pending_columns": budget.documents["report"]["scheduling"]["pending_count"],
            "pending_codes": [failure.code for failure in budget_failures.items],
            "resume_mode": budget_resume.manifest["run_mode"],
            "resume_equal": budget_resume.documents["keypoints"] == budget.documents["keypoints"],
        }

        edited_dataset = root / "edited"
        shutil.copytree(dataset, edited_dataset)
        edited_config = root / "config-edited.json"
        shutil.copyfile(config_path, edited_config)
        before, _ = run_from_paths(edited_dataset, edited_config, root / "edited-run")
        replacement = root / "replacement.png"
        source = Image.open(edited_dataset / "images" / "small_layout_01.png").convert("L")
        ImageDraw.Draw(source).rectangle((15, 905, 35, 925), fill=24)
        source.save(replacement)
        edit_failures = apply_edit_stream(
            edited_dataset,
            edited_config,
            [{"op": "replace_image", "image": "small_layout_01.png", "source": str(replacement)}],
        )
        after, _ = run_from_paths(edited_dataset, edited_config, root / "edited-run")
        image_statuses = {
            f"{entry['image']}:{entry['stage']}": entry["status"]
            for entry in after.documents["reuse"]["entries"]
            if entry["scope"] == "image"
        }
        invalidation_summary = {
            "edit_failures": [failure.as_dict() for failure in edit_failures.items],
            "run_mode": after.manifest["run_mode"],
            "cache": after.documents["reuse"]["summary"],
            "image_stage_statuses": dict(sorted(image_statuses.items())),
            "replaced_image_invalidated": image_statuses.get("small_layout_01.png:sliding") == "miss",
            "unaffected_images_reused": (
                image_statuses.get("large_layout_01.png:sliding") == "hit"
                and image_statuses.get("medium_layout_01.png:components") == "hit"
            ),
        }

        full_rebuild, _ = run_from_paths(edited_dataset, edited_config, root / "edited-full", force_rebuild=True)
        incremental_payload = after.canonical_results(include_reuse=False)
        rebuild_payload = full_rebuild.canonical_results(include_reuse=False)
        rebuild_equal = incremental_payload == rebuild_payload
        if not rebuild_equal:
            raise SystemExit("incremental final payload differs from full rebuild")

        threaded, _ = run_from_paths(dataset, config_path, root / "threaded", force_rebuild=True, serial=False)
        serial, _ = run_from_paths(dataset, config_path, root / "serial", force_rebuild=True, serial=True)
        concurrency_equal = (
            threaded.documents["keypoints"] == serial.documents["keypoints"]
            and threaded.documents["reading_order"] == serial.documents["reading_order"]
            and threaded.documents["failures"] == serial.documents["failures"]
        )
        if not concurrency_equal:
            raise SystemExit("threaded and serial outputs differ")

        test_suite = run_test_suite()
        if not test_suite["ok"]:
            raise SystemExit(f"test suite failed: {test_suite}")
        record = {
            "ok": True,
            "python": sys.version.split()[0],
            "environment": {
                "resolution": [640, 960],
                "pixel_mode": "L",
                "samples": [entry["image"] for entry in baseline.documents["coarse"]["images"]],
                "sample_content_hash": baseline.manifest["dataset_hash"],
                "config_hash": baseline.manifest["config_hash"],
            },
            "validation": {"failures": [], "stats": validated_dataset.stats()},
            "baseline": baseline_summary,
            "repeat": repeat_summary,
            "tiers": tiers,
            "budget": budget_summary,
            "invalidation": invalidation_summary,
            "incremental_vs_full_rebuild": {
                "equal_without_reuse_status": rebuild_equal,
                "incremental_hits": after.documents["reuse"]["summary"]["hits"],
                "full_rebuild_hits": full_rebuild.documents["reuse"]["summary"]["hits"],
            },
            "concurrency": {
                "equal": concurrency_equal,
                "configured_max_workers": threaded.documents["report"]["scheduling"]["max_workers_configured"],
                "execution_modes": ["thread-pool", "serial"],
            },
            "test_suite": test_suite,
            "verification_hash": stable_hash(
                {
                    "baseline": baseline_summary["global_version"],
                    "tiers": {name: item["candidate_count"] for name, item in tiers.items()},
                    "budget": budget_summary,
                    "invalidation": invalidation_summary["cache"],
                    "tests_run": test_suite["tests_run"],
                    "tests_ok": test_suite["ok"],
                }
            ),
        }
        write_json(args.json_out, record)
        print(json.dumps({"ok": True, "json_out": args.json_out, "verification_hash": record["verification_hash"]}, indent=2))
    return 0


def summarize(result, failures) -> dict:
    return {
        "run_mode": result.manifest["run_mode"],
        "tier": result.manifest["tier"],
        "global_version": result.manifest["global_version"],
        "completed_images": result.manifest["completed_images"],
        "keypoint_count": len(result.documents["keypoints"]["keypoints"]),
        "candidate_count": result.documents["report"]["stages"]["candidate_count"],
        "component_count": result.documents["report"]["stages"]["component_count"],
        "merge_records": result.documents["report"]["stages"]["merge_record_count"],
        "split_records": result.documents["report"]["stages"]["split_record_count"],
        "conflict_counts": result.documents["report"]["conflict_counts"],
        "business_code_counts": result.documents["report"]["business_code_counts"],
        "cache": result.documents["reuse"]["summary"],
        "failures": [failure.as_dict() for failure in failures.items],
    }


def run_test_suite() -> dict:
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    stream = io.StringIO()
    runner = unittest.TextTestRunner(verbosity=0, stream=stream)
    result = runner.run(suite)
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "ok": result.wasSuccessful(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
