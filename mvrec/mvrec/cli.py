"""Command line entry point for the reconstruction pipeline."""

import argparse
import json
import sys

from .config import load_config
from .dataset import scan_views
from .errors import PipelineError
from .pipeline import ReconstructionPipeline
from .samplegen import generate_sample
from .snapshot import atomic_write_json


def load_injections(path):
    if not path:
        return {"items": {}}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    items = {}
    for entry in payload.get("injections", []):
        items[str(entry["view_id"])] = entry["effects"]
        entry.setdefault("hit", False)
    payload["items"] = items
    return payload


def mark_injection_hits(injection_config, failures):
    hits = {}
    for entry in injection_config.get("injections", []):
        view_id = str(entry["view_id"])
        for effect in entry["effects"]:
            hits[(view_id, effect)] = entry["id"]
    hit_ids = []
    for failure in failures:
        view_id = str(failure["view_id"])
        effect_by_code = {
            "E_PNG_INVALID": "corrupt_png",
            "E_CAMERA_INVALID": "missing_camera",
            "E_PARALLAX_INVALID": "parallax_violation",
            "E_FEATURE_NONFINITE": "nonfinite_feature",
            "E_SLOT_BUDGET_EXHAUSTED": "slot_budget_zero",
            "E_POINT_BUDGET_EXHAUSTED": "point_budget_zero",
        }
        effect = effect_by_code.get(failure["code"])
        candidate_ids = [str(view_id)]
        if "/" in view_id:
            candidate_ids.extend(part.strip() for part in view_id.split("/"))
        for candidate in candidate_ids:
            injection_id = hits.get((candidate, effect))
            if injection_id is not None and injection_id not in hit_ids:
                hit_ids.append(injection_id)
        # Budget injections are global overrides; any matching budget code hit
        # counts regardless of the suspended group's synthetic view id.
        if effect in ("slot_budget_zero", "point_budget_zero"):
            for (injected_view, injected_effect), injection_id in hits.items():
                if injected_effect == effect and injection_id not in hit_ids:
                    hit_ids.append(injection_id)
    for entry in injection_config.get("injections", []):
        entry["hit"] = entry["id"] in hit_ids
    return sorted(hit_ids)


def run_pipeline(input_dir, config_path, injection_path, snapshot_root, report_path):
    config = load_config(config_path)
    injection_config = load_injections(injection_path)
    # Budget-level injections override the configuration deterministically.
    for entry in injection_config.get("injections", []):
        effects = entry.get("effects", {})
        if effects.get("slot_budget_zero"):
            config["slot_budget"] = 0
        if effects.get("point_budget_zero"):
            config["point_budget"] = 0
    views, load_failures = scan_views(input_dir, injection_config["items"])
    cameras = [
        {"view_id": view["view_id"], "K": view["K"].tolist(),
         "R": view["R"].tolist(), "t": view["t"].tolist()}
        for view in views
    ]
    pipeline = ReconstructionPipeline(
        views, config, cameras,
        injection_config=injection_config, snapshot_root=snapshot_root,
    )
    pipeline.install_signal_handlers()
    try:
        report = pipeline.run()
    finally:
        pipeline.restore_signal_handlers()
    report["load_failures"] = sorted(
        load_failures, key=lambda item: (str(item["view_id"]), item["stage"], item["code"])
    )
    report["failures"] = sorted(
        report["failures"] + load_failures,
        key=lambda item: (str(item["view_id"]), item["stage"], item["code"]),
    )
    hit_ids = mark_injection_hits(injection_config, report["failures"])
    report["injection_hits"] = [
        {"id": entry["id"], "view_id": entry["view_id"],
         "effects": entry["effects"], "hit": entry["hit"]}
        for entry in injection_config.get("injections", [])
    ]
    report["hit_injection_ids"] = hit_ids
    atomic_write_json(report_path, report)
    return report, pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description="deterministic multi-view reconstruction")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate a fixed-seed sample directory")
    gen.add_argument("output_dir")
    gen.add_argument("--views", type=int, default=8)

    run = sub.add_parser("run", help="run the reconstruction pipeline")
    run.add_argument("--input", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--injections")
    run.add_argument("--snapshot")
    run.add_argument("--report", required=True)

    args = parser.parse_args(argv)
    if args.command == "generate":
        generate_sample(args.output_dir, args.views)
        return 0
    try:
        report, pipeline = run_pipeline(
            args.input, args.config, args.injections, args.snapshot, args.report
        )
    except PipelineError as exc:
        atomic_write_json(args.report, {"fatal": exc.to_record()})
        return 2
    if pipeline.shutdown_requested.is_set():
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
