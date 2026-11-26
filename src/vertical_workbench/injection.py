from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from .config import WorkbenchConfig
from .engine import WorkbenchEngine
from .loader import load_config_file, load_workspace
from .synth import SampleSpec, build_sample, default_specs, write_sample_workspace


def _truncate_idat(png_bytes: bytes) -> bytes:
    marker = png_bytes.find(b"IDAT")
    if marker < 4:
        return png_bytes[: max(0, len(png_bytes) // 2)]
    length_pos = marker - 4
    (chunk_length,) = struct.unpack(">I", png_bytes[length_pos:marker])
    cut_length = max(1, chunk_length // 2)
    return png_bytes[: marker + 4 + cut_length]


def run_injection_suite(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    lab = root / "injection_lab"
    (lab / "samples").mkdir(parents=True, exist_ok=True)
    config = WorkbenchConfig(sample_count=4, max_active_leases=32, global_point_budget=220)
    specs = default_specs(4, config.seed)
    write_sample_workspace(lab, config, specs)

    injected_specs = [
        SampleSpec("inj_png_trunc", config.seed + 9001, (5, 6, 5), 3),
        SampleSpec("inj_kp_nonint", config.seed + 9002, (5, 5, 6), 5),
        SampleSpec("inj_order_skip", config.seed + 9003, (6, 5, 5), 7),
    ]
    for spec in injected_specs:
        bundle = build_sample(spec, config)
        png_path = lab / "samples" / f"{spec.image_id}.png"
        truth_path = lab / "truth" / f"{spec.image_id}.json"
        if spec.image_id == "inj_png_trunc":
            png_path.write_bytes(_truncate_idat(bundle["png_bytes"]))
        else:
            png_path.write_bytes(bundle["png_bytes"])
        truth = bundle["truth"]
        if spec.image_id == "inj_kp_nonint":
            truth["keypoints"][0]["x"] = 12.5
        if spec.image_id == "inj_order_skip":
            truth["keypoints"][1]["reading_order"] = int(
                truth["keypoints"][0]["reading_order"]
            ) + 2
        truth_path.write_text(
            json.dumps(truth, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )

    injection_ids = {
        "inj_png_trunc.png": "INJ-PNG-TRUNC-001",
        "inj_png_trunc.json": "INJ-PNG-TRUNC-001",
        "inj_kp_nonint.json": "INJ-KP-NONINT-002",
        "inj_order_skip.json": "INJ-ORDER-SKIP-003",
    }
    loaded = load_workspace(lab, config=config, injection_ids=injection_ids)
    engine = WorkbenchEngine(lab, write_snapshots=False)
    engine.injection_ids = injection_ids
    result = engine.run(9001)

    invalid_config = config.to_dict()
    invalid_config["quant_gear"] = "unknown_quant_gear"
    invalid_config_path = lab / "configs" / "invalid_quant_workbench.json"
    invalid_config_path.write_text(
        json.dumps(invalid_config, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    _bad_config, config_failure = load_config_file(invalid_config_path)
    config_hit = bool(
        config_failure
        and any("unknown quant_gear" in reason for reason in config_failure["reasons"])
    )

    expected = [
        ("INJ-PNG-TRUNC-001", "truncate_png_data_chunk"),
        ("INJ-KP-NONINT-002", "noninteger_keypoint_x"),
        ("INJ-ORDER-SKIP-003", "reading_order_gap"),
        ("INJ-QUANT-UNKNOWN-004", "unknown_quant_gear"),
    ]
    injection_results = []
    for injection_id, kind in expected:
        if injection_id == "INJ-QUANT-UNKNOWN-004":
            hit = config_hit
            failure = config_failure
        else:
            matches = [f for f in loaded.failures if f.get("injection_id") == injection_id]
            hit = bool(matches)
            failure = matches[0] if matches else None
        injection_results.append(
            {
                "injection_id": injection_id,
                "kind": kind,
                "hit": hit,
                "code": "E_SAMPLE_INVALID" if hit else None,
                "failure": failure,
            }
        )

    good_ids = [spec.image_id for spec in specs]
    processed_ids = [sample["image_id"] for sample in result["samples"]]
    report = {
        "suite": "vertical-workbench-injection/v1",
        "lab": str(lab),
        "all_injections_hit": all(item["hit"] for item in injection_results),
        "injection_results": injection_results,
        "good_sample_ids": good_ids,
        "processed_sample_ids": processed_ids,
        "good_samples_isolated": processed_ids == good_ids,
        "engine_failure_count": len(result["failures"]),
        "business_code_counts": result["business_code_counts"],
    }
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "injection_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    configs_dir = root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    (configs_dir / "injection_suite.json").write_text(
        json.dumps(
            {
                "suite": "vertical-workbench-injection-files/v1",
                "configs": [
                    "configs/injections/inj_png_trunc_001.json",
                    "configs/injections/inj_keypoint_nonint_002.json",
                    "configs/injections/inj_reading_skip_003.json",
                    "configs/injections/inj_quant_unknown_004.json",
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return report
