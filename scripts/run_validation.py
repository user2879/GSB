"""End-to-end acceptance harness.

Produces a single diagnostic report covering:
  * deterministic fixed-point conclusions on the local generated corpus,
  * an edit sequence (add/replace sample, add/delete assertion, relation edit,
    threshold/frac/quant changes) with incremental vs clean full rebuild diff,
  * serial vs concurrent field equality,
  * cross-quantization-level consistency summary,
  * injection-id -> business-code hit mapping,
  * forced-termination snapshot recovery and recovery point version.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from mm_pipeline.config import parse_config
from mm_pipeline.injection import load_injections
from mm_pipeline.pipeline import execute_pipeline, write_report
from mm_pipeline.report import compare_runs
from scripts.generate_samples import build_assertions, build_relations

COMPARE_FIELDS = ("region_results", "boundary_samples", "ledger", "failures")


def comparable(report: dict) -> dict:
    out = {field: report[field] for field in COMPARE_FIELDS}
    for row in out["region_results"]:
        row.pop("reused", None)
    out["failures"] = [f for f in out["failures"] if f["code"] != "E_SNAPSHOT_STALE"]
    return out


def assert_equal(label: str, a: dict, b: dict, checks: list) -> None:
    result = compare_runs("incremental", comparable(a), "full_rebuild", comparable(b))
    checks.append({"check": label, **result})
    if not result["equal"]:
        raise AssertionError(f"{label} mismatch: {result['mismatches'][:1]}")


def clone(src: Path, dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst


def edit_assertion(sample_dir: Path, mutate) -> None:
    path = sample_dir / "assertions.json"
    doc = json.loads(path.read_text())
    mutate(doc)
    path.write_text(json.dumps(doc, sort_keys=True))


def edit_relation(sample_dir: Path, mutate) -> None:
    path = sample_dir / "relations.json"
    doc = json.loads(path.read_text())
    mutate(doc)
    path.write_text(json.dumps(doc, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    workspace = Path(args.workspace)
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    evolving = clone(Path(args.samples), workspace / "samples_evolving")
    config_path = Path(args.config)
    inc_work = workspace / "work_incremental"
    full_counter = {"n": 0}

    def full_rebuild() -> dict:
        full_counter["n"] += 1
        return execute_pipeline(
            evolving, config_path,
            workspace / f"work_full_{full_counter['n']}",
            force_reset=True,
        )

    checks: list[dict] = []

    # 0. Baseline.
    inc = execute_pipeline(evolving, config_path, inc_work)
    assert_equal("baseline", inc, full_rebuild(), checks)

    # 1. Add a sample.
    from scripts.generate_samples import render_image
    new_id = "sample_900"
    new_dir = evolving / new_id
    new_dir.mkdir(parents=True)
    seed = 99001
    (new_dir / "image.png").write_bytes(render_image(seed))
    assertions = build_assertions(new_id, seed)
    (new_dir / "assertions.json").write_text(json.dumps(assertions, sort_keys=True))
    (new_dir / "relations.json").write_text(
        json.dumps(build_relations(new_id, assertions["assertions"], seed), sort_keys=True)
    )
    inc = execute_pipeline(evolving, config_path, inc_work)
    assert_equal("add_sample", inc, full_rebuild(), checks)
    add_audit = {
        "reused_count": len(inc["audit"]["reused"]),
        "recomputed_count": len(inc["audit"]["recomputed"]),
        "new_sample_recomputed": any(
            item.startswith("feature:sample_900") for item in inc["audit"]["recomputed"]
        ),
    }

    # 2. Replace an image.
    first_sample = sorted(p.name for p in evolving.iterdir() if p.is_dir())[0]
    (evolving / first_sample / "image.png").write_bytes(render_image(7777))
    inc = execute_pipeline(evolving, config_path, inc_work)
    assert_equal("replace_image", inc, full_rebuild(), checks)
    affected = {item.split(":")[1] for item in inc["audit"]["recomputed"]}
    image_audit = {"recomputed_samples": sorted(affected), "edited_sample": first_sample}

    # 3. Add assertion + relation.
    edit_assertion(
        new_dir,
        lambda doc: doc["assertions"].append(
            {"assertion_id": "a99", "entity": doc["entities"][0],
             "attribute": "density", "value": 0.321}
        ),
    )
    edit_relation(
        new_dir,
        lambda doc: doc["relations"].append(
            {"relation_id": "rel99", "relation_type": "supplement",
             "assertion_id": "a99", "region": 1, "weight": 1.2}
        ),
    )
    inc = execute_pipeline(evolving, config_path, inc_work)
    assert_equal("add_assertion_relation", inc, full_rebuild(), checks)

    # 4. Delete assertion + its relations, edit a relation weight.
    edit_assertion(
        new_dir,
        lambda doc: doc.update(
            assertions=[a for a in doc["assertions"] if a["assertion_id"] != "a99"]
        ),
    )
    edit_relation(
        new_dir,
        lambda doc: doc.update(
            relations=[r for r in doc["relations"] if r["assertion_id"] != "a99"]
        ),
    )
    edit_relation(
        evolving / first_sample,
        lambda doc: doc["relations"][0].__setitem__("weight", 1.31),
    )
    inc = execute_pipeline(evolving, config_path, inc_work)
    assert_equal("delete_assertion_edit_relation", inc, full_rebuild(), checks)

    # 5. Threshold / frac / quant switches (config hash change -> stale reset).
    base_cfg = parse_config(config_path).to_dict()
    switch_rows = []
    for name, overrides in (
        ("threshold", {"confidence_threshold": 0.63}),
        ("frac_bits", {"frac_bits": 24, "quant_level": 16}),
        ("quant_level", {"frac_bits": 24, "quant_level": 8}),
    ):
        tuned_path = workspace / f"cfg_{name}.yaml"
        tuned_path.write_text(yaml.safe_dump({**base_cfg, **overrides}, sort_keys=True))
        switched = execute_pipeline(evolving, tuned_path, inc_work)
        clean = execute_pipeline(
            evolving, tuned_path, workspace / f"work_switch_{name}", force_reset=True
        )
        result = compare_runs(
            "incremental", comparable(switched), "full_rebuild", comparable(clean)
        )
        checks.append({"check": f"switch_{name}", **result})
        if not result["equal"]:
            raise AssertionError(name)
        switch_rows.append(
            {
                "switch": name,
                "overrides": overrides,
                "stale_detected": switched.get("stale_snapshot", {}).get("code")
                == "E_SNAPSHOT_STALE",
                "region_count": len(switched["region_results"]),
                "boundary_count": len(switched["boundary_samples"]),
            }
        )

    # 6. Serial vs concurrent.
    serial_cfg = workspace / "cfg_serial.yaml"
    parallel_cfg = workspace / "cfg_parallel.yaml"
    serial_cfg.write_text(yaml.safe_dump({**base_cfg, "max_workers": 1}))
    parallel_cfg.write_text(yaml.safe_dump({**base_cfg, "max_workers": 16}))
    serial = execute_pipeline(evolving, serial_cfg, workspace / "work_serial",
                              force_reset=True)
    parallel = execute_pipeline(evolving, parallel_cfg, workspace / "work_parallel",
                                force_reset=True)
    concurrency = compare_runs(
        "serial", comparable(serial), "concurrent", comparable(parallel)
    )
    checks.append({"check": "serial_vs_concurrent", **concurrency})

    # 7. Cross-quantization-level consistency summary.
    quant_summary = []
    for level in (8, 12, 16):
        qcfg = workspace / f"cfg_q{level}.yaml"
        qcfg.write_text(yaml.safe_dump(
            {**base_cfg, "quant_level": level, "frac_bits": 24}, sort_keys=True))
        qreport = execute_pipeline(evolving, qcfg, workspace / f"work_q{level}",
                                   force_reset=True)
        confidence_fingerprint = __import__(
            "mm_pipeline.hashing", fromlist=["hash_iterable"]
        ).hash_iterable(
            [r["confidence_fixed"] for r in qreport["region_results"]],
            prefix=f"conf-q{level}",
        )[:16]
        modality_ties = sum(
            1 for r in qreport["region_results"]
            if r["visual_quantized"] == r["text_quantized"]
        )
        quant_summary.append(
            {
                "quant_level": level,
                "regions": len(qreport["region_results"]),
                "boundary": len(qreport["boundary_samples"]),
                "modality_tie_regions": modality_ties,
                "confidence_fingerprint": confidence_fingerprint,
                "categories": {
                    cat: sum(1 for r in qreport["region_results"] if r["category"] == cat)
                    for cat in ("support", "reject", "neutral")
                },
            }
        )

    # 7b. Boundary demonstration: exact threshold tie and modality-tie path.
    from mm_pipeline.fixedpoint import to_float
    probe = execute_pipeline(evolving, config_path, workspace / "work_probe",
                             force_reset=True)
    tied_conf = probe["region_results"][0]["confidence_fixed"]
    boundary_cfg = workspace / "cfg_boundary.yaml"
    boundary_cfg.write_text(yaml.safe_dump(
        {**base_cfg,
         "confidence_threshold": to_float(tied_conf, base_cfg["frac_bits"])},
        sort_keys=True))
    boundary_report = execute_pipeline(
        evolving, boundary_cfg, workspace / "work_boundary", force_reset=True
    )
    boundary_demo = {
        "pinned_confidence_fixed": tied_conf,
        "threshold_tie_regions": sum(
            1 for row in boundary_report["region_results"]
            if "threshold_tie" in row["boundary_reasons"]
        ),
        "modality_tie_regions": sum(
            1 for row in boundary_report["region_results"]
            if "modality_tie" in row["boundary_reasons"]
        ),
        "exact_tie_examples": [
            {
                "sample_id": row["sample_id"],
                "region": row["region"],
                "category": row["category"],
                "boundary_reasons": row["boundary_reasons"],
            }
            for row in boundary_report["region_results"]
            if row["boundary"]
        ][:8],
    }

    # 8. Injection diagnostics on a fresh copy of the original corpus.
    injection_path = Path("configs/injections.yaml")
    file_only = workspace / "injections_file.yaml"
    all_injections = load_injections(injection_path)
    file_only.write_text(
        yaml.safe_dump(
            {
                "injections": [
                    {
                        "injection_id": inj.injection_id,
                        "injection_type": inj.injection_type,
                        "sample_id": inj.sample_id,
                        "params": inj.params,
                    }
                    for inj in all_injections
                    if inj.injection_type
                    in ("truncate_image", "missing_assertion_field",
                        "dangling_relation_region")
                ]
            },
            sort_keys=True,
        )
    )
    injected = execute_pipeline(
        Path(args.samples), config_path, workspace / "work_injected",
        injection_path=file_only, force_reset=True,
    )
    injection_hits = injected["injections"]

    # 9. Forced termination and recovery.
    marker = workspace / "crash_marker.txt"
    crash_work = workspace / "work_crash"
    code = (
        "import sys; from mm_pipeline.pipeline import execute_pipeline; "
        "execute_pipeline("
        + repr(str(Path(args.samples))) + ", "
        + repr(str(config_path)) + ", "
        + repr(str(crash_work)) + ", "
        + "crash_env={'MM_CRASH_AFTER_STAGE':'feature', 'MM_CRASH_MARKER': "
        + repr(str(marker)) + "})"
    )
    crashed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    recovered = execute_pipeline(
        Path(args.samples), config_path, crash_work
    )
    clean = execute_pipeline(
        Path(args.samples), config_path, workspace / "work_crash_clean",
        force_reset=True,
    )
    recovery_equal = compare_runs(
        "recovered", comparable(recovered), "clean", comparable(clean)
    )
    checks.append({"check": "crash_recovery", **recovery_equal})

    final_report = {
        "acceptance": {
            "all_checks_equal": all(item["equal"] for item in checks),
            "checks": checks,
        },
        "conclusion_summary": {
            "regions": len(inc["region_results"]),
            "ledger_entries": len(inc["ledger"]),
            "boundary_samples": len(inc["boundary_samples"]),
            "categories": {
                cat: sum(1 for r in inc["region_results"] if r["category"] == cat)
                for cat in ("support", "reject", "neutral")
            },
        },
        "incremental_audit": {
            "add_sample": add_audit,
            "replace_image": image_audit,
            "final_reused": len(inc["audit"]["reused"]),
            "final_recomputed": len(inc["audit"]["recomputed"]),
        },
        "config_switches": switch_rows,
        "cross_quantization": quant_summary,
        "boundary_demo": boundary_demo,
        "injection_hits": injection_hits,
        "snapshot_recovery": {
            "crash_exit_code": crashed.returncode,
            "marker_written": marker.exists(),
            "recovery_version": recovered["meta"]["recovery_version"],
            "final_version": recovered["meta"]["final_version"],
            "equal_to_clean": recovery_equal["equal"],
        },
        "sample_region_example": inc["region_results"][0],
    }
    write_report(final_report, args.out)
    print(json.dumps({
        "out": args.out,
        "all_equal": final_report["acceptance"]["all_checks_equal"],
        "checks": len(checks),
        "injection_hits": sum(1 for i in injection_hits if i["hit"]),
    }, ensure_ascii=False, indent=2))
    return 0 if final_report["acceptance"]["all_checks_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
