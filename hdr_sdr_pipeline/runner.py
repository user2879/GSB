"""Example orchestration entry point for the HDR->SDR component.

Sub-commands:
  generate    synthesize the deterministic frame sequence + truth manifest
  run         full build, incremental transitions, injection campaign,
              localization, minimal counterexample and clean shutdown
  crash-demo  force a crash mid-commit and demonstrate restart recovery

All artifacts live under the chosen working directory; re-running with the
same configuration produces byte-identical outputs.
"""

import argparse
import json
import os
import shutil

import numpy as np

from .config import ProcessingState
from .detector import run_detection
from .engine import IncrementalEngine
from .hashing import canonical_json
from .injection import (
    InjectionConfig, illegal_transfer_param, peak_out_of_bounds,
    stage_replacement, tonemap_constant_tamper, truth_sample_tamper,
)
from .locate import locate_first_affected, minimal_counterexample
from .ledger import commit_frame_output, new_ledger
from .persistence import (
    atomic_write_json, is_closed, recover, write_closed_marker, write_ledger,
)
from .pipeline import InjectionContext, process_frame
from .synth import generate_sequence


def reset_workdir(work_dir):
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)


def cmd_generate(args):
    reset_workdir(args.work_dir)
    manifest = generate_sequence(args.work_dir)
    print("generated %d frames under %s"
          % (manifest["generator"]["frame_count"], args.work_dir))
    print("manifest hash:", manifest["manifest_hash"])
    return manifest


def _save_injection(work_dir, config):
    path = os.path.join(work_dir, "injections", config.config_id + ".json")
    atomic_write_json(path, config.to_dict())
    return path


def build_context(config, manifest):
    """Construct pipeline-stage replacements from serialized items."""
    replacements = {}
    width = manifest["generator"]["width"]
    height = manifest["generator"]["height"]
    for item in config.items:
        if item.kind != "stage_replacement":
            continue
        payload = item.payload["replacement"]
        if item.stage == "quantize":
            arr = np.full((height, width, 3), int(payload["value"]),
                          dtype=np.uint8)
        else:
            arr = np.full((height, width, 3), float(payload["value"]),
                          dtype=np.float32)
        replacements[(item.frame, item.stage)] = arr
    return InjectionContext(stage_replacements=replacements)


def _mirror_frames(src, dst):
    """Mirror the immutable inputs (frames + truth) into a fresh workdir."""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)
    shutil.copytree(os.path.join(src, "frames"), os.path.join(dst, "frames"))
    shutil.copy2(os.path.join(src, "truth.json"),
                 os.path.join(dst, "truth.json"))


def _output_hashes(ledger):
    return {k: rec["output_hash"] for k, rec in ledger["outputs"].items()}


def _outside_unchanged(ledger, before, changed_frames):
    changed = {str(f) for f in changed_frames}
    for key, h in before.items():
        if key not in changed and ledger["outputs"][key]["output_hash"] != h:
            return False
    return True


def _campaign_config():
    """Mixed-layer campaign with explicit detected/missed oracle items."""
    return InjectionConfig(
        config_id="campaign_demo",
        items=[
            peak_out_of_bounds("inj_peak_oob"),
            illegal_transfer_param("inj_gamma_bad"),
            truth_sample_tamper(
                "inj_truth_sample", frame=3, row=10, col=10, delta=0.5
            ),
            tonemap_constant_tamper("inj_aces_a"),
            stage_replacement(
                "inj_stage_tm", frame=20, stage="tonemap",
                replacement={"value": 0.125},
                note="force frame 20 tonemap output to a constant",
            ),
        ],
    )


def _commit_injected_frame(work_dir, ledger, manifest, state, context,
                           frame_index):
    width = manifest["generator"]["width"]
    height = manifest["generator"]["height"]
    u8, stage_hashes, _ = process_frame(
        frame_index, os.path.join(work_dir, "frames"), state,
        context=context, width=width, height=height,
    )
    commit_frame_output(
        work_dir, ledger, frame_index, u8, stage_hashes,
        seq=ledger["current_seq"], recomputed_stages=list(stage_hashes),
    )


def cmd_run(args):
    work_dir = args.work_dir
    if not os.path.exists(os.path.join(work_dir, "truth.json")):
        generate_sequence(work_dir)
    manifest = json.load(open(os.path.join(work_dir, "truth.json")))

    # ---- incremental pipeline on the primary workdir ----
    state = ProcessingState()
    ledger = new_ledger(state, manifest["generator"]["frame_count"],
                        manifest["manifest_hash"])
    engine = IncrementalEngine(work_dir, ledger, manifest, state)
    engine.build_all()
    write_ledger(work_dir, ledger)
    snapshot_hashes = _output_hashes(ledger)

    state_b = ProcessingState(exposure_ev=1.0)
    engine.transition(state_b, target_frames=list(range(0, 16)),
                      route="interval")
    write_ledger(work_dir, ledger)
    outside_unchanged = _outside_unchanged(
        ledger, snapshot_hashes, range(0, 16))

    state_c = ProcessingState(exposure_ev=1.0, tonemap_operator="aces")
    engine.transition(state_c, target_frames=list(range(16, 32)),
                      route="interval")
    write_ledger(work_dir, ledger)
    interval_hashes = _output_hashes(ledger)

    # ---- per-frame route reaches the same final state on a mirror ----
    cmp_dir = work_dir + "_cmp"
    if os.path.exists(cmp_dir):
        shutil.rmtree(cmp_dir)
    os.makedirs(cmp_dir, exist_ok=True)
    _mirror_frames(work_dir, cmp_dir)
    cmp_manifest = json.load(open(os.path.join(cmp_dir, "truth.json")))
    ledger2 = new_ledger(ProcessingState(),
                         cmp_manifest["generator"]["frame_count"],
                         cmp_manifest["manifest_hash"])
    engine2 = IncrementalEngine(cmp_dir, ledger2, cmp_manifest,
                                ProcessingState())
    engine2.build_all()
    engine2.transition(ProcessingState(exposure_ev=1.0),
                       target_frames=list(range(0, 16)), route="interval")
    engine2.transition_per_frame(
        ProcessingState(exposure_ev=1.0, tonemap_operator="aces"),
        target_frames=list(range(16, 32)),
    )
    write_ledger(cmp_dir, ledger2)
    route_equal = interval_hashes == _output_hashes(ledger2)

    # ---- isolated fault workdir with one committed state ----
    fault_dir = work_dir + "_fault"
    if os.path.exists(fault_dir):
        shutil.rmtree(fault_dir)
    os.makedirs(fault_dir, exist_ok=True)
    _mirror_frames(work_dir, fault_dir)
    fault_manifest = json.load(open(os.path.join(fault_dir, "truth.json")))
    fault_state = ProcessingState(tonemap_operator="aces")
    fault_ledger = new_ledger(fault_state,
                              fault_manifest["generator"]["frame_count"],
                              fault_manifest["manifest_hash"])
    fault_engine = IncrementalEngine(fault_dir, fault_ledger, fault_manifest,
                                     fault_state)
    fault_engine.build_all()

    campaign = _campaign_config()
    inj_path = _save_injection(fault_dir, campaign)
    fault_ledger["injection_files"].append(
        os.path.relpath(inj_path, fault_dir))
    context = build_context(campaign, fault_manifest)
    _commit_injected_frame(fault_dir, fault_ledger, fault_manifest,
                           fault_state, context, frame_index=20)
    write_ledger(fault_dir, fault_ledger)

    from .detector import classify, validate_manifest, validate_state
    from .injection import apply_manifest_tamper, apply_state_tamper

    width = fault_manifest["generator"]["width"]
    height = fault_manifest["generator"]["height"]
    spec_kinds = ("peak_out_of_bounds", "illegal_transfer_param",
                  "truth_sample_tamper")
    spec_items = [it for it in campaign.items if it.kind in spec_kinds]
    constant_items = [it for it in campaign.items
                      if it.kind == "tonemap_constant_tamper"]
    pipe_items = [it for it in campaign.items
                  if it.kind == "stage_replacement"]

    # --- specification layer: tampered manifest/state domain checks ---
    tampered_manifest = fault_manifest
    tampered_state = fault_state
    for item in spec_items:
        if item.kind in ("peak_out_of_bounds", "truth_sample_tamper"):
            tampered_manifest = apply_manifest_tamper(tampered_manifest, item)
        else:
            tampered_state = apply_state_tamper(tampered_state, item)
    spec_findings = validate_manifest(tampered_manifest)
    spec_findings.extend(validate_state(tampered_state))
    spec_report = classify(
        InjectionConfig("spec_layer", spec_items), spec_findings
    )

    # --- pipeline layer (a): committed faulty frame 20, clean parameters ---
    pipe_report = run_detection(
        fault_dir, fault_manifest, fault_state,
        InjectionConfig("pipe_layer", pipe_items), context=context
    )

    # --- pipeline layer (b): tone-map constant tamper is parameter-legal but
    #     changes every recomputed stage hash. Check against an independent
    #     clean ledger so the frame-20 fault cannot contaminate this result.
    const_dir = work_dir + "_const"
    if os.path.exists(const_dir):
        shutil.rmtree(const_dir)
    os.makedirs(const_dir, exist_ok=True)
    _mirror_frames(work_dir, const_dir)
    const_manifest = json.load(open(os.path.join(const_dir, "truth.json")))
    const_state = ProcessingState(tonemap_operator="aces")
    const_ledger = new_ledger(const_state,
                              const_manifest["generator"]["frame_count"],
                              const_manifest["manifest_hash"])
    const_engine = IncrementalEngine(const_dir, const_ledger, const_manifest,
                                     const_state)
    const_engine.build_all()
    write_ledger(const_dir, const_ledger)

    tampered_const_state = const_state
    for item in constant_items:
        tampered_const_state = apply_state_tamper(tampered_const_state, item)
    from .detector import verify_stage_hashes

    constant_findings = verify_stage_hashes(
        const_dir, const_manifest, tampered_const_state
    )
    constant_report = classify(
        InjectionConfig("constant_layer", constant_items), constant_findings
    )

    # False-positive demonstration: an unrelated rogue finding maps to no item.
    rogue = {"code": "E_HASH_MISMATCH", "scope": "stage", "frame": 50,
             "stage": "transfer", "expected": "n/a", "observed": "n/a",
             "message": "synthetic rogue finding for FP demo"}
    fp_report = classify(
        InjectionConfig("pipe_layer", pipe_items),
        pipe_report["findings"] + [rogue],
    )

    # Missed-injection demonstration: transient fault overwritten before
    # commit leaves no committed hash for any detector to observe.
    transient = stage_replacement(
        "inj_transient_overwritten", frame=7, stage="tonemap",
        replacement={"value": 0.9},
        note="recomputed clean before commit, so the ledger never sees it",
    )
    transient_ctx = build_context(
        InjectionConfig("transient", [transient]), fault_manifest
    )
    _u8_dirty, _h_dirty, _ = process_frame(
        7, os.path.join(fault_dir, "frames"), fault_state,
        context=transient_ctx, width=width, height=height,
    )
    u8_clean, h_clean, _ = process_frame(
        7, os.path.join(fault_dir, "frames"), fault_state,
        width=width, height=height,
    )
    commit_frame_output(
        fault_dir, fault_ledger, 7, u8_clean, h_clean,
        seq=fault_ledger["current_seq"], recomputed_stages=list(h_clean),
    )
    transient_report = run_detection(
        fault_dir, fault_manifest, fault_state,
        InjectionConfig("transient", [transient]),
    )

    merged_counts = {
        "injected": len(campaign.items) + 1,
        "detected": spec_report["counts"]["detected"]
        + pipe_report["counts"]["detected"]
        + constant_report["counts"]["detected"],
        "missed": 1 if not any(
            f.get("frame") == 7 for f in transient_report["findings"]
        ) else 0,
        "false_positive": len([
            fp for fp in fp_report["false_positive"]
            if fp.get("finding", {}).get("frame") == 50
        ]),
        "spec_layer": spec_report["counts"],
        "pipeline_stage_layer": pipe_report["counts"],
        "constant_layer": constant_report["counts"],
    }
    report = {
        "spec_report": spec_report,
        "pipeline_report": pipe_report,
        "false_positive_demo": {
            "rogue_finding": rogue,
            "classified_false_positive": any(
                fp.get("finding", {}).get("frame") == 50
                for fp in fp_report["false_positive"]
            ),
        },
        "missed_demo": {
            "item_id": transient.item_id,
            "committed_hash_clean": h_clean["quantize"],
            "detector_findings_for_item": len([
                f for f in transient_report["findings"]
                if f.get("frame") == 7
            ]),
            "judgement": "the transient fault never reached a committed "
                         "hash, so no deterministic detector can observe it",
        },
        "constant_report": constant_report,
        "counts": merged_counts,
        "findings": (spec_findings + pipe_report["findings"]
                     + constant_findings),
    }
    atomic_write_json(os.path.join(fault_dir, "detection_report.json"),
                      report)

    first, _r, all_locations = locate_first_affected(
        fault_dir, fault_manifest, fault_state, campaign, context=context
    )
    atomic_write_json(os.path.join(fault_dir, "first_affected.json"),
                      {"first": first, "all": all_locations})

    def context_builder(items):
        return build_context(
            InjectionConfig(config_id=campaign.config_id, items=items),
            fault_manifest,
        )

    mce = minimal_counterexample(
        fault_dir, fault_manifest, fault_state, campaign,
        context_builder=context_builder,
    )
    atomic_write_json(os.path.join(fault_dir, "mce.json"), mce.to_dict())

    write_closed_marker(fault_dir, fault_ledger)
    write_ledger(fault_dir, fault_ledger)
    write_closed_marker(work_dir, ledger)
    write_ledger(work_dir, ledger)

    summary = {
        "outside_closure_unchanged": outside_unchanged,
        "interval_vs_perframe_equal": route_equal,
        "detection_counts": report["counts"],
        "first_affected": first,
        "mce_items": [it.item_id for it in mce.items],
        "closed": is_closed(work_dir),
        "fault_workdir_closed": is_closed(fault_dir),
    }
    atomic_write_json(os.path.join(work_dir, "run_summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def cmd_crash_demo(args):
    """Simulate a forced termination inside a commit and prove recovery."""
    work_dir = args.work_dir
    if not os.path.exists(os.path.join(work_dir, "truth.json")):
        generate_sequence(work_dir)
    manifest = json.load(open(os.path.join(work_dir, "truth.json")))

    state = ProcessingState()
    ledger = new_ledger(state, manifest["generator"]["frame_count"],
                        manifest["manifest_hash"])
    engine = IncrementalEngine(work_dir, ledger, manifest, state)
    engine.build_all()
    write_ledger(work_dir, ledger)

    # Simulated forced termination: unpublished temp file, no closed marker.
    tmp = os.path.join(work_dir, "frames_out", "frame_0000.u8.tmp.crash")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    with open(tmp, "wb") as fh:
        fh.write(b"\x00" * 64)
    assert not is_closed(work_dir)

    recovered_ledger, info = recover(work_dir)
    atomic_write_json(os.path.join(work_dir, "crash_recovery.json"), info)

    engine2 = IncrementalEngine(work_dir, recovered_ledger, manifest,
                                ProcessingState())
    engine2.transition(ProcessingState(exposure_ev=0.5),
                       target_frames=list(range(0, 8)), route="interval")
    write_ledger(work_dir, recovered_ledger)

    fresh_dir = work_dir + "_fresh"
    if os.path.exists(fresh_dir):
        shutil.rmtree(fresh_dir)
    os.makedirs(fresh_dir, exist_ok=True)
    _mirror_frames(work_dir, fresh_dir)
    fresh_manifest = json.load(open(os.path.join(fresh_dir, "truth.json")))
    fresh_ledger = new_ledger(ProcessingState(),
                              fresh_manifest["generator"]["frame_count"],
                              fresh_manifest["manifest_hash"])
    fresh_engine = IncrementalEngine(fresh_dir, fresh_ledger, fresh_manifest,
                                     ProcessingState())
    fresh_engine.build_all()
    fresh_engine.transition(ProcessingState(exposure_ev=0.5),
                            target_frames=list(range(0, 8)),
                            route="interval")
    write_ledger(fresh_dir, fresh_ledger)

    equal = _output_hashes(recovered_ledger) == _output_hashes(fresh_ledger)
    write_closed_marker(work_dir, recovered_ledger)
    write_ledger(work_dir, recovered_ledger)
    result = {
        "orphans_reclaimed": len(info["reclaimed"]),
        "recovered_seq": recovered_ledger["committed_seq"],
        "post_recovery_matches_full_rerun": equal,
        "closed": is_closed(work_dir),
    }
    atomic_write_json(os.path.join(work_dir, "crash_summary.json"), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="HDR->SDR pipeline runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate")
    p_gen.add_argument("--work-dir", default="work/demo")
    p_gen.set_defaults(func=cmd_generate)

    p_run = sub.add_parser("run")
    p_run.add_argument("--work-dir", default="work/demo")
    p_run.set_defaults(func=cmd_run)

    p_crash = sub.add_parser("crash-demo")
    p_crash.add_argument("--work-dir", default="work/crash")
    p_crash.set_defaults(func=cmd_crash_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
