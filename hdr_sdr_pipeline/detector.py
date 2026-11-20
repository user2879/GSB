"""Fault detection and detected/missed/false-positive classification.

Detection mechanisms (all deterministic):
  * specification checks: state parameter domains, manifest peak bounds and
    manifest content hash -> E_INJECT_BOUNDS / E_HASH_MISMATCH;
  * stage hashes: a clean recomputation of every frame is compared against
    the committed stage hashes; any disagreement is E_HASH_MISMATCH at the
    exact (frame, stage);
  * output file hashes: committed uint8 files are re-hashed independently.

The injection config carries the oracle (``expect_detected`` per item), so the
report classifies every item and every raw finding into:
  detected        finding matching an injected item expected to be detected
  missed          injected item expected to be detected but no finding fired
  false_positive  finding with no corresponding injected item, or an item
                  explicitly expected to stay silent that nevertheless fired
"""

from dataclasses import replace

from .config import TONEMAP_INPUT_MAX
from .errors import (
    ErrorCode, HashMismatchError, InjectBoundsError,
    StateVersionError,
)
from .hashing import hash_json
from .injection import (
    apply_manifest_tamper, apply_state_tamper, expected_error_code,
)
from .pipeline import InjectionContext, process_frame


def validate_manifest(manifest):
    """Specification-layer manifest checks."""
    findings = []
    expected_hash = manifest.get("manifest_hash")
    recomputed = hash_json(
        {k: v for k, v in manifest.items() if k != "manifest_hash"}
    )
    if expected_hash != recomputed:
        findings.append(
            {"code": ErrorCode.HASH_MISMATCH, "scope": "manifest",
             "expected": expected_hash, "observed": recomputed,
             "message": "truth manifest content hash mismatch"}
        )
    for entry in manifest["frames"]:
        peak = float(entry["peak_luma"])
        if peak < 0.0 or peak > TONEMAP_INPUT_MAX:
            findings.append(
                {"code": ErrorCode.INJECT_BOUNDS, "scope": "frame_truth",
                 "frame": entry["frame"], "observed": peak,
                 "legal_max": TONEMAP_INPUT_MAX,
                 "message": "peak luma outside declared HDR bound"}
            )
        for sample in entry["samples"]:
            if any(not isinstance(v, (int, float)) for v in sample["rgb"]):
                findings.append(
                    {"code": ErrorCode.INJECT_BOUNDS, "scope": "sample",
                     "frame": entry["frame"],
                     "message": "sample reference value not numeric"}
                )
    return findings


def validate_state(state):
    """State domain checks; ProcessingState.validate raises typed errors."""
    try:
        state.validate()
        return []
    except (InjectBoundsError, StateVersionError) as exc:
        return [{"code": exc.code, "scope": "state",
                 "message": str(exc), "details": getattr(exc, "details", {})}]


def verify_stage_hashes(work_dir, manifest, state, context=None,
                        frames_dir=None):
    """Clean recomputation compared with committed stage/output hashes."""
    import os

    from .persistence import read_ledger

    ledger = read_ledger(work_dir)
    frames_dir = frames_dir or os.path.join(work_dir, "frames")
    findings = []
    width = manifest["generator"]["width"]
    height = manifest["generator"]["height"]
    ctx = InjectionContext()  # clean recompute, never the injected one
    for frame_index in range(manifest["generator"]["frame_count"]):
        _u8, stage_hashes, _arrays = process_frame(
            frame_index, frames_dir, state, context=ctx,
            width=width, height=height,
        )
        committed = ledger["outputs"].get(str(frame_index))
        if committed is None:
            continue
        for stage, observed in stage_hashes.items():
            expected = committed["stage_hashes"].get(stage)
            if expected is not None and expected != observed:
                findings.append(
                    {
                        "code": ErrorCode.HASH_MISMATCH,
                        "scope": "stage",
                        "frame": frame_index,
                        "stage": stage,
                        "expected": expected,
                        "observed": observed,
                        "message": f"stage {stage} hash mismatch",
                    }
                )
    return findings


def _match_key(finding):
    if finding["scope"] == "stage":
        return ("stage_replacement", finding.get("frame"), finding.get("stage"))
    if finding["scope"] == "frame_truth":
        return ("peak_out_of_bounds", finding.get("frame"), None)
    if finding["scope"] == "manifest":
        return ("manifest", None, None)
    if finding["scope"] == "state":
        return ("state", None, None)
    return ("unknown", None, None)


def classify(injection_config, findings):
    """Produce the detected/missed/false-positive classification report."""
    items = injection_config.items if injection_config else []
    matched_item_ids = set()
    detected, false_positive = [], []

    item_index = {}
    for item in items:
        if item.kind == "stage_replacement":
            item_index[("stage_replacement", item.frame, item.stage)] = item
        elif item.kind == "peak_out_of_bounds":
            frame_idx = item.payload.get("frame", 0)
            item_index[("peak_out_of_bounds", frame_idx, None)] = item
        elif item.kind == "illegal_transfer_param":
            item_index.setdefault(("state", None, None), []).append(item)
        elif item.kind == "tonemap_constant_tamper":
            item_index.setdefault(("tonemap_tamper", None, None), []).append(
                item
            )
        elif item.kind == "truth_sample_tamper":
            item_index[("manifest", None, None)] = item

    for finding in findings:
        key = _match_key(finding)
        match = item_index.get(key)
        if isinstance(match, list):
            item = next((it for it in match if it.expect_detected), match[0])
        else:
            item = match
        if item is None and finding.get("scope") == "stage":
            pool = item_index.get(("tonemap_tamper", None, None), [])
            item = next((it for it in pool if it.expect_detected), None)
        if item is None and key[0] == "manifest":
            # any truth tamper manifests as a manifest hash mismatch
            item = next((it for it in items
                         if it.kind == "truth_sample_tamper"), None)
        if item is None:
            false_positive.append(
                {"finding": finding,
                 "basis": "no injection item maps to this finding"}
            )
            continue
        matched_item_ids.add(item.item_id)
        if item.expect_detected:
            detected.append(
                {"item_id": item.item_id, "kind": item.kind,
                 "expected_code": expected_error_code(item),
                 "finding_code": finding["code"], "finding": finding}
            )
        else:
            false_positive.append(
                {"finding": finding, "item_id": item.item_id,
                 "basis": "item was expected to stay silent (stealth "
                          "oracle) but a finding fired"}
            )

    missed = []
    for item in items:
        if item.expect_detected and item.item_id not in matched_item_ids:
            missed.append(
                {
                    "item_id": item.item_id,
                    "kind": item.kind,
                    "expected_code": expected_error_code(item),
                    "basis": _miss_basis(item),
                }
            )
    unique_detected = sorted({d["item_id"] for d in detected})
    return {
        "detected": detected,
        "missed": missed,
        "false_positive": false_positive,
        "counts": {
            "injected": len(items),
            "detected": len(unique_detected),
            "missed": len(missed),
            "false_positive": len(
                [fp for fp in false_positive if "item_id" not in fp]
            ),
            "stealth_fired": len(
                [fp for fp in false_positive if "item_id" in fp]
            ),
            "raw_findings": None,
        },
    }


def _miss_basis(item):
    if item.kind == "stage_replacement":
        return (
            "replacement at frame %s stage %s reproduced the committed hash; "
            "the altered bytes never reached a committed output"
            % (item.frame, item.stage)
        )
    if item.kind == "truth_sample_tamper":
        return (
            "manifest hash still matched after a tamper smaller than the "
            "declared detection resolution (or recomputed before commit)"
        )
    return "no deterministic detector fired for the injected condition"


def run_detection(work_dir, manifest, state, injection_config, *, context=None):
    """End-to-end specification + stage detection and classification."""
    findings = []
    findings.extend(validate_manifest(manifest))
    state_findings = validate_state(state)
    findings.extend(state_findings)
    # Stage-hash verification only makes sense once outputs are committed and
    # the state itself is inside its legal domain.
    from .persistence import ledger_exists

    if ledger_exists(work_dir) and not state_findings:
        findings.extend(
            verify_stage_hashes(work_dir, manifest, state, context=context)
        )
    report = classify(injection_config, findings)
    report["findings"] = findings
    report["counts"]["raw_findings"] = len(findings)
    return report


def assert_no_miss(report):
    """Strict acceptance helper: raise E_DETECT_MISS on any missed item."""
    from .errors import DetectMissError

    if report["missed"]:
        raise DetectMissError(
            "%d injected fault(s) were not detected" % len(report["missed"]),
            details={"missed": report["missed"]},
        )
