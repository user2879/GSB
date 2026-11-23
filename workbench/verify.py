"""Equality helpers: incremental-after-edits vs from-scratch full rebuild.

Content equality compares every artifact field; the per-artifact ``origin``
flag (reused vs recomputed) and the budget trace are execution-path metadata
and are excluded by definition (documented in TECH_NOTES.md).
"""
from __future__ import annotations

import copy

import numpy as np

from .snapshot import SnapshotStore


def normalize_report(report: dict) -> dict:
    rep = copy.deepcopy(report)
    rep.pop("budget_trace", None)
    ledger = rep.get("invalidation_ledger", {})
    for group in ("tiles", "fusions"):
        for entry in ledger.get(group, {}).values():
            entry.pop("origin", None)
    return rep


def reports_equal(rep_a: dict, rep_b: dict) -> tuple:
    na, nb = normalize_report(rep_a), normalize_report(rep_b)
    if na != nb:
        diffs = _diff(na, nb, path="")
        return False, diffs
    return True, []


def _diff(a, b, path):
    diffs = []
    if type(a) != type(b):
        diffs.append(f"{path}: type {type(a).__name__} != {type(b).__name__}")
        return diffs
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a:
                diffs.append(f"{path}.{key}: missing in first")
            elif key not in b:
                diffs.append(f"{path}.{key}: missing in second")
            else:
                diffs.extend(_diff(a[key], b[key], f"{path}.{key}"))
    elif isinstance(a, list):
        if len(a) != len(b):
            diffs.append(f"{path}: len {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diffs.extend(_diff(x, y, f"{path}[{i}]"))
    elif a != b:
        diffs.append(f"{path}: {a!r} != {b!r}")
    return diffs


def fusion_artifacts_equal(work_a, work_b) -> tuple:
    sa, sb = SnapshotStore(f"{work_a}/snapshot"), SnapshotStore(f"{work_b}/snapshot")
    state_a = sa.load_manifest_state() or {}
    state_b = sb.load_manifest_state() or {}
    diffs = []
    for pk in sorted(set(state_a.get("fusions", {})) | set(state_b.get("fusions", {}))):
        fa, fb = sa.load_fusion_result(pk), sb.load_fusion_result(pk)
        if fa is None or fb is None:
            diffs.append(f"fusion {pk}: missing on one side")
            continue
        if not np.array_equal(fa["fused"], fb["fused"]):
            diffs.append(f"fusion {pk}: fused raster differs")
        if fa["records"] != fb["records"]:
            diffs.append(f"fusion {pk}: vote records differ")
        if fa["boundary_events"] != fb["boundary_events"]:
            diffs.append(f"fusion {pk}: boundary events differ")
    ga = f"{work_a}/snapshot/global/global_class.npy"
    gb = f"{work_b}/snapshot/global/global_class.npy"
    try:
        if not np.array_equal(np.load(ga), np.load(gb)):
            diffs.append("global class raster differs")
    except FileNotFoundError as exc:
        diffs.append(f"missing global raster: {exc}")
    return (not diffs), diffs
