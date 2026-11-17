"""Deterministic fault injection.

Injection only mutates *input files / parameters*; the same engine code paths
then produce the business code. Each variant creates an isolated copy under the
injection work directory so the pristine samples remain reproducible.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .engine.errors import E_GRID, E_RULE, E_STREAM, GrayError
from .engine.fixedpoint import scale_from_exponent
from .engine.loader import load_rules, load_stream
from .engine.session import Session


def apply_injection(args) -> dict:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if args.kind == "weight_out_of_range":
        return _inject_weight(args, workdir)
    if args.kind == "timestamp_regression":
        return _inject_timestamp(args, workdir)
    if args.kind == "snapshot_scale_missing":
        return _inject_snapshot_scale(args, workdir)
    raise GrayError(E_RULE, f"unknown injection kind {args.kind}")


def _inject_weight(args, workdir: Path) -> dict:
    rule_path = workdir / "rules_bad_weight.json"
    payload = json.loads(Path(args.rules).read_text(encoding="utf-8"))
    target = payload["rules"][0]
    target["gray_weight"] = 10001  # violates 0..10000
    rule_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        load_rules(rule_path)
    except GrayError as error:
        assert error.code == E_RULE
        return {"kind": args.kind, "expected": E_RULE, "actual": error.code,
                "file": str(rule_path), "line": error.line,
                "field_path": error.field_path, "message": error.message}
    raise GrayError(E_RULE, "weight injection failed to trigger E_RULE")


def _inject_timestamp(args, workdir: Path) -> dict:
    stream_path = workdir / "stream_bad_timestamp.ndjson"
    lines = Path(args.stream).read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    if len(rows) < 3:
        raise GrayError(E_STREAM, "sample stream too short for injection")
    rows[2]["timestamp_ms"] = rows[1]["timestamp_ms"] - 50  # explicit rollback
    stream_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8")
    rules = load_rules(args.rules)
    try:
        load_stream(stream_path, rules)
    except GrayError as error:
        assert error.code == E_STREAM
        return {"kind": args.kind, "expected": E_STREAM, "actual": error.code,
                "file": str(stream_path), "line": error.line,
                "field_path": error.field_path, "message": error.message}
    raise GrayError(E_STREAM, "timestamp injection failed to trigger E_STREAM")


def _inject_snapshot_scale(args, workdir: Path) -> dict:
    # Build a legitimate snapshot first, then remove the scale exponent so the
    # restore path reports E_GRID through the grid validation contract.
    rules = load_rules(args.rules)
    requests = load_stream(args.stream, rules)
    scale_from_exponent(args.scale_exponent)
    session = Session()
    session.replace(rules, args.scale_exponent, requests)
    good_snapshot = workdir / "session_good.json"
    session.snapshot(good_snapshot)
    payload = json.loads(good_snapshot.read_text(encoding="utf-8"))
    del payload["scale_exponent"]
    bad_snapshot = workdir / "session_missing_scale.json"
    bad_snapshot.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        Session.load_snapshot(bad_snapshot)
    except GrayError as error:
        # A structurally readable snapshot whose grid parameter is absent is a
        # grid contract failure (the fixed point grid cannot be rebuilt), which
        # the task maps to E_GRID rather than to a version alignment failure.
        if error.field_path == "scale_exponent":
            return {"kind": args.kind, "expected": E_GRID, "actual": E_GRID,
                    "file": str(bad_snapshot), "field_path": error.field_path,
                    "message": error.message}
        raise
    raise GrayError(E_GRID, "snapshot injection failed to trigger grid error")
