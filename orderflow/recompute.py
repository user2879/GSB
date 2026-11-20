from __future__ import annotations

import json
from pathlib import Path

from .adjudicate import adjudicate
from .model import Config
from .parser import parse_lines


def verify_workspace(workspace, cfg: Config):
    """Independent recompute: read only on-disk snapshots/ledger/samples and
    re-derive the adjudication sequence and arbitration outcomes, locating the
    first mismatching step per match."""
    workspace = Path(workspace)
    report = {}
    matches_dir = workspace / "matches"
    if not matches_dir.exists():
        return report
    for match_dir in sorted(p for p in matches_dir.iterdir() if p.is_dir()):
        manifest_path = match_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        committed = int(manifest["committed"])
        sample = workspace / "samples" / f"{match_dir.name}.jsonl"
        orders, _errors, parse_anomalies = parse_lines(
            sample.read_text(encoding="utf-8").splitlines(), cfg
        )
        res = adjudicate(match_dir.name, orders, parse_anomalies, cfg, store=None)

        first_mismatch = None
        for step in range(committed):
            snap = json.loads(
                (match_dir / "snapshots" / f"step_{step:06d}.json").read_text(encoding="utf-8")
            )
            expected = res["steps"][step]["hash"] if step < len(res["steps"]) else None
            if snap["hash"] != expected:
                first_mismatch = {"step": step, "on_disk": snap["hash"], "recomputed": expected}
                break

        ledger_ok = True
        ledger_path = match_dir / "ledger.jsonl"
        ledger_lines = (
            ledger_path.read_text(encoding="utf-8").splitlines() if ledger_path.exists() else []
        )
        if len(ledger_lines) != committed:
            ledger_ok = False
        else:
            for step, line in enumerate(ledger_lines):
                entry = json.loads(line)
                if step >= len(res["steps"]) or entry["snapshot_hash"] != res["steps"][step]["hash"]:
                    ledger_ok = False
                    break

        report[match_dir.name] = {
            "status": "OK" if first_mismatch is None and ledger_ok else "MISMATCH",
            "checked_steps": committed,
            "first_mismatch": first_mismatch,
            "ledger_ok": ledger_ok,
        }
    return report
