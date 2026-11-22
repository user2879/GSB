"""Versioned session snapshots with whole-state replacement and replay checks.

A snapshot is one canonical JSON document carrying pool/task/scale versions,
per-round water levels, the preemption sequence, shard outcome digests, the
ledger cursor and a content digest.  On reopen, any version or digest mismatch
triggers a *whole-state* replacement (never an incremental merge) and a stable
list of differences.  Replay divergence raises ``E_REPLAY`` pinpointing the
first inconsistent round, shard and field.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..digest import canonical_bytes, digest_of
from ..errors import E_REPLAY, DiagError
from ..models import Declaration


@dataclass
class ReplayDiff:
    round_no: int
    shard_id: str | None
    field: str
    expected: Any
    actual: Any

    def as_dict(self) -> dict[str, Any]:
        return {"round": self.round_no, "shard_id": self.shard_id,
                "field": self.field, "expected": self.expected,
                "actual": self.actual}


def build_snapshot(decl: Declaration, rounds: list[dict[str, Any]],
                   outcomes: dict[str, dict[str, Any]],
                   ledger_cursor: int) -> dict[str, Any]:
    body = {
        "snapshot_format_version": 1,
        "versions": {
            "pool_version": decl.params.pool_version,
            "task_version": decl.params.task_version,
            "scale_version": decl.params.scale_version,
            "ledger_format_version": decl.params.ledger_format_version,
        },
        "scale_factor": decl.params.scale_factor,
        "declaration_digest": digest_of({
            "pools": {pid: pool.__dict__ for pid, pool in decl.pools.items()},
            "tasks": {tid: task.__dict__ for tid, task in decl.tasks.items()},
        }),
        "ledger_cursor": ledger_cursor,
        "rounds": [{
            "round": r["round"],
            "water_levels": r["water_levels"],
            "preemptions": r["preemptions"],
            "completed": r["completed"],
        } for r in rounds],
        "preemption_sequence": [p for r in rounds for p in r["preemptions"]],
        "shard_outcomes": outcomes,
    }
    body["content_digest"] = digest_of({k: v for k, v in body.items()})
    return body


def save_snapshot(path: str | Path, snapshot: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(snapshot))


def load_snapshot(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def diff_versions(snapshot: dict[str, Any], decl: Declaration) -> list[dict[str, str]]:
    expected = {
        "pool_version": decl.params.pool_version,
        "task_version": decl.params.task_version,
        "scale_version": decl.params.scale_version,
        "ledger_format_version": decl.params.ledger_format_version,
    }
    stored = snapshot["versions"]
    diffs = []
    for key in sorted(expected):
        if str(stored.get(key)) != str(expected[key]):
            diffs.append({"field": key, "snapshot": str(stored.get(key)),
                          "current": str(expected[key]),
                          "action": "whole-state replacement"})
    current_decl_digest = digest_of({
        "pools": {pid: pool.__dict__ for pid, pool in decl.pools.items()},
        "tasks": {tid: task.__dict__ for tid, task in decl.tasks.items()},
    })
    if snapshot.get("declaration_digest") != current_decl_digest:
        diffs.append({"field": "declaration_digest",
                      "snapshot": snapshot.get("declaration_digest"),
                      "current": current_decl_digest,
                      "action": "whole-state replacement"})
    if snapshot.get("content_digest"):
        recomputed = digest_of({k: v for k, v in snapshot.items()
                                if k != "content_digest"})
        if recomputed != snapshot["content_digest"]:
            diffs.append({"field": "content_digest",
                          "snapshot": snapshot["content_digest"],
                          "current": recomputed,
                          "action": "whole-state replacement"})
    return diffs


def replay_compare(snapshot: dict[str, Any],
                   replay_rounds: list[dict[str, Any]],
                   replay_outcomes: dict[str, dict[str, Any]]) -> None:
    stored_rounds = snapshot["rounds"]
    count = min(len(stored_rounds), len(replay_rounds))
    for index in range(count):
        stored = stored_rounds[index]
        actual = replay_rounds[index]
        if stored["round"] != actual["round"]:
            raise DiagError(E_REPLAY, "replay round cursor divergence",
                            [ReplayDiff(actual["round"], None, "round",
                                        stored["round"], actual["round"]).as_dict()])
        if canonical_bytes(stored["water_levels"]) != \
                canonical_bytes(actual["water_levels"]):
            field, exp, got = _first_mapping_diff(
                stored["water_levels"], actual["water_levels"])
            raise DiagError(E_REPLAY, "replay water level divergence",
                            [ReplayDiff(actual["round"], None,
                                        f"water_levels.{field}", exp, got).as_dict()])
        if canonical_bytes(stored["preemptions"]) != \
                canonical_bytes(actual["preemptions"]):
            raise DiagError(E_REPLAY, "replay preemption sequence divergence",
                            [ReplayDiff(actual["round"], None, "preemptions",
                                        stored["preemptions"],
                                        actual["preemptions"]).as_dict()])
        if list(stored["completed"]) != list(actual["completed"]):
            raise DiagError(E_REPLAY, "replay completion divergence",
                            [ReplayDiff(actual["round"], None, "completed",
                                        stored["completed"],
                                        actual["completed"]).as_dict()])
    if len(stored_rounds) != len(replay_rounds):
        raise DiagError(E_REPLAY, "replay round count divergence",
                        [ReplayDiff(count + 1, None, "round_count",
                                    len(stored_rounds),
                                    len(replay_rounds)).as_dict()])
    stored_outcomes = snapshot["shard_outcomes"]
    for sid in sorted(set(stored_outcomes) | set(replay_outcomes)):
        left = stored_outcomes.get(sid)
        right = replay_outcomes.get(sid)
        if canonical_bytes(left) != canonical_bytes(right):
            field, exp, got = _first_mapping_diff(left or {}, right or {})
            raise DiagError(E_REPLAY, "replay shard outcome divergence",
                            [ReplayDiff(len(replay_rounds), sid,
                                        f"outcome.{field}", exp, got).as_dict()])


def _first_mapping_diff(left: dict[str, Any], right: dict[str, Any]
                        ) -> tuple[str, Any, Any]:
    for key in sorted(set(left) | set(right)):
        if canonical_bytes(left.get(key)) != canonical_bytes(right.get(key)):
            return key, left.get(key), right.get(key)
    return "", None, None
