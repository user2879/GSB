"""Crash recovery and deterministic consistency replay.

Recovery only trusts *committed* ledger batches.  Uncommitted batches are
reported as UNKNOWN and flagged for rerun - no exit report is synthesized.
Replay recompiles the plan from the manifest version and scale recorded in the
ledger, recomputes every expected digest and compares committed receipts
field-by-field in stable order, raising :class:`ReplayError` at the first
inconsistent (path, node, field).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .compiler import Compiler
from .converge import authoritative_recompute
from .errors import ConvergeError, ReplayError
from .ledger import Ledger
from .models import PlanItem, Receipt, VersionManifest


@dataclass
class RecoveryReport:
    unknown_batches: List[Dict[str, Any]]
    committed_batches: List[Dict[str, Any]]
    rerun_required: List[str]
    replayed_items: int
    matched_items: int
    rows: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unknown_batches": self.unknown_batches,
            "committed_batches": self.committed_batches,
            "rerun_required": self.rerun_required,
            "replayed_items": self.replayed_items,
            "matched_items": self.matched_items,
            "rows": self.rows,
        }


def _plan_items(plan_doc: Dict[str, Any]) -> List[PlanItem]:
    items: List[PlanItem] = []
    for doc in plan_doc["items"]:
        items.append(
            PlanItem(
                item_id=doc["item_id"],
                version=int(doc["version"]),
                path=doc["path"],
                node_id=doc["node_id"],
                strategy=doc["strategy"],
                priority=int(doc["priority"]),
                expected_digest=int(doc["expected_digest"]),
                shard_hashes=tuple(int(v) for v in doc["shard_hashes"]),
                shard_hit_digest=int(doc["shard_hit_digest"]),
                expected_error=dict(doc["expected_error"]),
                quant_version=doc["quant_version"],
                compiled_artifact_digest=doc["compiled_artifact_digest"],
            )
        )
    return items


def _receipts(docs: List[Dict[str, Any]]) -> List[Receipt]:
    return [
        Receipt(
            item_id=doc["item_id"],
            node_id=doc["node_id"],
            version=int(doc["version"]),
            path=doc["path"],
            derived_digest=int(doc["derived_digest"]),
            shard_hit_digest=int(doc["shard_hit_digest"]),
            cost_ms=int(doc["cost_ms"]),
            quant_version=doc["quant_version"],
            attempt=int(doc["attempt"]),
            status=doc["status"],
            error_code=doc.get("error_code"),
            waited_ms=int(doc.get("waited_ms", 0)),
            service_ms=int(doc.get("service_ms", 0)),
        )
        for doc in docs
    ]


def reopen(ledger: Ledger) -> RecoveryReport:
    """Classify batches after a hard kill; never claim an exit report."""
    unknown = ledger.unknown_batches()
    committed = ledger.committed_batches()
    rerun_required: List[str] = []
    for batch in unknown:
        ledger.mark_batch_rerun(
            batch["batch_id"],
            "uncommitted batch found on reopen; full rerun required",
        )
        rerun_required.append(batch["batch_id"])
    return RecoveryReport(
        unknown_batches=unknown,
        committed_batches=committed,
        rerun_required=rerun_required,
        replayed_items=0,
        matched_items=0,
        rows=[],
    )


def verify_recompilation(
    ledger: Ledger,
    manifests: List[VersionManifest],
    content_seed: int,
    scale: int,
) -> Tuple[List[PlanItem], List[Dict[str, Any]], str]:
    """Recompile from ledger identity and compare plan digests byte-for-byte."""
    manifest_by_version = {m.version: m for m in manifests}
    compiler = Compiler(content_seed=content_seed, scale=scale)
    diffs: List[Dict[str, Any]] = []
    all_items: List[PlanItem] = []
    plan_versions = [
        row["version"]
        for row in ledger.conn.execute(
            "SELECT version FROM plans ORDER BY version ASC"
        ).fetchall()
    ]
    for version in plan_versions:
        plan_doc = ledger.load_plan(version)
        if plan_doc is None:
            raise ReplayError(
                f"plan for version {version} missing in ledger",
                f"plans[{version}]",
            )
        if int(plan_doc["scale"]) != scale:
            raise ReplayError(
                "scale recorded in ledger differs from replay grid",
                f"plans[{version}].scale",
                {"ledger_scale": plan_doc["scale"], "replay_scale": scale},
            )
        manifest = manifest_by_version.get(version)
        if manifest is None:
            raise ReplayError(
                f"manifest version {version} unavailable for replay",
                f"manifests[{version}]",
            )
        _compiled, recompiled_items = compiler.compile_version(manifest)
        from .ledger import content_digest

        recompiled_digest = content_digest(
            [item.to_dict() for item in recompiled_items]
        )
        if recompiled_digest != plan_doc["plan_digest"]:
            raise ReplayError(
                "recompiled plan digest differs from committed plan",
                f"plans[{version}].plan_digest",
                {
                    "expected": plan_doc["plan_digest"],
                    "actual": recompiled_digest,
                },
            )
        stored_items = _plan_items(plan_doc)
        for stored, fresh in zip(stored_items, recompiled_items):
            if stored.compiled_artifact_digest != fresh.compiled_artifact_digest:
                raise ReplayError(
                    "compiled artifact not reproducible",
                    f"version {version} path {fresh.path} node {fresh.node_id}",
                    {
                        "path": fresh.path,
                        "node_id": fresh.node_id,
                        "ledger_artifact": stored.compiled_artifact_digest,
                        "replay_artifact": fresh.compiled_artifact_digest,
                    },
                )
        all_items.extend(stored_items)
    return all_items, diffs, plan_versions and str(plan_versions[-1]) or ""


def replay(
    ledger: Ledger,
    manifests: List[VersionManifest],
    content_seed: int,
    scale: int,
) -> RecoveryReport:
    """Full recovery + authoritative replay of every committed receipt."""
    report = reopen(ledger)
    items, _diffs, _last_version = verify_recompilation(
        ledger, manifests, content_seed, scale
    )
    receipts = _receipts(ledger.all_committed_receipts())
    try:
        rows, deviations = authoritative_recompute(items, receipts)
    except ConvergeError as exc:
        first = exc.context["deviations"][0]
        field = first["diff_fields"][0]
        raise ReplayError(
            "committed receipt not reproducible on replay",
            f"path {first['path']} node {first['node_id']} field {field}",
            {"first_deviation": first},
        ) from exc
    if deviations:
        first = deviations[0]
        raise ReplayError(
            "committed receipt not reproducible on replay",
            f"path {first['path']} node {first['node_id']} field {first['diff_fields'][0]}",
            {"first_deviation": first},
        )
    report.replayed_items = len(items)
    report.matched_items = sum(1 for row in rows if row.ok)
    report.rows = [row.to_dict() for row in rows]
    return report
