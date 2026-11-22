"""Session state machine and whole-state snapshot replacement.

States: INIT -> LOADED -> COMPILED -> DISPATCHING -> COMMITTED -> CONVERGED
(and RECOVERED / REPLACED / FAILED as recovery branches).  Every transition is
persisted as a state event.  On reopen the stored manifest version plus content
digest are compared to the current inputs: any mismatch triggers *whole-state*
replacement (never incremental merge) with a stable before/after diff.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .compiler import Compiler
from .content import canonical_json
from .ledger import Ledger, content_digest
from .logging_setup import JsonlLogger, mono_ms
from .models import CompiledPath, PlanItem, VersionManifest

LEGAL_TRANSITIONS = {
    "INIT": {"LOADED"},
    "LOADED": {"COMPILED", "FAILED"},
    "COMPILED": {"DISPATCHING", "REPLACED", "FAILED"},
    "DISPATCHING": {"COMMITTED", "FAILED"},
    "COMMITTED": {"CONVERGED", "DISPATCHING", "FAILED"},
    "CONVERGED": {"COMPILED", "DISPATCHING", "REPLACED"},
    "RECOVERED": {"COMPILED", "CONVERGED", "FAILED"},
    "REPLACED": {"COMPILED", "FAILED"},
    "FAILED": {"LOADED", "REPLACED"},
}


class IllegalTransition(RuntimeError):
    pass


@dataclass
class SnapshotDiff:
    replaced: bool
    reason: str
    before: Dict[str, Any]
    after: Dict[str, Any]
    changed_items: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "replaced": self.replaced,
            "reason": self.reason,
            "before": self.before,
            "after": self.after,
            "changed_items": self.changed_items,
        }


class Session:
    def __init__(
        self,
        ledger: Ledger,
        logger: JsonlLogger,
        content_seed: int,
        scale: int,
    ) -> None:
        self.ledger = ledger
        self.logger = logger
        self.content_seed = content_seed
        self.scale = scale
        self.state = "INIT"
        self.compiler = Compiler(content_seed=content_seed, scale=scale)
        self.plan_items: List[PlanItem] = []
        self.compiled: Dict[str, CompiledPath] = {}

    def transition(self, to_state: str, reason: str, detail: Any) -> None:
        allowed = LEGAL_TRANSITIONS.get(self.state, set())
        if to_state not in allowed:
            raise IllegalTransition(
                f"cannot move {self.state} -> {to_state} (allowed {sorted(allowed)})"
            )
        ts = mono_ms()
        self.ledger.record_state(ts, self.state, to_state, reason, detail)
        self.logger.info(
            "state_transition",
            **{"from": self.state, "to": to_state, "reason": reason},
        )
        self.state = to_state

    def snapshot_identity(
        self, manifest: VersionManifest
    ) -> Dict[str, Any]:
        return {
            "version": manifest.version,
            "scale": self.scale,
            "content_seed": self.content_seed,
            "manifest_digest": content_digest(
                [entry.to_dict() for entry in manifest.entries]
            ),
            "quant_version": self.compiler.quant,
        }

    def check_or_replace_snapshot(
        self, manifest: VersionManifest
    ) -> SnapshotDiff:
        """Compare stored identity with current inputs; replace wholesale on drift."""
        current = self.snapshot_identity(manifest)
        stored = self.ledger.get_meta("session_identity")
        if stored is None:
            self.ledger.set_meta("session_identity", current)
            return SnapshotDiff(False, "fresh_session", {}, current, [])
        before_items = self._stored_item_index()
        same_version = stored.get("version") == current["version"]
        if same_version and (
            stored.get("manifest_digest") == current["manifest_digest"]
            and stored.get("scale") == current["scale"]
        ):
            return SnapshotDiff(False, "identity_match", stored, current, [])
        if not same_version:
            # Normal version evolution: preserve full version history and only
            # track the new identity; whole-state replacement is reserved for
            # drift inside the *same* manifest version.
            self.ledger.set_meta("session_identity", current)
            return SnapshotDiff(
                False,
                "version_evolution",
                stored,
                current,
                self._stable_item_diff(before_items, self._stored_item_index()),
            )
        self.ledger.replace_all_runtime_state()
        self.ledger.set_meta("session_identity", current)
        after_items = self._stored_item_index()
        changed = self._stable_item_diff(before_items, after_items)
        self.ledger.record_state(
            mono_ms(),
            self.state,
            "REPLACED",
            "snapshot_identity_mismatch",
            {"before": stored, "after": current, "changed_items": changed},
        )
        self.logger.warning(
            "snapshot_replaced",
            before_version=stored.get("version"),
            after_version=current["version"],
            changed=len(changed),
        )
        self.state = "REPLACED"
        return SnapshotDiff(
            True, "snapshot_identity_mismatch", stored, current, changed
        )

    def _stored_item_index(self) -> Dict[str, Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}
        for row in self.ledger.conn.execute(
            "SELECT version, items_json FROM plans ORDER BY version ASC"
        ).fetchall():
            import json

            for item in json.loads(row["items_json"]):
                index[item["item_id"]] = item
        return index

    @staticmethod
    def _stable_item_diff(
        before: Dict[str, Dict[str, Any]], after: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        diffs: List[Dict[str, Any]] = []
        for item_id in sorted(set(before) | set(after)):
            if item_id not in before:
                diffs.append({"item_id": item_id, "change": "added", "fields": []})
            elif item_id not in after:
                diffs.append({"item_id": item_id, "change": "removed", "fields": []})
            elif canonical_json(before[item_id]) != canonical_json(after[item_id]):
                fields = sorted(
                    key
                    for key in set(before[item_id]) | set(after[item_id])
                    if before[item_id].get(key) != after[item_id].get(key)
                )
                diffs.append(
                    {"item_id": item_id, "change": "modified", "fields": fields}
                )
        return diffs

    def compile_and_persist(
        self, manifest: VersionManifest
    ) -> Tuple[Dict[str, CompiledPath], List[PlanItem], str]:
        compiled, items = self.compiler.compile_version(manifest)
        error_rows = [dict(compiled[name].error) for name in sorted(compiled)]
        identity = self.snapshot_identity(manifest)
        plan_digest = self.ledger.save_plan(
            version=manifest.version,
            manifest_digest=identity["manifest_digest"],
            quant_version=self.compiler.quant,
            scale=self.scale,
            content_seed=self.content_seed,
            items=items,
            error_rows=error_rows,
        )
        self.ledger.set_meta("session_identity", identity)
        self.compiled = compiled
        self.plan_items = items
        return compiled, items, plan_digest
