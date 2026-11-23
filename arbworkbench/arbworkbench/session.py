"""Session orchestration: arbitration runs, whole-view replacement, snapshots."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .canonical import canonical_json, content_digest, fingerprint_environments
from .engine import ArbitrationEngine, KeyConclusion, ObservationVerdict, ViewConclusion
from .errors import ReplayError, VersionError
from .ledger import Ledger
from .spec import Specification
from .versioning import VersionRegistry, diff_conclusions

SNAPSHOT_FORMAT_VERSION = 1
SCALE_SCHEME_VERSION = 1

# Fields contractually compared during replay, in first-difference order.
_REPLAY_FIELDS = (
    "authority_q",
    "bound_q",
    "adjacent_diff_q",
    "error_influenced",
    "attribution",
    "attribution_contributions",
    "accepted_obs_ids",
    "rejected",
    "conflict_detail",
)


def key_conclusion_digest(kc: KeyConclusion) -> str:
    return content_digest(asdict(kc))


class Session:
    def __init__(self, spec: Specification, ledger: Ledger,
                 registry: VersionRegistry | None = None):
        self.spec = spec
        self.ledger = ledger
        self.registry = registry or VersionRegistry()
        self.registry.reconcile(spec.views)
        self.engine = ArbitrationEngine(spec)
        self.conclusions: dict[str, ViewConclusion] = {}
        self._all_committed: dict[str, dict] = {}
        self._load_ledger_state()

    # ------------------------------------------------------------------
    def _load_ledger_state(self) -> None:
        committed = self.ledger.load_committed()
        self._all_committed = dict(committed)
        for view_id, record in committed.items():
            if view_id not in self.spec.views:
                continue
            expected_version = self.registry.version_of(view_id)
            if record["version"] != expected_version:
                # Stale conclusion from an older view version: excluded from display.
                continue
            self.conclusions[view_id] = self._hydrate(view_id, record["payload"])

    def _previous_key_digests(self, view_id: str, current_version: int) -> dict[str, str]:
        """Most recent committed digests strictly below the current version."""
        record = self._all_committed.get(view_id)
        if record is None or record["version"] >= current_version:
            return {}
        previous = self._hydrate(view_id, record["payload"])
        return {key: key_conclusion_digest(kc) for key, kc in previous.keys.items()}

    def _hydrate(self, view_id: str, payload: dict) -> ViewConclusion:
        keys: dict[str, KeyConclusion] = {}
        for key, row in payload["keys"].items():
            row["rejected"] = [ObservationVerdict(**r) for r in row["rejected"]]
            row["accepted_verdicts"] = [
                ObservationVerdict(**r) for r in row["accepted_verdicts"]
            ]
            keys[key] = KeyConclusion(**row)
        stable_key_digests = {
            key: key_conclusion_digest(kc) for key, kc in sorted(keys.items())
        }
        return ViewConclusion(
            view_id=view_id,
            version=payload["version"],
            keys=keys,
            conclusion_digest=content_digest(stable_key_digests),
            added_keys=payload.get("added_keys", []),
            removed_keys=payload.get("removed_keys", []),
            changed_keys=payload.get("changed_keys", []),
        )

    # ------------------------------------------------------------------
    def run_view(self, view_id: str, crash_after_insert: bool = False) -> ViewConclusion:
        """Arbitrate one view and replace its conclusion atomically."""
        view = self.spec.views[view_id]
        version = self.registry.version_of(view_id)
        old = self.conclusions.get(view_id)
        if old and old.version == version:
            old_digests = {k: key_conclusion_digest(kc) for k, kc in old.keys.items()}
        else:
            # Whole-view replacement after an edit: diff against the prior version.
            old_digests = self._previous_key_digests(view_id, version)
        keys: dict[str, KeyConclusion] = {}
        for key in sorted(view.policies):
            keys[key] = self.engine.arbitrate_key(view, key)
        new_digests = {k: key_conclusion_digest(kc) for k, kc in keys.items()}
        diff = diff_conclusions(old_digests, new_digests)
        conclusion = ViewConclusion(
            view_id=view_id,
            version=version,
            keys=keys,
            conclusion_digest=content_digest(dict(sorted(new_digests.items()))),
            added_keys=diff.added_keys,
            removed_keys=diff.removed_keys,
            changed_keys=diff.changed_keys,
        )
        if crash_after_insert:
            self._simulate_kill(conclusion)
        self.ledger.write_view(conclusion)
        self.conclusions[view_id] = conclusion
        return conclusion

    def _simulate_kill(self, conclusion: ViewConclusion) -> None:
        conn = self.ledger.conn
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO view_conclusions(view_id, version, conclusion_digest, "
            "payload, created_order) VALUES(?, ?, ?, ?, ?)",
            (conclusion.view_id, conclusion.version, conclusion.conclusion_digest,
             canonical_json({"view_id": conclusion.view_id}).decode("utf-8"),
             999999),
        )
        # No commit and no committed marker: recovery treats this as unknown.
        conn.close()
        raise RuntimeError("E_INJECTED_CRASH: process terminated before commit")

    def run_all(self, crash_view_id: str | None = None) -> dict[str, ViewConclusion]:
        results: dict[str, ViewConclusion] = {}
        for view_id in sorted(self.spec.views):
            results[view_id] = self.run_view(
                view_id, crash_after_insert=(crash_view_id == view_id)
            )
        return results

    # ------------------------------------------------------------------
    def unknown_views(self) -> list[str]:
        return self.ledger.unknown_views(set(self.spec.views))

    def rerun_unknown(self) -> dict[str, ViewConclusion]:
        return {view_id: self.run_view(view_id) for view_id in self.unknown_views()}

    # ------------------------------------------------------------------
    def build_snapshot(self) -> dict:
        per_view = {}
        for view_id in sorted(self.conclusions):
            vc = self.conclusions[view_id]
            per_view[view_id] = {
                "version": vc.version,
                "conclusion_digest": vc.conclusion_digest,
                "key_digests": {
                    key: key_conclusion_digest(kc) for key, kc in sorted(vc.keys.items())
                },
                "keys": {
                    key: asdict(kc) for key, kc in sorted(vc.keys.items())
                },
            }
        snapshot = {
            "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
            "scale_scheme_version": SCALE_SCHEME_VERSION,
            "scale_exponent": self.spec.scale_exponent,
            "environment_version": fingerprint_environments(self.spec),
            "views": self.registry.snapshot(),
            "completed_views": sorted(self.conclusions),
            "view_conclusions": per_view,
        }
        snapshot["content_digest"] = content_digest(
            {k: v for k, v in snapshot.items() if k != "content_digest"}
        )
        return snapshot

    def save_snapshot(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.build_snapshot(), fh, indent=2, sort_keys=True)
        return path

    # ------------------------------------------------------------------
    def verify_snapshot(self, snapshot: dict, check_digest: bool = True) -> None:
        if check_digest:
            stored_digest = snapshot.get("content_digest")
            body = {k: v for k, v in snapshot.items() if k != "content_digest"}
            actual_digest = content_digest(body)
            if stored_digest != actual_digest:
                raise VersionError(
                    "snapshot content digest mismatch; whole-state replacement required",
                    location="snapshot.content_digest",
                    detail={"stored": stored_digest, "actual": actual_digest},
                )
        if snapshot.get("snapshot_format_version") != SNAPSHOT_FORMAT_VERSION:
            raise VersionError(
                "snapshot format version unsupported",
                location="snapshot.snapshot_format_version",
            )
        if snapshot.get("scale_scheme_version") != SCALE_SCHEME_VERSION:
            raise VersionError(
                "scale scheme version mismatch",
                location="snapshot.scale_scheme_version",
            )
        if snapshot.get("environment_version") != fingerprint_environments(self.spec):
            raise VersionError(
                "environment/observation set differs from snapshot; whole-state replacement required",
                location="snapshot.environment_version",
            )
        if snapshot.get("scale_exponent") != self.spec.scale_exponent:
            raise VersionError(
                "session scale exponent differs from snapshot",
                location="snapshot.scale_exponent",
            )
        for view_id, version_row in snapshot.get("views", {}).items():
            current = self.registry.snapshot().get(view_id)
            if (
                current is None
                or current["version"] != version_row["version"]
                or current["fingerprint"] != version_row["fingerprint"]
            ):
                raise VersionError(
                    f"view '{view_id}' version/fingerprint differs from snapshot",
                    location=f"snapshot.views[{view_id}]",
                )

    def replay_against_snapshot(self, snapshot: dict) -> None:
        """Re-arbitrate per snapshot and locate the first divergent field."""
        self.verify_snapshot(snapshot)
        stored_views = snapshot["view_conclusions"]
        for view_id in sorted(stored_views):
            fresh = self.run_view(view_id)
            stored = stored_views[view_id]
            if fresh.version != stored["version"]:
                raise ReplayError(
                    f"view '{view_id}' version diverged",
                    location=f"views[{view_id}].version",
                )
            for key in sorted(stored["keys"]):
                fresh_kc = fresh.keys.get(key)
                if fresh_kc is None:
                    raise ReplayError(
                        f"missing key '{key}' after replay",
                        location=f"views[{view_id}].keys[{key}]",
                    )
                stored_kc = self._stored_kc(stored["keys"][key])
                for field_name in _REPLAY_FIELDS:
                    if getattr(fresh_kc, field_name) != getattr(stored_kc, field_name):
                        raise ReplayError(
                            f"replay mismatch for view '{view_id}' key '{key}'",
                            location=f"views[{view_id}].keys[{key}].{field_name}",
                            detail={
                                "stored": _safe(getattr(stored_kc, field_name)),
                                "replayed": _safe(getattr(fresh_kc, field_name)),
                            },
                        )

    @staticmethod
    def _stored_kc(row: dict) -> KeyConclusion:
        row = dict(row)
        row["rejected"] = [ObservationVerdict(**r) for r in row["rejected"]]
        row["accepted_verdicts"] = [
            ObservationVerdict(**r) for r in row["accepted_verdicts"]
        ]
        return KeyConclusion(**row)


def _safe(value):
    if isinstance(value, list) and value and hasattr(value[0], "__dataclass_fields__"):
        return [asdict(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value
