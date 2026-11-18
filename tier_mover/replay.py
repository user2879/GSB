"""Deterministic replay validation against the persisted ledger."""

from __future__ import annotations

import json
from typing import Any, Dict

from .errors import ReplayError
from .ledger import Ledger
from .models import Manifest
from .planner import derive_cycle_plan, derive_initial_plan
from .storage import object_path, sha256_file


class Replayer:
    def __init__(self, workdir: str, manifest: Manifest) -> None:
        self.workdir = workdir
        self.manifest = manifest
        self.ledger = Ledger.open(workdir)

    def close(self) -> None:
        self.ledger.close()

    def _fail(self, key: str, tier_id: str, field: str, expected: Any, actual: Any, basis: str) -> None:
        raise ReplayError(
            f"replay mismatch at key={key} tier={tier_id} field={field}",
            {
                "first_mismatch_key": key,
                "tier_id": tier_id,
                "field": field,
                "expected": expected,
                "actual": actual,
                "basis": basis,
            },
        )

    def _check_versions_and_digests(self) -> None:
        versions = self.ledger.versions()
        if versions["policy_version"] != self.manifest.policy_version:
            raise ReplayError(
                "policy version changed; replay not reproducible",
                {
                    "field": "policy_version",
                    "ledger": versions["policy_version"],
                    "session": self.manifest.policy_version,
                    "basis": "snapshot policy version vs current session",
                },
            )
        if versions["tier_param_version"] != self.manifest.tier_param_version:
            raise ReplayError(
                "tier parameter version changed; replay not reproducible",
                {
                    "field": "tier_param_version",
                    "ledger": versions["tier_param_version"],
                    "session": self.manifest.tier_param_version,
                    "basis": "snapshot tier param version vs current session",
                },
            )
        # Object content digests must match what was originally declared.
        for row in self.ledger.object_rows():
            obj = self.manifest.object_by_key(row["key"])
            if obj.digest != row["digest"]:
                raise ReplayError(
                    f"object content digest changed for {row['key']}; replay not reproducible",
                    {
                        "first_mismatch_key": row["key"],
                        "field": "digest",
                        "ledger_digest": row["digest"],
                        "manifest_digest": obj.digest,
                        "basis": "object content digest changed since snapshot",
                    },
                )

    def validate(self) -> Dict[str, Any]:
        self._check_versions_and_digests()
        recorded = self.ledger.migrations()
        if not recorded:
            raise ReplayError(
                "no committed migrations to replay",
                {"field": "migration", "basis": "ledger contains no migration rows"},
            )

        phases = sorted({row["cycle_id"] for row in recorded})
        cycle_meta = {row["cycle_id"]: row for row in self.ledger.cycles()}
        ownership: Dict[str, str] = {}
        usage = {tier.tier_id: 0 for tier in self.manifest.tiers}
        counts = {tier.tier_id: 0 for tier in self.manifest.tiers}
        checked = 0

        for cycle_id in phases:
            cycle_rows = [row for row in recorded if row["cycle_id"] == cycle_id]
            phase = cycle_meta[cycle_id]["phase"]
            if phase == "initial":
                derived_actions = [a.to_dict() for a in derive_initial_plan(self.manifest)]
            elif phase == "recovery":
                # A recovery cycle starts from a crash-distorted ownership set
                # that cannot be reconstructed from committed rows alone; the
                # deterministic recovery plan was persisted in the cycle row.
                stored_plan = json.loads(cycle_meta[cycle_id]["plan_json"])
                derived_actions = stored_plan["actions"]
                # Rebuild the exact post-crash, reconciled baseline.
                usage = dict(stored_plan["baseline_usage"])
                counts = {
                    tier.tier_id: 0 for tier in self.manifest.tiers
                }
                ownership = dict(stored_plan["baseline_ownership"])
                size_by_key = {o.key: o.size_bytes for o in self.manifest.objects}
                for key, tier_id in ownership.items():
                    if tier_id is not None:
                        counts[tier_id] += 1
                assert sum(usage.values()) == sum(size_by_key.values())
            else:
                derived_actions = derive_cycle_plan(self.manifest, ownership)["actions"]

            if len(derived_actions) != len(cycle_rows):
                raise ReplayError(
                    "re-derived plan action count differs from ledger",
                    {
                        "field": "actions",
                        "cycle_id": cycle_id,
                        "expected": len(derived_actions),
                        "actual": len(cycle_rows),
                        "basis": "deterministic plan re-derivation vs recorded migrations",
                    },
                )

            for derived_action, row in zip(derived_actions, cycle_rows):
                key = row["key"]
                target_id = row["target_tier"]
                source_id = row["source_tier"]
                size_bytes = row["size_bytes"]

                # Plan-level fields must match the re-derived deterministic plan.
                for field in ("key", "action", "direction", "target_tier",
                              "source_tier", "reason", "digest", "size_bytes"):
                    if derived_action[field] != row[field]:
                        self._fail(
                            key, str(target_id), field,
                            derived_action[field], row[field],
                            "re-derived plan field vs recorded migration",
                        )

                if row["action"] == "place":
                    if row["dst_used_before"] != usage[target_id]:
                        self._fail(key, target_id, "dst_used_before", usage[target_id],
                                   row["dst_used_before"], "replayed occupancy vs ledger")
                    usage[target_id] += size_bytes
                    counts[target_id] += 1
                else:
                    if row["src_used_before"] != usage[source_id]:
                        self._fail(key, source_id, "src_used_before", usage[source_id],
                                   row["src_used_before"], "replayed occupancy vs ledger")
                    if row["dst_used_before"] != usage[target_id]:
                        self._fail(key, target_id, "dst_used_before", usage[target_id],
                                   row["dst_used_before"], "replayed occupancy vs ledger")
                    usage[source_id] -= size_bytes
                    usage[target_id] += size_bytes
                    counts[source_id] -= 1
                    counts[target_id] += 1
                ownership[key] = target_id

                if source_id is None:
                    if row["src_used_after"] != 0:
                        self._fail(key, target_id, "src_used_after", 0,
                                   row["src_used_after"], "placement reports zero source usage")
                elif row["src_used_after"] != usage[source_id]:
                    self._fail(key, source_id, "src_used_after", usage[source_id],
                               row["src_used_after"], "replayed occupancy vs ledger")
                if row["dst_used_after"] != usage[target_id]:
                    self._fail(key, target_id, "dst_used_after", usage[target_id],
                               row["dst_used_after"], "replayed occupancy vs ledger")
                checked += 1

            # Note: physical files only retain their *final* location, so file
            # presence/digest is validated once at the end below.  At each
            # boundary we only assert the numeric ledger transitions.

        # Final ownership and per-tier occupancy must equal the ledger state.
        ledger_ownership = self.ledger.ownership()
        for key in sorted(set(ownership) | set(ledger_ownership)):
            if ownership.get(key) != ledger_ownership.get(key):
                self._fail(
                    key, str(ledger_ownership.get(key)), "tier_id",
                    ownership.get(key), ledger_ownership.get(key),
                    "replayed ownership vs ledger object_state",
                )
        for row in self.ledger.tier_rows():
            tier_id = row["tier_id"]
            if usage[tier_id] != row["used_bytes"]:
                self._fail("__tier__", tier_id, "used_bytes", usage[tier_id],
                           row["used_bytes"], "replayed occupancy vs ledger tier_state")
            if counts[tier_id] != row["object_count"]:
                self._fail("__tier__", tier_id, "object_count", counts[tier_id],
                           row["object_count"], "replayed counts vs ledger tier_state")
            if row["free_bytes"] != row["capacity_bytes"] - row["used_bytes"]:
                self._fail("__tier__", tier_id, "free_bytes",
                           row["capacity_bytes"] - row["used_bytes"], row["free_bytes"],
                           "free = capacity - used invariant")

        # Final physical verification: every object file exists on the tier the
        # replay converged to and its sha256 matches the recorded digest.
        for key in sorted(ownership):
            tier_id = ownership[key]
            obj = self.manifest.object_by_key(key)
            path = object_path(self.workdir, tier_id, obj.digest)
            if not path.exists():
                self._fail(key, tier_id, "file_presence", "exists", "missing",
                           "final tier directory scan")
            actual_digest = sha256_file(path)
            if actual_digest != obj.digest:
                self._fail(key, tier_id, "digest", obj.digest, actual_digest,
                           "sha256 of final physical object file")

        return {
            "status": "ok",
            "policy_version": self.manifest.policy_version,
            "tier_param_version": self.manifest.tier_param_version,
            "migrations_replayed": checked,
            "cycles_replayed": len(phases),
            "final_usage": usage,
            "final_object_counts": counts,
        }



def replay(workdir: str, manifest: Manifest) -> Dict[str, Any]:
    replayer = Replayer(workdir, manifest)
    try:
        return replayer.validate()
    finally:
        replayer.close()
