"""Explicit revision chain, rollback and receipt arbitration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .canonical import canonical_json, finite_number, fingerprint
from .codes import (
    E_CURRENT_REVISION_UNKNOWN,
    E_NON_FINITE_WEIGHT,
    E_RECEIPT_DISCONNECTED_REVISION,
    E_RECEIPT_FUTURE_REVISION,
    E_RECEIPT_PROFILE_MISMATCH,
    E_RECEIPT_STALE_COMPATIBLE,
    E_RECEIPT_STALE_INCOMPATIBLE,
    E_REVISION_CYCLE,
    E_REVISION_NON_MONOTONIC,
    E_REVISION_PARENT_INVALID,
    E_REVISION_UNKNOWN_PARENT,
    E_INVALID_ENCODING,
    E_INVALID_REVISION,
    Issue,
)
from .models import CompressionProfile, RevisionNode


def _as_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _profile_from_raw(
    raw: dict[str, Any],
    inherited: CompressionProfile | None,
    issue_version: int | None,
    issues: list[Issue],
) -> CompressionProfile | None:
    has_count = "bucket_count" in raw
    has_weights = "bucket_weights" in raw
    has_encoding = "encoding" in raw
    changes: dict[str, Any] = {}
    if inherited is None:
        weights = []
        encoding = ""
        bucket_count = raw.get("bucket_count", 0)
        weights_raw: Any = raw.get("bucket_weights", [])
        encoding = raw.get("encoding", "")
    else:
        weights = list(inherited.bucket_weights)
        encoding = inherited.encoding
        bucket_count = len(weights)
        weights_raw = weights
    if has_count or has_weights or inherited is None:
        bucket_count = raw.get("bucket_count", bucket_count)
        weights_raw = raw.get("bucket_weights", weights_raw)
        if not _as_positive_int(bucket_count):
            issues.append(
                Issue(
                    code="E_BUCKET_COUNT_INVALID",
                    message="bucket_count must be a positive integer",
                    object_type="revision",
                    object_id=str(issue_version),
                    field="bucket_count",
                )
            )
            return None
        if not isinstance(weights_raw, list) or len(weights_raw) != bucket_count:
            issues.append(
                Issue(
                    code="E_BUCKET_WEIGHT_COUNT_MISMATCH",
                    message="bucket_weights length must equal bucket_count",
                    object_type="revision",
                    object_id=str(issue_version),
                    field="bucket_weights",
                )
            )
            return None
        weights = []
        for index, weight in enumerate(weights_raw):
            if not finite_number(weight):
                issues.append(
                    Issue(
                        code=E_NON_FINITE_WEIGHT,
                        message="bucket weight must be finite",
                        object_type="revision",
                        object_id=str(issue_version),
                        field=f"bucket_weights[{index}]",
                    )
                )
                return None
            if float(weight) < 0:
                issues.append(
                    Issue(
                        code="E_INVALID_WEIGHT",
                        message="bucket weight must be non-negative",
                        object_type="revision",
                        object_id=str(issue_version),
                        field=f"bucket_weights[{index}]",
                    )
                )
                return None
            weights.append(float(weight))
        if not isinstance(encoding, str) or not encoding.strip():
            issues.append(
                Issue(
                    code=E_INVALID_ENCODING,
                    message="encoding must be a non-empty string",
                    object_type="revision",
                    object_id=str(issue_version),
                    field="encoding",
                )
            )
            return None
        changes["explicit_profile"] = True
    if has_encoding:
        encoding = raw.get("encoding")
    if not isinstance(encoding, str) or not encoding.strip():
        issues.append(
            Issue(
                code=E_INVALID_ENCODING,
                message="encoding must be a non-empty string",
                object_type="revision",
                object_id=str(issue_version),
                field="encoding",
            )
        )
        return None

    updates = raw.get("weight_updates", {})
    if updates:
        if not isinstance(updates, dict):
            issues.append(
                Issue(
                    code="E_INVALID_REVISION",
                    message="weight_updates must be an object",
                    object_type="revision",
                    object_id=str(issue_version),
                    field="weight_updates",
                )
            )
            return None
        normalized_updates: dict[str, float] = {}
        for key, value in updates.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                issues.append(
                    Issue(
                        code="E_INVALID_REVISION",
                        message="weight_updates keys must be bucket indices",
                        object_type="revision",
                        object_id=str(issue_version),
                        field="weight_updates",
                    )
                )
                return None
            if index < 0 or index >= len(weights) or not finite_number(value):
                issues.append(
                    Issue(
                        code=E_NON_FINITE_WEIGHT if not finite_number(value) else "E_INVALID_WEIGHT",
                        message="weight update is outside the finite bucket domain",
                        object_type="revision",
                        object_id=str(issue_version),
                        field=f"weight_updates[{index}]",
                    )
                )
                return None
            weights[index] = float(value)
            normalized_updates[str(index)] = float(value)
        changes["weight_updates"] = normalized_updates

    removals = raw.get("remove_buckets", [])
    if removals:
        if not isinstance(removals, list) or any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0 or index >= len(weights)
            for index in removals
        ):
            issues.append(
                Issue(
                    code="E_INVALID_REVISION",
                    message="remove_buckets must contain valid bucket indices",
                    object_type="revision",
                    object_id=str(issue_version),
                    field="remove_buckets",
                )
            )
            return None
        removed_indices = sorted(set(removals))
        for index in removed_indices:
            weights[index] = -1.0
        changes["remove_buckets"] = removed_indices

    additions = raw.get("add_buckets", [])
    if additions:
        if not isinstance(additions, list):
            issues.append(
                Issue(
                    code="E_INVALID_REVISION",
                    message="add_buckets must be a list",
                    object_type="revision",
                    object_id=str(issue_version),
                    field="add_buckets",
                )
            )
            return None
        added: list[dict[str, Any]] = []
        for item in additions:
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                weight = float(item)
            elif isinstance(item, dict):
                weight = item.get("weight", 1.0)
            else:
                weight = None
            if weight is None or not finite_number(weight) or float(weight) < 0:
                issues.append(
                    Issue(
                        code=E_NON_FINITE_WEIGHT,
                        message="added bucket weight must be finite and non-negative",
                        object_type="revision",
                        object_id=str(issue_version),
                        field="add_buckets",
                    )
                )
                return None
            weights.append(float(weight))
            added.append({"bucket_index": len(weights) - 1, "weight": float(weight)})
        changes["add_buckets"] = added

    weights = [weight for weight in weights if weight >= 0]
    if not weights:
        issues.append(
            Issue(
                code="E_BUCKET_COUNT_INVALID",
                message="revision must leave at least one bucket",
                object_type="revision",
                object_id=str(issue_version),
                field="bucket_count",
            )
        )
        return None
    changes["bucket_count_before"] = inherited.bucket_count if inherited is not None else 0
    changes["bucket_count_after"] = len(weights)
    return CompressionProfile(bucket_count=len(weights), bucket_weights=tuple(weights), encoding=encoding)


def build_revision_chain(
    base_profile: CompressionProfile,
    raw_revisions: Any,
    current_revision: Any,
) -> tuple[tuple[RevisionNode, ...], int, list[Issue]]:
    """Resolve a parent-linked revision list.

    A revision may replace the whole compression profile, or inherit it and
    apply ``weight_updates``, ``remove_buckets`` and ``add_buckets``.
    Rollback is represented by pointing ``current_revision`` at any ancestor.
    """

    issues: list[Issue] = []
    if raw_revisions is None:
        node = RevisionNode(
            version=int(current_revision) if _as_positive_int(current_revision) else 1,
            parent=None,
            profile=base_profile,
            changes={"source": "standalone_compression"},
        )
        return (node,), node.version, issues
    if not isinstance(raw_revisions, list):
        issues.append(
            Issue(
                code="E_INVALID_REVISION",
                message="revisions must be a list",
                object_type="revision_chain",
                object_id="revisions",
            )
        )
        return (
            (RevisionNode(version=1, parent=None, profile=base_profile, changes={"source": "fallback"}),),
            1,
            issues,
        )

    nodes: list[RevisionNode] = []
    seen: set[int] = set()
    resolved: dict[int, RevisionNode] = {}
    previous_version: int | None = None
    for position, raw in enumerate(raw_revisions):
        if not isinstance(raw, dict):
            issues.append(
                Issue(
                    code="E_INVALID_REVISION",
                    message="revision must be an object",
                    object_type="revision",
                    object_id=str(position),
                )
            )
            continue
        version = raw.get("version")
        if not _as_positive_int(version):
            issues.append(
                Issue(
                    code=E_INVALID_REVISION,
                    message="revision version must be a positive integer",
                    object_type="revision",
                    object_id=str(position),
                    field="version",
                )
            )
            continue
        if version in seen:
            issues.append(
                Issue(
                    code=E_INVALID_REVISION,
                    message="revision version must be unique",
                    object_type="revision",
                    object_id=str(version),
                    field="version",
                )
            )
            continue
        if previous_version is not None and version < previous_version:
            issues.append(
                Issue(
                    code=E_REVISION_NON_MONOTONIC,
                    message="revision versions must be monotonically non-decreasing",
                    object_type="revision",
                    object_id=str(version),
                    field="version",
                )
            )
            continue
        previous_version = version
        parent = raw.get("parent")
        if parent is not None:
            if not isinstance(parent, int) or isinstance(parent, bool) or parent <= 0:
                issues.append(
                    Issue(
                        code=E_REVISION_PARENT_INVALID,
                        message="revision parent must be a positive integer or null",
                        object_type="revision",
                        object_id=str(version),
                        field="parent",
                    )
                )
                continue
            if parent not in resolved:
                issues.append(
                    Issue(
                        code=E_REVISION_UNKNOWN_PARENT,
                        message="revision parent must be an earlier revision",
                        object_type="revision",
                        object_id=str(version),
                        field="parent",
                    )
                )
                continue
            inherited = resolved[parent].profile
        else:
            inherited = None
        profile = _profile_from_raw(raw, inherited, version, issues)
        if profile is None:
            continue
        changes = dict(raw.get("changes", {})) if isinstance(raw.get("changes"), dict) else {}
        if "encoding" in raw:
            changes.setdefault("encoding", raw["encoding"])
        node = RevisionNode(
            version=version,
            parent=parent,
            profile=profile,
            changes=changes,
            encoding=raw.get("encoding") if isinstance(raw.get("encoding"), str) else None,
        )
        nodes.append(node)
        resolved[version] = node
        seen.add(version)

    if not nodes:
        issues.append(
            Issue(
                code=E_INVALID_REVISION,
                message="no valid revisions remained; using compression fallback",
                object_type="revision_chain",
                object_id="revisions",
            )
        )
        fallback_version = int(current_revision) if _as_positive_int(current_revision) else 1
        node = RevisionNode(version=fallback_version, parent=None, profile=base_profile, changes={"source": "fallback"})
        return (node,), fallback_version, issues

    # Detect cycles/disconnected entries defensively; parent checks above reject
    # cycles already, but explicit walk makes the invariant auditable.
    for node in nodes:
        cursor: RevisionNode | None = node
        visited: set[int] = set()
        while cursor is not None:
            if cursor.version in visited:
                issues.append(
                    Issue(
                        code=E_REVISION_CYCLE,
                        message="revision parent chain contains a cycle",
                        object_type="revision",
                        object_id=str(node.version),
                    )
                )
                break
            visited.add(cursor.version)
            cursor = resolved.get(cursor.parent) if cursor.parent is not None else None

    requested_current = int(current_revision) if _as_positive_int(current_revision) else None
    if requested_current is None or requested_current not in resolved:
        issues.append(
            Issue(
                code=E_CURRENT_REVISION_UNKNOWN,
                message="current_revision must name a revision in the chain",
                object_type="revision_chain",
                object_id=str(current_revision),
                field="current_revision",
                details={"available_revisions": [node.version for node in nodes]},
            )
        )
        current = max(node.version for node in nodes)
    else:
        current = requested_current
    return tuple(nodes), current, issues


@dataclass(frozen=True)
class ArbitrationResult:
    served_revision: int
    current_revision: int
    decision: str
    code: str
    basis: tuple[str, ...]
    covered_by: tuple[int, ...]
    compatible: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "served_revision": self.served_revision,
            "current_revision": self.current_revision,
            "decision": self.decision,
            "code": self.code,
            "basis": list(self.basis),
            "covered_by": list(self.covered_by),
            "compatible": self.compatible,
        }


class RevisionChain:
    def __init__(self, nodes: Iterable[RevisionNode], current_revision: int):
        self.nodes = tuple(sorted(nodes, key=lambda node: node.version))
        self.by_version = {node.version: node for node in self.nodes}
        self.current_revision = current_revision

    @property
    def current_profile(self) -> CompressionProfile:
        return self.profile(self.current_revision)

    def profile(self, version: int) -> CompressionProfile:
        return self.by_version[version].profile

    def profile_fingerprint(self, version: int) -> str:
        return fingerprint(self.profile(version).to_dict())

    def ancestry(self, version: int) -> tuple[int, ...]:
        if version not in self.by_version:
            return ()
        path: list[int] = []
        cursor: int | None = version
        while cursor is not None:
            path.append(cursor)
            cursor = self.by_version[cursor].parent
        return tuple(reversed(path))

    def path_between(self, ancestor: int, descendant: int) -> tuple[int, ...]:
        if ancestor not in self.by_version or descendant not in self.by_version:
            return ()
        ancestor_path = self.ancestry(ancestor)
        descendant_path = self.ancestry(descendant)
        if not ancestor_path or not descendant_path:
            return ()
        if ancestor_path[-1] != ancestor:
            return ()
        if ancestor_path == descendant_path[: len(ancestor_path)]:
            return descendant_path[len(ancestor_path) :]
        return ()

    @staticmethod
    def profile_diff(old: CompressionProfile, new: CompressionProfile) -> dict[str, Any]:
        changed: list[Any] = []
        if old.bucket_count != new.bucket_count:
            changed.append("bucket_count")
        if old.encoding != new.encoding:
            changed.append("encoding")
        common = min(old.bucket_count, new.bucket_count)
        changed_weights = [
            index
            for index in range(common)
            if float(old.bucket_weights[index]) != float(new.bucket_weights[index])
        ]
        if changed_weights:
            changed.append({"bucket_weights": changed_weights})
        return {
            "changed": changed,
            "old_fingerprint": fingerprint(old.to_dict()),
            "new_fingerprint": fingerprint(new.to_dict()),
        }

    def compatibility(self, served: int, current: int, bucket_indices: Iterable[int]) -> tuple[bool, list[str], list[int]]:
        """Decide whether a stale receipt remains valid under the current profile.

        A receipt is directly reusable when its revision is on the current
        ancestor path and the compression domains used by the receipt are
        byte-identical after the chain transition.  Any encoding, bucket-count
        or referenced-bucket weight change is a hard incompatibility.
        """

        if served == current:
            return True, ["served_revision_equals_current"], []
        path = self.path_between(served, current)
        if not path:
            served_ancestry = self.ancestry(served)
            current_ancestry = self.ancestry(current)
            if served in current_ancestry and served != current:
                return False, ["served_revision_is_ancestor"], list(path)
            if current in served_ancestry:
                return False, ["receipt_is_from_a_future_revision"], list(served_ancestry[len(current_ancestry) :])
            return False, ["revisions_are_disconnected"], sorted(set(served_ancestry) | set(current_ancestry))
        diff = self.profile_diff(self.profile(served), self.profile(current))
        buckets = sorted(set(int(index) for index in bucket_indices))
        if not diff["changed"]:
            return True, ["compression_domain_unchanged"], list(path)
        encoding_changed = "encoding" in diff["changed"]
        count_changed = "bucket_count" in diff["changed"]
        changed_weights: set[int] = set()
        for item in diff["changed"]:
            if isinstance(item, dict):
                changed_weights.update(int(index) for index in item["bucket_weights"])
        intersects = bool(set(buckets) & changed_weights)
        if encoding_changed or count_changed or intersects:
            reasons = ["revision_path_changed_compression_domain"]
            if encoding_changed:
                reasons.append("encoding_changed")
            if count_changed:
                reasons.append("bucket_count_changed")
            if intersects:
                reasons.append("referenced_bucket_weight_changed")
            return False, reasons, list(path)
        return True, ["revision_path_preserved_referenced_buckets"], list(path)

    def arbitrate(
        self,
        served_revision: int,
        bucket_indices: Iterable[int],
        *,
        receipt_profile: CompressionProfile | None = None,
    ) -> ArbitrationResult:
        current = self.current_revision
        buckets = sorted(set(int(index) for index in bucket_indices))
        if served_revision == current:
            chain_basis = ["served_revision_equals_current"]
            if receipt_profile is not None:
                expected = self.profile(current)
                if fingerprint(expected.to_dict()) != fingerprint(receipt_profile.to_dict()):
                    return ArbitrationResult(
                        served_revision=served_revision,
                        current_revision=current,
                        decision="rejected",
                        code=E_RECEIPT_PROFILE_MISMATCH,
                        basis=("receipt_compression_does_not_match_current_revision",),
                        covered_by=(current,),
                        compatible=False,
                    )
            return ArbitrationResult(
                served_revision=served_revision,
                current_revision=current,
                decision="accepted",
                code="OK",
                basis=tuple(chain_basis),
                covered_by=(),
                compatible=True,
            )

        served_ancestry = self.ancestry(served_revision)
        current_ancestry = self.ancestry(current)
        if served_revision not in self.by_version:
            return ArbitrationResult(
                served_revision=served_revision,
                current_revision=current,
                decision="rejected",
                code=E_RECEIPT_DISCONNECTED_REVISION,
                basis=("served_revision_not_in_chain",),
                covered_by=tuple(current_ancestry),
                compatible=False,
            )
        if current in served_ancestry:
            return ArbitrationResult(
                served_revision=served_revision,
                current_revision=current,
                decision="rejected",
                code=E_RECEIPT_FUTURE_REVISION,
                basis=("receipt_revision_is_newer_than_effective_revision",),
                covered_by=tuple(served_ancestry[len(current_ancestry) :]),
                compatible=False,
            )
        if served_revision not in current_ancestry:
            return ArbitrationResult(
                served_revision=served_revision,
                current_revision=current,
                decision="rejected",
                code=E_RECEIPT_DISCONNECTED_REVISION,
                basis=("revisions_are_disconnected",),
                covered_by=tuple(current_ancestry),
                compatible=False,
            )

        covered_by = self.path_between(served_revision, current)
        expected_served = self.profile(served_revision)
        if receipt_profile is not None and fingerprint(expected_served.to_dict()) != fingerprint(receipt_profile.to_dict()):
            return ArbitrationResult(
                served_revision=served_revision,
                current_revision=current,
                decision="rejected",
                code=E_RECEIPT_PROFILE_MISMATCH,
                basis=("served_revision_profile_does_not_match_receipt",),
                covered_by=covered_by,
                compatible=False,
            )
        compatible, basis, path = self.compatibility(served_revision, current, buckets)
        if compatible:
            return ArbitrationResult(
                served_revision=served_revision,
                current_revision=current,
                decision="accepted_compatible",
                code=E_RECEIPT_STALE_COMPATIBLE,
                basis=tuple(basis + [f"covered_by_revision_{version}" for version in covered_by]),
                covered_by=covered_by,
                compatible=True,
            )
        if "receipt_is_from_a_future_revision" in basis:
            code = E_RECEIPT_FUTURE_REVISION
        elif "revisions_are_disconnected" in basis:
            code = E_RECEIPT_DISCONNECTED_REVISION
        else:
            code = E_RECEIPT_STALE_INCOMPATIBLE
        return ArbitrationResult(
            served_revision=served_revision,
            current_revision=current,
            decision="rejected",
            code=code,
            basis=tuple(basis + [f"covered_by_revision_{version}" for version in path]),
            covered_by=tuple(path),
            compatible=False,
        )


__all__ = ["ArbitrationResult", "RevisionChain", "build_revision_chain", "canonical_json"]
