"""Deterministic backup-integrity and recovery engine."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .canonical import (
    DIGEST_PREFIX_LENGTH,
    SUPPORTED_ALGORITHMS,
    canonical_json_bytes,
    compute_content_digest,
    compute_snapshot_fingerprint,
    decode_content,
    digest_prefix,
    encode_content,
    normalize_algorithm,
    normalize_digest,
    normalize_hex_fingerprint,
    normalize_relative_path,
    path_sort_key,
    sha256_hex,
    strict_json_loads,
)
from .storage import (
    LockConflict,
    WorkspaceLock,
    atomic_write_json,
    cleanup_orphan_temps,
    file_sha256,
)


ENGINE_VERSION = "1.0"
CHECKPOINT_SCHEMA_VERSION = 1
RESTORE_SCHEMA_VERSION = 1
SNAPSHOT_SCHEMA_VERSION = 1

EXCLUSION_CODES = {
    "E_ALGORITHM",
    "E_DIGEST_FORMAT",
    "E_LENGTH",
    "E_TAMPER",
}
STRUCTURAL_CODES = {
    "E_ALGORITHM",
    "E_CHAIN_BREAK",
    "E_CYCLE",
    "E_FINGERPRINT",
    "E_FORK",
    "E_JSON",
    "E_ROLLBACK",
    "E_VERSION",
}


def _normalize_algorithms(values: Iterable[str] | None) -> tuple[str, ...]:
    if values is None:
        return tuple(sorted(SUPPORTED_ALGORITHMS))
    if isinstance(values, str):
        values = [item.strip() for item in values.split(",") if item.strip()]
    normalized: list[str] = []
    for value in values:
        algorithm = normalize_algorithm(value)
        if algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(f"unsupported algorithm: {value!r}")
        if algorithm not in normalized:
            normalized.append(algorithm)
    if not normalized:
        raise ValueError("at least one algorithm is required")
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class Policy:
    algorithms: tuple[str, ...]
    hmac_key: bytes | None = field(default=None, repr=False, compare=False)

    @property
    def hmac_key_id(self) -> str | None:
        return sha256_hex(self.hmac_key) if self.hmac_key is not None else None

    def to_json(self) -> dict[str, Any]:
        return {
            "algorithms": list(self.algorithms),
            "hmac_key_id": self.hmac_key_id,
        }


@dataclass
class ParsedEntry:
    index: int
    path: str | None
    accepted: bool
    deletion: bool
    exclusion_candidate: bool
    findings: list[dict[str, Any]]
    content_b64: str | None = None
    digest: str | None = None
    length: int | None = None


@dataclass
class Snapshot:
    raw: dict[str, Any]
    version: int
    algorithm: str
    previous_fingerprint: str | None
    previous_version: int | None
    fingerprint: str
    entry_results: list[ParsedEntry]
    top_findings: list[dict[str, Any]]
    previous_link_valid: bool = True

    @property
    def fingerprint_prefix(self) -> str:
        return digest_prefix(self.fingerprint)

    def meta(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "fingerprint": self.fingerprint,
            "fingerprint_prefix": self.fingerprint_prefix,
            "version": self.version,
        }

    def sort_key(self) -> tuple[int, str]:
        return (self.version, self.fingerprint)


def _finding(
    code: str,
    message: str,
    *,
    version: Any = None,
    entry_index: int | None = None,
    path: str | None = None,
    fingerprint: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {"code": code, "message": message}
    if version is not None:
        result["version"] = version
    if entry_index is not None:
        result["entry_index"] = entry_index
        result["entry_number"] = entry_index + 1
    if path is not None:
        result["path"] = path
    if fingerprint is not None:
        result["snapshot_fingerprint"] = fingerprint
        result["snapshot_fingerprint_prefix"] = digest_prefix(fingerprint)
    for key, value in extra.items():
        if value is not None:
            result[key] = value
    return result


def _finding_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    version = item.get("version")
    version_key = version if isinstance(version, int) else -1
    entry_index = item.get("entry_index")
    entry_key = entry_index if isinstance(entry_index, int) else -1
    return (
        version_key,
        entry_key,
        str(item.get("code", "")),
        str(item.get("path", "")),
        str(item.get("snapshot_fingerprint", "")),
        str(item.get("message", "")),
    )


def _sorted_findings(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=_finding_sort_key)


class BackupEngine:
    """Offline deterministic backup recovery engine.

    Public methods return JSON-serializable reports and never raise for expected
    business failures. Parameter/programming errors still raise ValueError.
    """

    def __init__(
        self,
        workspace: str | os.PathLike[str],
        *,
        hmac_key: bytes | str | None = None,
        algorithms: Iterable[str] | None = None,
    ):
        self.workspace = Path(workspace).expanduser().resolve()
        if hmac_key is None:
            environment_key = os.environ.get("BACKUP_ENGINE_HMAC_KEY")
            if environment_key is not None:
                hmac_key = environment_key.encode("utf-8")
        if isinstance(hmac_key, str):
            hmac_key = hmac_key.encode("utf-8")
        self.policy = Policy(_normalize_algorithms(algorithms), hmac_key)
        self.snapshots_dir = self.workspace / "snapshots"
        self.restore_dir = self.workspace / "restore"
        self.checkpoint_path = self.workspace / "checkpoint.json"
        self.lock_path = self.workspace / ".backup-engine.lock"

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_snapshot_object(
        self, raw: Any, *, include_source: str | None = None
    ) -> tuple[Snapshot | None, list[dict[str, Any]]]:
        findings: list[dict[str, Any]] = []
        source_extra = {"source": include_source} if include_source else {}
        if not isinstance(raw, dict):
            return None, [
                _finding("E_JSON", "snapshot root must be a JSON object", **source_extra)
            ]

        raw_version = raw.get("version")
        version: int | None
        if isinstance(raw_version, int) and not isinstance(raw_version, bool) and raw_version > 0:
            version = raw_version
        else:
            version = None
            findings.append(
                _finding(
                    "E_VERSION",
                    "snapshot version must be a positive integer",
                    **source_extra,
                )
            )

        schema_version = raw.get("schema_version", SNAPSHOT_SCHEMA_VERSION)
        if (
            not isinstance(schema_version, int)
            or isinstance(schema_version, bool)
            or schema_version < 1
        ):
            findings.append(
                _finding(
                    "E_VERSION",
                    "snapshot schema_version must be a positive integer",
                    version=version,
                    **source_extra,
                )
            )

        algorithm_value = raw.get("algorithm")
        if algorithm_value is None:
            for alias in ("algo", "digest_algorithm"):
                if alias in raw:
                    algorithm_value = raw[alias]
                    break
        algorithm = normalize_algorithm(algorithm_value)
        if algorithm is None:
            findings.append(
                _finding(
                    "E_ALGORITHM",
                    "snapshot algorithm must be SHA-256 or HMAC-SHA256",
                    version=version,
                    **source_extra,
                )
            )
            return None, findings
        if algorithm not in self.policy.algorithms:
            findings.append(
                _finding(
                    "E_ALGORITHM",
                    "snapshot algorithm is not in the active whitelist",
                    version=version,
                    algorithm=algorithm,
                    allowed_algorithms=list(self.policy.algorithms),
                    **source_extra,
                )
            )

        prev_present = False
        prev_value: Any = None
        for key in (
            "previous_fingerprint",
            "prev_fingerprint",
            "previous_snapshot_fingerprint",
            "parent_fingerprint",
        ):
            if key in raw:
                prev_present = True
                prev_value = raw[key]
                break
        previous_link_valid = True
        previous_fingerprint: str | None = None
        if prev_present and prev_value is not None:
            previous_fingerprint = normalize_hex_fingerprint(prev_value)
            if previous_fingerprint is None:
                previous_link_valid = False
                findings.append(
                    _finding(
                        "E_FINGERPRINT",
                        "previous_fingerprint must be a 64-character hexadecimal digest",
                        version=version,
                        declared_fingerprint=prev_value,
                        **source_extra,
                    )
                )

        previous_version_value = raw.get("previous_version")
        if previous_version_value is None:
            previous_version_value = raw.get("prev_version")
        if previous_version_value is None:
            previous_version_value = raw.get("parent_version")
        previous_version: int | None = None
        if previous_version_value is not None:
            if (
                isinstance(previous_version_value, int)
                and not isinstance(previous_version_value, bool)
                and previous_version_value > 0
            ):
                previous_version = previous_version_value
            else:
                previous_link_valid = False
                findings.append(
                    _finding(
                        "E_CHAIN_BREAK",
                        "previous_version must be a positive integer",
                        version=version,
                        **source_extra,
                    )
                )

        entries_value = raw.get("entries")
        if not isinstance(entries_value, list):
            findings.append(
                _finding(
                    "E_JSON",
                    "snapshot entries must be a list",
                    version=version,
                    **source_extra,
                )
            )
            return None, findings
        if version is None:
            return None, findings

        if algorithm == "hmac-sha256" and self.policy.hmac_key is None:
            findings.append(
                _finding(
                    "E_ALGORITHM",
                    "HMAC-SHA256 snapshot cannot be verified without an HMAC key",
                    version=version,
                    algorithm=algorithm,
                    **source_extra,
                )
            )
            return None, findings

        try:
            fingerprint = compute_snapshot_fingerprint(raw, algorithm, self.policy.hmac_key)
        except (TypeError, ValueError) as exc:
            findings.append(
                _finding(
                    "E_JSON",
                    f"snapshot cannot be fingerprinted: {exc}",
                    version=version,
                    **source_extra,
                )
            )
            return None, findings

        declared_fingerprint = raw.get("fingerprint")
        if declared_fingerprint is not None:
            normalized_declared = normalize_hex_fingerprint(declared_fingerprint)
            if normalized_declared != fingerprint:
                findings.append(
                    _finding(
                        "E_FINGERPRINT",
                        "declared self fingerprint does not match the computed fingerprint",
                        version=version,
                        fingerprint=fingerprint,
                        actual_prefix=digest_prefix(fingerprint),
                        declared_prefix=(
                            digest_prefix(normalized_declared)
                            if normalized_declared is not None
                            else None
                        ),
                        **source_extra,
                    )
                )

        entry_results, entry_findings = self._parse_entries(
            entries_value, version, algorithm
        )
        findings.extend(entry_findings)
        snapshot = Snapshot(
            raw=raw,
            version=version,
            algorithm=algorithm,
            previous_fingerprint=previous_fingerprint,
            previous_version=previous_version,
            fingerprint=fingerprint,
            entry_results=entry_results,
            top_findings=[
                item for item in findings if item.get("entry_index") is None
            ],
            previous_link_valid=previous_link_valid,
        )
        return snapshot, _sorted_findings(findings)

    def _parse_entries(
        self, raw_entries: list[Any], version: int, fallback_algorithm: str
    ) -> tuple[list[ParsedEntry], list[dict[str, Any]]]:
        results: list[ParsedEntry] = []
        findings: list[dict[str, Any]] = []
        seen_paths: dict[str, int] = {}

        for index, raw_entry in enumerate(raw_entries):
            item_findings: list[dict[str, Any]] = []
            path: str | None = None
            exclusion_candidate = False
            deletion = False
            content_b64: str | None = None
            normalized_digest: str | None = None
            declared_length: int | None = None

            if not isinstance(raw_entry, dict):
                item_findings.append(
                    _finding(
                        "E_JSON",
                        "entry must be a JSON object",
                        version=version,
                        entry_index=index,
                    )
                )
                results.append(
                    ParsedEntry(index, None, False, False, False, item_findings)
                )
                findings.extend(item_findings)
                continue

            path, path_error = normalize_relative_path(raw_entry.get("path"))
            if path_error is not None:
                item_findings.append(
                    _finding(
                        path_error,
                        "entry path must be a safe, non-absolute relative path",
                        version=version,
                        entry_index=index,
                        declared_path=raw_entry.get("path"),
                    )
                )
                results.append(
                    ParsedEntry(index, None, False, False, False, item_findings)
                )
                findings.extend(item_findings)
                continue

            path_key = path.casefold()
            if path_key in seen_paths:
                item_findings.append(
                    _finding(
                        "E_DUPLICATE",
                        "entry path is duplicated within this snapshot",
                        version=version,
                        entry_index=index,
                        path=path,
                        first_entry_index=seen_paths[path_key],
                    )
                )
                results.append(
                    ParsedEntry(index, path, False, False, False, item_findings)
                )
                findings.extend(item_findings)
                continue
            seen_paths[path_key] = index
            exclusion_candidate = True

            tampered_value = raw_entry.get("tampered", False)
            if not isinstance(tampered_value, bool):
                item_findings.append(
                    _finding(
                        "E_TAMPER",
                        "tampered marker must be a boolean",
                        version=version,
                        entry_index=index,
                        path=path,
                    )
                )
                tampered_value = True

            deleted_value = raw_entry.get("deleted", False)
            if not isinstance(deleted_value, bool):
                item_findings.append(
                    _finding(
                        "E_TAMPER",
                        "deleted marker must be a boolean",
                        version=version,
                        entry_index=index,
                        path=path,
                    )
                )
                deleted_value = False
            deletion = deleted_value

            if tampered_value:
                item_findings.append(
                    _finding(
                        "E_TAMPER",
                        "entry is marked as tampered",
                        version=version,
                        entry_index=index,
                        path=path,
                    )
                )

            if deletion:
                accepted = not item_findings
                results.append(
                    ParsedEntry(
                        index,
                        path,
                        accepted,
                        True,
                        exclusion_candidate and bool(item_findings),
                        item_findings,
                    )
                )
                findings.extend(item_findings)
                continue

            declared_length_value = raw_entry.get("length")
            if declared_length_value is None:
                declared_length_value = raw_entry.get("declared_length")
            if (
                isinstance(declared_length_value, int)
                and not isinstance(declared_length_value, bool)
                and declared_length_value >= 0
            ):
                declared_length = declared_length_value
            else:
                item_findings.append(
                    _finding(
                        "E_LENGTH",
                        "declared length must be a non-negative integer",
                        version=version,
                        entry_index=index,
                        path=path,
                        declared_length=declared_length_value,
                    )
                )

            digest_value = raw_entry.get("digest")
            if digest_value is None:
                digest_value = raw_entry.get("content_digest")
            digest_algorithm, digest_hex, digest_error = normalize_digest(
                digest_value, fallback_algorithm
            )
            if digest_error is not None:
                item_findings.append(
                    _finding(
                        digest_error,
                        "entry digest must use a whitelisted algorithm and 64 hexadecimal characters",
                        version=version,
                        entry_index=index,
                        path=path,
                        declared_digest=digest_value,
                    )
                )
            elif digest_algorithm not in self.policy.algorithms:
                item_findings.append(
                    _finding(
                        "E_ALGORITHM",
                        "entry digest algorithm is not in the active whitelist",
                        version=version,
                        entry_index=index,
                        path=path,
                        algorithm=digest_algorithm,
                    )
                )
            elif digest_hex is not None:
                normalized_digest = f"{digest_algorithm}:{digest_hex}"

            content, content_error = decode_content(raw_entry)
            if content_error is not None:
                item_findings.append(
                    _finding(
                        "E_TAMPER",
                        "entry content is missing, ambiguous, or not valid base64/UTF-8",
                        version=version,
                        entry_index=index,
                        path=path,
                    )
                )
            elif content is not None:
                content_b64 = encode_content(content)
                actual_digest = None
                if digest_algorithm is not None and digest_hex is not None:
                    if digest_algorithm == "hmac-sha256" and self.policy.hmac_key is None:
                        item_findings.append(
                            _finding(
                                "E_ALGORITHM",
                                "HMAC-SHA256 entry cannot be verified without an HMAC key",
                                version=version,
                                entry_index=index,
                                path=path,
                                algorithm=digest_algorithm,
                            )
                        )
                    else:
                        try:
                            actual_digest = compute_content_digest(
                                digest_algorithm, content, self.policy.hmac_key
                            ).hex()
                        except ValueError:
                            actual_digest = None
                if actual_digest is not None and digest_hex is not None:
                    if actual_digest != digest_hex:
                        item_findings.append(
                            _finding(
                                "E_TAMPER",
                                "entry content digest does not match the declared digest",
                                version=version,
                                entry_index=index,
                                path=path,
                                actual_prefix=actual_digest[:DIGEST_PREFIX_LENGTH],
                                declared_prefix=digest_hex[:DIGEST_PREFIX_LENGTH],
                            )
                        )
                if declared_length is not None and len(content) != declared_length:
                    item_findings.append(
                        _finding(
                            "E_LENGTH",
                            "entry content length does not match the declared length",
                            version=version,
                            entry_index=index,
                            path=path,
                            actual_length=len(content),
                            declared_length=declared_length,
                        )
                    )

            accepted = not item_findings
            results.append(
                ParsedEntry(
                    index=index,
                    path=path,
                    accepted=accepted,
                    deletion=False,
                    exclusion_candidate=exclusion_candidate and bool(item_findings),
                    findings=item_findings,
                    content_b64=content_b64,
                    digest=normalized_digest,
                    length=len(content) if content is not None else declared_length,
                )
            )
            findings.extend(item_findings)

        return results, _sorted_findings(findings)

    def _parse_snapshot_file(
        self, path: Path, *, include_source: bool = False
    ) -> tuple[Snapshot | None, list[dict[str, Any]]]:
        try:
            raw_bytes = path.read_bytes()
        except OSError as exc:
            return None, [
                _finding(
                    "E_IO",
                    f"cannot read snapshot: {exc}",
                    **({"source": path.name} if include_source else {}),
                )
            ]
        try:
            raw = strict_json_loads(raw_bytes)
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            return None, [
                _finding(
                    "E_JSON",
                    f"snapshot is not valid UTF-8 JSON: {exc}",
                    **({"source": path.name} if include_source else {}),
                )
            ]
        return self._parse_snapshot_object(raw, include_source=path.name if include_source else None)

    # ------------------------------------------------------------------
    # Deterministic graph evaluation
    # ------------------------------------------------------------------

    def _evaluate_structure(
        self,
        snapshots: list[Snapshot],
        baseline: dict[str, Any] | None,
        explicit_fingerprints: set[str] | None = None,
    ) -> dict[str, Any]:
        by_fp = {snapshot.fingerprint: snapshot for snapshot in snapshots}
        by_version: dict[int, list[Snapshot]] = {}
        for snapshot in snapshots:
            by_version.setdefault(snapshot.version, []).append(snapshot)
        for values in by_version.values():
            values.sort(key=lambda item: item.fingerprint)

        findings: list[dict[str, Any]] = []
        blocked: set[str] = set()
        cycle_bad: set[str] = set()
        fork_bad: set[str] = set()
        fingerprint_bad: set[str] = set()
        rollback_bad: set[str] = set()
        chain_bad: set[str] = set()

        for snapshot in snapshots:
            for finding in snapshot.top_findings:
                findings.append(finding)
            if snapshot.top_findings:
                blocked.add(snapshot.fingerprint)
                chain_bad.add(snapshot.fingerprint)
                if any(
                    finding.get("code") == "E_FINGERPRINT"
                    for finding in snapshot.top_findings
                ):
                    fingerprint_bad.add(snapshot.fingerprint)

        # Detect cycles in the declared previous-fingerprint graph. Nodes have
        # at most one outgoing edge, so a colour walk is sufficient.
        edges: dict[str, str] = {}
        for snapshot in snapshots:
            previous = snapshot.previous_fingerprint
            if previous is not None and snapshot.previous_link_valid and previous in by_fp:
                edges[snapshot.fingerprint] = previous

        visited: set[str] = set()
        active: set[str] = set()
        cycle_groups: list[list[str]] = []
        for start in sorted(by_fp):
            if start in visited:
                continue
            walk: list[str] = []
            positions: dict[str, int] = {}
            current: str | None = start
            while current is not None and current not in visited:
                if current in active and current in positions:
                    cycle = walk[positions[current] :]
                    if cycle:
                        cycle_groups.append(sorted(cycle))
                    break
                active.add(current)
                positions[current] = len(walk)
                walk.append(current)
                current = edges.get(current)
            for node in walk:
                visited.add(node)
                active.discard(node)

        cycle_signatures: set[tuple[str, ...]] = set()
        for group in sorted(cycle_groups):
            signature = tuple(group)
            if signature in cycle_signatures:
                continue
            cycle_signatures.add(signature)
            cycle_bad.update(group)
            branches = [by_fp[item].meta() for item in group]
            findings.append(
                _finding(
                    "E_CYCLE",
                    "previous-fingerprint graph contains a cycle or self-reference",
                    versions=sorted({by_fp[item].version for item in group}),
                    branch_identifiers=group,
                    branches=branches,
                )
            )

        # A version-self reference is also a cycle even if the fingerprint edge
        # points somewhere else (or nowhere).
        for snapshot in snapshots:
            if snapshot.previous_version == snapshot.version:
                if snapshot.fingerprint not in cycle_bad:
                    cycle_bad.add(snapshot.fingerprint)
                    findings.append(
                        _finding(
                            "E_CYCLE",
                            "snapshot references its own version as the previous version",
                            version=snapshot.version,
                            fingerprint=snapshot.fingerprint,
                            branch_identifiers=[snapshot.fingerprint],
                            branches=[snapshot.meta()],
                        )
                    )

        # Fingerprint mismatch and chain continuity checks.
        for snapshot in snapshots:
            if not snapshot.previous_link_valid:
                continue
            if snapshot.fingerprint in cycle_bad:
                continue
            previous_fp = snapshot.previous_fingerprint
            previous_version = snapshot.previous_version

            if previous_version is not None:
                predecessors = by_version.get(previous_version, [])
                if not predecessors:
                    chain_bad.add(snapshot.fingerprint)
                    findings.append(
                        _finding(
                            "E_CHAIN_BREAK",
                            "declared previous_version does not exist in the snapshot set",
                            version=snapshot.version,
                            fingerprint=snapshot.fingerprint,
                            previous_version=previous_version,
                            actual_prefix=snapshot.fingerprint_prefix,
                        )
                    )
                else:
                    matching = [
                        item
                        for item in predecessors
                        if previous_fp is not None
                        and item.fingerprint == previous_fp
                    ]
                    if previous_fp is None:
                        chain_bad.add(snapshot.fingerprint)
                        findings.append(
                            _finding(
                                "E_FINGERPRINT",
                                "previous_version is present but previous_fingerprint is missing",
                                version=snapshot.version,
                                fingerprint=snapshot.fingerprint,
                                previous_version=previous_version,
                                actual_prefix=snapshot.fingerprint_prefix,
                            )
                        )
                    elif not matching:
                        chain_bad.add(snapshot.fingerprint)
                        fingerprint_bad.add(snapshot.fingerprint)
                        expected = sorted(item.fingerprint for item in predecessors)
                        findings.append(
                            _finding(
                                "E_FINGERPRINT",
                                "declared previous fingerprint does not match the snapshot at previous_version",
                                version=snapshot.version,
                                fingerprint=snapshot.fingerprint,
                                previous_version=previous_version,
                                actual_prefix=snapshot.fingerprint_prefix,
                                declared_prefix=digest_prefix(previous_fp),
                                expected_prefixes=[
                                    digest_prefix(item) for item in expected
                                ],
                            )
                        )
                if snapshot.version != previous_version + 1:
                    chain_bad.add(snapshot.fingerprint)
                    findings.append(
                        _finding(
                            "E_CHAIN_BREAK",
                            "snapshot version is not exactly one greater than previous_version",
                            version=snapshot.version,
                            fingerprint=snapshot.fingerprint,
                            previous_version=previous_version,
                            actual_prefix=snapshot.fingerprint_prefix,
                        )
                    )
            elif previous_fp is None:
                if snapshot.version != 1:
                    chain_bad.add(snapshot.fingerprint)
                    findings.append(
                        _finding(
                            "E_CHAIN_BREAK",
                            "only a version-1 snapshot may be a chain root",
                            version=snapshot.version,
                            fingerprint=snapshot.fingerprint,
                            actual_prefix=snapshot.fingerprint_prefix,
                        )
                    )
            else:
                predecessor = by_fp.get(previous_fp)
                if predecessor is None:
                    chain_bad.add(snapshot.fingerprint)
                    findings.append(
                        _finding(
                            "E_CHAIN_BREAK",
                            "previous fingerprint is not present in the snapshot set",
                            version=snapshot.version,
                            fingerprint=snapshot.fingerprint,
                            actual_prefix=snapshot.fingerprint_prefix,
                            declared_prefix=digest_prefix(previous_fp),
                        )
                    )
                elif snapshot.version != predecessor.version + 1:
                    chain_bad.add(snapshot.fingerprint)
                    findings.append(
                        _finding(
                            "E_CHAIN_BREAK",
                            "snapshot version does not immediately follow its fingerprint predecessor",
                            version=snapshot.version,
                            fingerprint=snapshot.fingerprint,
                            previous_version=predecessor.version,
                            actual_prefix=snapshot.fingerprint_prefix,
                        )
                    )

        # Fork detection groups all successors of the same predecessor. A null
        # predecessor is a shared genesis branch point.
        predecessor_groups: dict[str | None, list[Snapshot]] = {}
        for snapshot in snapshots:
            if not snapshot.previous_link_valid:
                continue
            predecessor_groups.setdefault(snapshot.previous_fingerprint, []).append(snapshot)

        fork_signatures: set[tuple[str, ...]] = set()
        for predecessor in sorted(
            predecessor_groups, key=lambda value: "" if value is None else value
        ):
            group = predecessor_groups[predecessor]
            distinct = sorted({item.fingerprint for item in group})
            if len(distinct) <= 1:
                continue
            fork_signatures.add(tuple(distinct))
            fork_bad.update(distinct)
            findings.append(
                _finding(
                    "E_FORK",
                    "multiple snapshots declare the same previous fingerprint",
                    predecessor_fingerprint=predecessor,
                    versions=sorted({item.version for item in group}),
                    branch_identifiers=distinct,
                    branches=[by_fp[item].meta() for item in distinct],
                )
            )

        for version, group in sorted(by_version.items()):
            distinct = sorted({item.fingerprint for item in group})
            if len(distinct) <= 1 or tuple(distinct) in fork_signatures:
                continue
            fork_bad.update(distinct)
            findings.append(
                _finding(
                    "E_FORK",
                    "multiple distinct snapshots declare the same version",
                    versions=[version],
                    branch_identifiers=distinct,
                    branches=[by_fp[item].meta() for item in distinct],
                )
            )

        # Rollback is measured only against the persisted checkpoint watermark.
        rollback_snapshots: list[Snapshot] = []
        watermark = baseline["watermark"] if baseline is not None else 0
        applied_fps = (
            {item["fingerprint"] for item in baseline["applied_snapshots"]}
            if baseline is not None
            else set()
        )
        explicit_fingerprints = explicit_fingerprints or set()
        if baseline is not None:
            for snapshot in snapshots:
                if snapshot.version < watermark and (
                    snapshot.fingerprint not in applied_fps
                    or snapshot.fingerprint in explicit_fingerprints
                ):
                    rollback_snapshots.append(snapshot)
                    rollback_bad.add(snapshot.fingerprint)
                elif (
                    snapshot.version == watermark
                    and snapshot.fingerprint not in applied_fps
                ):
                    fork_bad.add(snapshot.fingerprint)
                    findings.append(
                        _finding(
                            "E_FORK",
                            "a different snapshot conflicts with the checkpoint at the watermark version",
                            version=snapshot.version,
                            fingerprint=snapshot.fingerprint,
                            watermark=watermark,
                            branch_identifiers=[snapshot.fingerprint],
                            branches=[snapshot.meta()],
                        )
                    )
        if rollback_snapshots:
            rollback_snapshots.sort(key=lambda item: item.sort_key())
            versions = sorted({item.version for item in rollback_snapshots})
            findings.append(
                _finding(
                    "E_ROLLBACK",
                    "snapshot version is lower than the persisted checkpoint watermark",
                    versions=versions,
                    watermark=watermark,
                    branch_identifiers=[
                        item.fingerprint for item in rollback_snapshots
                    ],
                    branches=[item.meta() for item in rollback_snapshots],
                )
            )

        blocked.update(cycle_bad)
        blocked.update(fork_bad)
        blocked.update(rollback_bad)
        blocked.update(chain_bad)
        return {
            "applied_fingerprints": applied_fps,
            "blocked": blocked,
            "by_fp": by_fp,
            "by_version": by_version,
            "chain_bad": chain_bad,
            "cycle_bad": cycle_bad,
            "findings": _sorted_findings(findings),
            "fork_bad": fork_bad,
            "fingerprint_bad": fingerprint_bad,
            "rollback_bad": rollback_bad,
            "watermark": watermark,
        }

    def _select_lineage(
        self,
        snapshots: list[Snapshot],
        structure: dict[str, Any],
        baseline: dict[str, Any] | None,
    ) -> list[Snapshot]:
        by_fp: dict[str, Snapshot] = structure["by_fp"]
        by_version: dict[int, list[Snapshot]] = structure["by_version"]
        blocked: set[str] = structure["blocked"]
        lineage: list[Snapshot] = []

        if baseline is not None:
            current_version = baseline["watermark"]
            current_fp = baseline["head_fingerprint"]
        else:
            roots = [
                item
                for item in by_version.get(1, [])
                if item.previous_link_valid
                and item.previous_fingerprint is None
                and item.previous_version is None
                and item.fingerprint not in blocked
            ]
            roots.sort(key=lambda item: item.fingerprint)
            if len(roots) != 1:
                return []
            root = roots[0]
            lineage.append(root)
            current_version = root.version
            current_fp = root.fingerprint

        while current_fp is not None:
            desired_version = current_version + 1
            candidates = [
                item
                for item in by_version.get(desired_version, [])
                if item.previous_link_valid
                and item.previous_fingerprint == current_fp
                and item.fingerprint not in blocked
            ]
            candidates.sort(key=lambda item: item.fingerprint)
            if len(candidates) != 1:
                break
            candidate = candidates[0]
            lineage.append(candidate)
            current_version = candidate.version
            current_fp = candidate.fingerprint
        return lineage

    def _apply_entries(
        self,
        entries: dict[str, dict[str, Any]],
        exclusions: dict[str, dict[str, Any]],
        snapshots: list[Snapshot],
    ) -> None:
        for snapshot in snapshots:
            for result in sorted(snapshot.entry_results, key=lambda item: item.index):
                if result.accepted:
                    if result.path is None:
                        continue
                    if result.deletion:
                        entries.pop(result.path, None)
                        exclusions.pop(result.path, None)
                        continue
                    if (
                        result.content_b64 is None
                        or result.digest is None
                        or result.length is None
                    ):
                        continue
                    entries[result.path] = {
                        "content_b64": result.content_b64,
                        "digest": result.digest,
                        "length": result.length,
                        "path": result.path,
                        "source_version": snapshot.version,
                    }
                    exclusions.pop(result.path, None)
                elif result.exclusion_candidate and result.path is not None:
                    entries.pop(result.path, None)
                    codes = sorted({finding["code"] for finding in result.findings})
                    existing = exclusions.get(result.path)
                    if existing is None:
                        exclusions[result.path] = {
                            "codes": codes,
                            "path": result.path,
                            "versions": [snapshot.version],
                        }
                    else:
                        existing["codes"] = sorted(set(existing["codes"]) | set(codes))
                        existing["versions"] = sorted(
                            set(existing["versions"]) | {snapshot.version}
                        )

    @staticmethod
    def _snapshot_status(
        snapshot: Snapshot,
        *,
        lineage: list[Snapshot],
        baseline: dict[str, Any] | None,
        structure: dict[str, Any],
    ) -> str:
        lineage_fps = {item.fingerprint for item in lineage}
        if snapshot.fingerprint in lineage_fps:
            return "applied"
        if snapshot.fingerprint in structure["rollback_bad"]:
            return "rejected_rollback"
        if (
            baseline is not None
            and snapshot.fingerprint in structure["applied_fingerprints"]
        ):
            return "already_applied"
        if snapshot.fingerprint in structure["fork_bad"]:
            return "rejected_fork"
        if snapshot.fingerprint in structure["cycle_bad"]:
            return "rejected_cycle"
        if snapshot.fingerprint in structure["fingerprint_bad"]:
            return "rejected_fingerprint"
        if snapshot.fingerprint in structure["chain_bad"]:
            return "rejected_chain"
        return "not_selected"

    # ------------------------------------------------------------------
    # Checkpoints and restore generations
    # ------------------------------------------------------------------

    def _read_checkpoint(self) -> tuple[dict[str, Any] | None, bool, str | None]:
        if not self.checkpoint_path.exists():
            return None, False, None
        try:
            raw = strict_json_loads(self.checkpoint_path.read_bytes())
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            return None, True, "checkpoint file is not valid UTF-8 JSON"
        if not isinstance(raw, dict):
            return None, True, "checkpoint root must be a JSON object"
        return raw, True, None

    def _restore_path_for_checkpoint(self, checkpoint: dict[str, Any]) -> Path | None:
        value = checkpoint.get("restore_file")
        if not isinstance(value, str) or not value:
            return None
        relative = PurePosixPath(value)
        if relative.is_absolute() or re.match(r"^[A-Za-z]:", value):
            return None
        if any(part in {"", ".", ".."} for part in relative.parts):
            return None
        candidate = (self.workspace / Path(*relative.parts)).resolve()
        try:
            if os.path.commonpath([str(candidate), str(self.workspace)]) != str(
                self.workspace
            ):
                return None
        except ValueError:
            return None
        return candidate

    def _validate_checkpoint_data(
        self, checkpoint: dict[str, Any]
    ) -> tuple[bool, list[dict[str, Any]], dict[str, Any] | None]:
        findings: list[dict[str, Any]] = []

        def reject(code: str, message: str, **extra: Any) -> None:
            findings.append(_finding(code, message, **extra))

        if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            reject(
                "E_CHECKPOINT",
                "unsupported checkpoint schema_version",
                declared_schema_version=checkpoint.get("schema_version"),
                expected_schema_version=CHECKPOINT_SCHEMA_VERSION,
            )
            return False, _sorted_findings(findings), None

        generation = checkpoint.get("generation")
        if (
            not isinstance(generation, int)
            or isinstance(generation, bool)
            or generation < 1
        ):
            reject("E_CHECKPOINT", "checkpoint generation must be a positive integer")

        policy = checkpoint.get("policy")
        if not isinstance(policy, dict):
            reject("E_CHECKPOINT", "checkpoint policy must be an object")
            return False, _sorted_findings(findings), None
        algorithms = policy.get("algorithms")
        if not isinstance(algorithms, list) or not algorithms:
            reject("E_CHECKPOINT", "checkpoint policy algorithms must be a non-empty list")
            return False, _sorted_findings(findings), None
        normalized_algorithms: list[str] = []
        for value in algorithms:
            normalized = normalize_algorithm(value)
            if normalized not in SUPPORTED_ALGORITHMS:
                reject("E_CHECKPOINT", "checkpoint policy contains an unsupported algorithm")
                return False, _sorted_findings(findings), None
            normalized_algorithms.append(normalized)
        if normalized_algorithms != sorted(set(normalized_algorithms)):
            reject("E_CHECKPOINT", "checkpoint policy algorithms are not normalized and sorted")

        restore_path = self._restore_path_for_checkpoint(checkpoint)
        if restore_path is None:
            reject("E_CHECKPOINT", "checkpoint restore_file is not a safe workspace-relative path")
            return False, _sorted_findings(findings), None
        if not restore_path.is_file():
            reject("E_CHECKPOINT", "checkpoint restore file is missing")
            return False, _sorted_findings(findings), None
        try:
            restore_sha = file_sha256(restore_path)
        except OSError as exc:
            reject("E_CHECKPOINT", f"cannot hash checkpoint restore file: {exc}")
            return False, _sorted_findings(findings), None
        if restore_sha != checkpoint.get("restore_sha256"):
            reject(
                "E_CHECKPOINT",
                "checkpoint restore file digest does not match",
                actual_prefix=restore_sha[:DIGEST_PREFIX_LENGTH],
                declared_prefix=str(checkpoint.get("restore_sha256"))[
                    :DIGEST_PREFIX_LENGTH
                ],
            )
        try:
            restore = strict_json_loads(restore_path.read_bytes())
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            reject("E_CHECKPOINT", "checkpoint restore file is not valid UTF-8 JSON")
            return False, _sorted_findings(findings), None
        if not isinstance(restore, dict):
            reject("E_CHECKPOINT", "checkpoint restore root must be an object")
            return False, _sorted_findings(findings), None

        if restore.get("schema_version") != RESTORE_SCHEMA_VERSION:
            reject("E_CHECKPOINT", "unsupported restore schema_version")
        if restore.get("policy") != policy:
            reject("E_CHECKPOINT", "checkpoint and restore policy differ")

        state_sha = restore.get("state_sha256")
        material = {key: value for key, value in restore.items() if key != "state_sha256"}
        actual_state_sha = sha256_hex(canonical_json_bytes(material))
        if state_sha != actual_state_sha:
            reject(
                "E_CHECKPOINT",
                "restore state digest does not match its contents",
                actual_prefix=actual_state_sha[:DIGEST_PREFIX_LENGTH],
                declared_prefix=str(state_sha)[:DIGEST_PREFIX_LENGTH],
            )
        if checkpoint.get("state_sha256") != state_sha:
            reject("E_CHECKPOINT", "checkpoint and restore state digests differ")

        applied = checkpoint.get("applied_snapshots")
        if not isinstance(applied, list):
            reject("E_CHECKPOINT", "applied_snapshots must be a list")
            applied = []
        normalized_applied: list[dict[str, Any]] = []
        seen_applied: set[str] = set()
        for index, item in enumerate(applied):
            if not isinstance(item, dict):
                reject("E_CHECKPOINT", "applied snapshot metadata must be an object")
                continue
            version = item.get("version")
            fingerprint = normalize_hex_fingerprint(item.get("fingerprint"))
            if (
                not isinstance(version, int)
                or isinstance(version, bool)
                or version < 1
                or fingerprint is None
            ):
                reject("E_CHECKPOINT", "applied snapshot metadata is malformed")
                continue
            if fingerprint in seen_applied:
                reject("E_CHECKPOINT", "applied snapshot fingerprint is duplicated")
            seen_applied.add(fingerprint)
            normalized_applied.append(
                {
                    "algorithm": normalize_algorithm(item.get("algorithm")) or "sha256",
                    "fingerprint": fingerprint,
                    "fingerprint_prefix": digest_prefix(fingerprint),
                    "version": version,
                }
            )
        expected_applied = sorted(
            normalized_applied, key=lambda item: (item["version"], item["fingerprint"])
        )
        if normalized_applied != expected_applied:
            reject("E_CHECKPOINT", "applied_snapshots are not in stable order")
        applied_versions_list = [item["version"] for item in normalized_applied]
        if len(applied_versions_list) != len(set(applied_versions_list)):
            reject("E_CHECKPOINT", "applied_snapshots contain duplicate versions")
        if applied_versions_list != list(range(1, len(applied_versions_list) + 1)):
            reject("E_CHECKPOINT", "applied snapshot versions are not a contiguous chain")
        for item in normalized_applied:
            if item["algorithm"] not in normalized_algorithms:
                reject(
                    "E_CHECKPOINT",
                    "applied snapshot algorithm is outside checkpoint policy",
                )
        if restore.get("applied_snapshots") != applied:
            reject("E_CHECKPOINT", "checkpoint and restore applied snapshots differ")

        watermark = checkpoint.get("watermark")
        expected_watermark = max((item["version"] for item in normalized_applied), default=0)
        if (
            not isinstance(watermark, int)
            or isinstance(watermark, bool)
            or watermark < 0
        ):
            reject("E_CHECKPOINT", "checkpoint watermark must be a non-negative integer")
        elif watermark != expected_watermark:
            reject("E_CHECKPOINT", "checkpoint watermark does not match applied snapshots")

        head = checkpoint.get("head_fingerprint")
        expected_head = expected_applied[-1]["fingerprint"] if expected_applied else None
        if head != expected_head:
            reject("E_CHECKPOINT", "checkpoint head_fingerprint is inconsistent")

        excluded = checkpoint.get("excluded_paths")
        if not isinstance(excluded, list):
            reject("E_CHECKPOINT", "excluded_paths must be a list")
            excluded = []
        if restore.get("excluded_paths") != excluded:
            reject("E_CHECKPOINT", "checkpoint and restore exclusions differ")
        normalized_exclusions: list[dict[str, Any]] = []
        for item in excluded:
            if not isinstance(item, dict):
                reject("E_CHECKPOINT", "excluded path record must be an object")
                continue
            path, path_error = normalize_relative_path(item.get("path"))
            versions = item.get("versions")
            codes = item.get("codes")
            if (
                path_error is not None
                or not isinstance(versions, list)
                or not isinstance(codes, list)
            ):
                reject("E_CHECKPOINT", "excluded path record is malformed")
                continue
            normalized_exclusions.append(
                {
                    "codes": sorted({str(code) for code in codes}),
                    "path": path,
                    "versions": sorted(
                        {value for value in versions if isinstance(value, int)}
                    ),
                }
            )
        if normalized_exclusions != sorted(
            normalized_exclusions, key=lambda item: path_sort_key(item["path"])
        ):
            reject("E_CHECKPOINT", "excluded paths are not in stable order")

        entries = restore.get("entries")
        if not isinstance(entries, list):
            reject("E_CHECKPOINT", "restore entries must be a list")
            entries = []
        normalized_entries: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        applied_versions = {item["version"] for item in normalized_applied}
        hmac_key_matches_checkpoint = (
            self.policy.hmac_key is not None
            and policy.get("hmac_key_id") == self.policy.hmac_key_id
        )
        for item in entries:
            if not isinstance(item, dict):
                reject("E_CHECKPOINT", "restore entry must be an object")
                continue
            path, path_error = normalize_relative_path(item.get("path"))
            if path_error is not None:
                reject("E_CHECKPOINT", "restore entry path is invalid")
                continue
            path_key = path.casefold()
            if path_key in seen_paths:
                reject("E_CHECKPOINT", "restore entry path is duplicated")
            seen_paths.add(path_key)
            digest_algorithm, digest_hex, digest_error = normalize_digest(
                item.get("digest"), "sha256"
            )
            if digest_error is not None or digest_algorithm is None or digest_hex is None:
                reject("E_CHECKPOINT", "restore entry digest is malformed", path=path)
                continue
            if digest_algorithm not in normalized_algorithms:
                reject(
                    "E_CHECKPOINT",
                    "restore entry digest algorithm is outside checkpoint policy",
                    path=path,
                )
            content, content_error = decode_content(item)
            if content_error is not None or content is None:
                reject("E_CHECKPOINT", "restore entry content is malformed", path=path)
                continue
            length = item.get("length")
            if (
                not isinstance(length, int)
                or isinstance(length, bool)
                or length < 0
                or length != len(content)
            ):
                reject("E_CHECKPOINT", "restore entry length is inconsistent", path=path)
            source_version = item.get("source_version")
            if source_version not in applied_versions:
                reject(
                    "E_CHECKPOINT",
                    "restore entry source_version is not an applied snapshot",
                    path=path,
                )
            if digest_algorithm == "hmac-sha256" and not hmac_key_matches_checkpoint:
                # The state checksum still protects bytes; cryptographic HMAC
                # re-verification requires the exact checkpoint key identity.
                pass
            else:
                try:
                    actual = compute_content_digest(
                        digest_algorithm, content, self.policy.hmac_key
                    ).hex()
                except ValueError:
                    actual = None
                if actual is not None and actual != digest_hex:
                    reject(
                        "E_CHECKPOINT",
                        "restore entry content digest does not match",
                        path=path,
                        actual_prefix=actual[:DIGEST_PREFIX_LENGTH],
                        declared_prefix=digest_hex[:DIGEST_PREFIX_LENGTH],
                    )
            normalized_entries.append(
                {
                    "content_b64": encode_content(content),
                    "digest": f"{digest_algorithm}:{digest_hex}",
                    "length": len(content),
                    "path": path,
                    "source_version": source_version,
                }
            )
        if normalized_entries != sorted(
            normalized_entries, key=lambda item: path_sort_key(item["path"])
        ):
            reject("E_CHECKPOINT", "restore entries are not in stable path order")
        excluded_paths = {item["path"] for item in normalized_exclusions}
        if excluded_paths & seen_paths:
            reject("E_CHECKPOINT", "a path cannot be both restored and excluded")
        for item in normalized_exclusions:
            if not item["versions"] or any(
                version not in applied_versions for version in item["versions"]
            ):
                reject(
                    "E_CHECKPOINT",
                    "excluded path source versions are not applied snapshots",
                    path=item["path"],
                )

        return not findings, _sorted_findings(findings), restore

    def _load_checkpoint_for_recovery(
        self,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]], bool, dict[str, Any] | None]:
        checkpoint, present, read_error = self._read_checkpoint()
        if checkpoint is None:
            if present:
                return (
                    None,
                    [
                        _finding(
                            "E_CHECKPOINT_REBUILD",
                            "checkpoint is unreadable; rebuilding from snapshots",
                            detail=read_error,
                        )
                    ],
                    True,
                    None,
                )
            return None, [], False, None
        policy_matches = checkpoint.get("policy") == self.policy.to_json()
        schema_matches = checkpoint.get("schema_version") == CHECKPOINT_SCHEMA_VERSION
        if not schema_matches:
            return (
                None,
                [
                    _finding(
                        "E_CHECKPOINT_REBUILD",
                        "checkpoint schema version changed; rebuilding from snapshots",
                        declared_schema_version=checkpoint.get("schema_version"),
                        expected_schema_version=CHECKPOINT_SCHEMA_VERSION,
                    )
                ],
                True,
                checkpoint,
            )
        if not policy_matches:
            return (
                None,
                [
                    _finding(
                        "E_POLICY_REBUILD",
                        "digest policy or HMAC key identity changed; rebuilding from snapshots",
                        checkpoint_policy=checkpoint.get("policy"),
                        active_policy=self.policy.to_json(),
                    )
                ],
                True,
                checkpoint,
            )
        valid, findings, restore = self._validate_checkpoint_data(checkpoint)
        if not valid or restore is None:
            return (
                None,
                [
                    _finding(
                        "E_CHECKPOINT_REBUILD",
                        "checkpoint is incomplete or inconsistent; rebuilding from snapshots",
                    ),
                    *findings,
                ],
                True,
                checkpoint,
            )
        applied = [
            {
                "algorithm": normalize_algorithm(item.get("algorithm")) or "sha256",
                "fingerprint": item["fingerprint"],
                "fingerprint_prefix": digest_prefix(item["fingerprint"]),
                "version": item["version"],
            }
            for item in checkpoint["applied_snapshots"]
        ]
        entries = {
            item["path"]: dict(item)
            for item in restore["entries"]
            if isinstance(item, dict) and isinstance(item.get("path"), str)
        }
        exclusions = {
            item["path"]: dict(item)
            for item in checkpoint["excluded_paths"]
            if isinstance(item, dict) and isinstance(item.get("path"), str)
        }
        baseline = {
            "applied_snapshots": applied,
            "entries": entries,
            "exclusions": exclusions,
            "generation": checkpoint.get("generation", 0),
            "head_fingerprint": checkpoint["head_fingerprint"],
            "state_sha256": checkpoint.get("state_sha256"),
            "watermark": checkpoint["watermark"],
        }
        return baseline, [], False, checkpoint

    def _next_generation(self, checkpoint: dict[str, Any] | None) -> int:
        maximum = 0
        if checkpoint is not None and isinstance(checkpoint.get("generation"), int):
            maximum = max(maximum, checkpoint["generation"])
        if self.restore_dir.exists():
            for path in self.restore_dir.glob("state-*.json"):
                match = re.match(r"state-(\d+)-", path.name)
                if match:
                    maximum = max(maximum, int(match.group(1)))
        return maximum + 1

    def _commit_state(
        self,
        *,
        entries: dict[str, dict[str, Any]],
        exclusions: dict[str, dict[str, Any]],
        applied_snapshots: list[dict[str, Any]],
        previous_checkpoint: dict[str, Any] | None,
    ) -> dict[str, Any]:
        applied = sorted(
            applied_snapshots, key=lambda item: (item["version"], item["fingerprint"])
        )
        deduplicated: list[dict[str, Any]] = []
        seen_fps: set[str] = set()
        for item in applied:
            fingerprint = item.get("fingerprint")
            if not isinstance(fingerprint, str) or fingerprint in seen_fps:
                continue
            seen_fps.add(fingerprint)
            deduplicated.append(
                {
                    "algorithm": normalize_algorithm(item.get("algorithm")) or "sha256",
                    "fingerprint": fingerprint,
                    "fingerprint_prefix": digest_prefix(fingerprint),
                    "version": int(item["version"]),
                }
            )
        applied = deduplicated
        watermark = max((item["version"] for item in applied), default=0)
        head_fingerprint = applied[-1]["fingerprint"] if applied else None
        excluded_list = sorted(
            (
                {
                    "codes": sorted({str(code) for code in item.get("codes", [])}),
                    "path": path,
                    "versions": sorted(
                        {value for value in item.get("versions", []) if isinstance(value, int)}
                    ),
                }
                for path, item in exclusions.items()
            ),
            key=lambda item: path_sort_key(item["path"]),
        )
        entries_list = sorted(
            (
                {
                    "content_b64": item["content_b64"],
                    "digest": item["digest"],
                    "length": item["length"],
                    "path": path,
                    "source_version": item["source_version"],
                }
                for path, item in entries.items()
            ),
            key=lambda item: path_sort_key(item["path"]),
        )
        restore_material: dict[str, Any] = {
            "applied_snapshots": applied,
            "entries": entries_list,
            "excluded_paths": excluded_list,
            "head_fingerprint": head_fingerprint,
            "policy": self.policy.to_json(),
            "schema_version": RESTORE_SCHEMA_VERSION,
            "watermark": watermark,
        }
        state_sha256 = sha256_hex(canonical_json_bytes(restore_material))
        restore_doc = dict(restore_material)
        restore_doc["state_sha256"] = state_sha256

        generation = self._next_generation(previous_checkpoint)
        restore_name = f"state-{generation:06d}-{state_sha256[:16]}.json"
        restore_path = self.restore_dir / restore_name
        if not restore_path.exists() or file_sha256(restore_path) != sha256_hex(
            canonical_json_bytes(restore_doc)
        ):
            atomic_write_json(
                restore_path,
                restore_doc,
                crash_after_temp_write="recovery_restore_after_temp",
                crash_before_replace="recovery_restore_before_replace",
            )
        if os.environ.get("BACKUP_ENGINE_CRASH_AT") == "recovery_after_restore_write":
            os._exit(94)
        restore_sha256 = file_sha256(restore_path)
        checkpoint_doc: dict[str, Any] = {
            "applied_snapshots": applied,
            "engine_version": ENGINE_VERSION,
            "excluded_paths": excluded_list,
            "generation": generation,
            "head_fingerprint": head_fingerprint,
            "policy": self.policy.to_json(),
            "restore_file": f"restore/{restore_name}",
            "restore_sha256": restore_sha256,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "state_sha256": state_sha256,
            "watermark": watermark,
        }
        atomic_write_json(
            self.checkpoint_path,
            checkpoint_doc,
            crash_after_temp_write="recovery_checkpoint_after_temp",
            crash_before_replace="recovery_before_checkpoint_replace",
        )
        valid, findings, _ = self._validate_checkpoint_data(checkpoint_doc)
        if not valid:
            raise RuntimeError(
                "committed checkpoint failed self-validation: "
                + "; ".join(item["message"] for item in findings)
            )
        return checkpoint_doc

    # ------------------------------------------------------------------
    # Public operations
    # ------------------------------------------------------------------

    def _store_snapshot(self, snapshot: Snapshot) -> tuple[str, str]:
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        filename = f"v{snapshot.version:010d}-{snapshot.fingerprint}.json"
        path = self.snapshots_dir / filename
        expected_sha = sha256_hex(canonical_json_bytes(snapshot.raw))
        if path.exists():
            try:
                if file_sha256(path) == expected_sha:
                    return filename, "already_imported"
            except OSError:
                pass
        atomic_write_json(path, snapshot.raw)
        return filename, "imported"

    def _load_snapshots_locked(self) -> tuple[list[Snapshot], list[dict[str, Any]]]:
        snapshots: list[Snapshot] = []
        findings: list[dict[str, Any]] = []
        if not self.snapshots_dir.exists():
            return snapshots, findings
        seen_fps: set[str] = set()
        for path in sorted(self.snapshots_dir.glob("*.json")):
            snapshot, parse_findings = self._parse_snapshot_file(path)
            if snapshot is None:
                findings.extend(parse_findings)
                continue
            if snapshot.fingerprint in seen_fps:
                continue
            filename_match = re.fullmatch(
                r"v\d+-([0-9a-f]{64})\.json", path.name
            )
            if filename_match and filename_match.group(1) != snapshot.fingerprint:
                snapshot.top_findings.append(
                    _finding(
                        "E_FINGERPRINT",
                        "stored snapshot content does not match its content-addressed filename",
                        version=snapshot.version,
                        fingerprint=snapshot.fingerprint,
                        actual_prefix=snapshot.fingerprint_prefix,
                        stored_prefix=filename_match.group(1)[:DIGEST_PREFIX_LENGTH],
                    )
                )
            seen_fps.add(snapshot.fingerprint)
            snapshots.append(snapshot)
        snapshots.sort(key=lambda item: item.sort_key())
        return snapshots, _sorted_findings(findings)

    @staticmethod
    def _report(command: str) -> dict[str, Any]:
        return {
            "command": command,
            "errors": [],
            "exit_code": 0,
            "ok": True,
            "summary": {},
            "warnings": [],
        }

    @staticmethod
    def _finalize(report: dict[str, Any]) -> dict[str, Any]:
        report["errors"] = _sorted_findings(report.get("errors", []))
        report["warnings"] = _sorted_findings(report.get("warnings", []))
        report["ok"] = not report["errors"]
        report["exit_code"] = 0 if report["ok"] else 1
        report.setdefault("summary", {})
        report["summary"]["error_count"] = len(report["errors"])
        report["summary"]["warning_count"] = len(report["warnings"])
        return report

    def import_snapshots(self, paths: Iterable[str | os.PathLike[str]]) -> dict[str, Any]:
        path_list = [Path(path).expanduser() for path in paths]
        if not path_list:
            raise ValueError("at least one snapshot path is required")
        report = self._report("import")
        report["snapshots"] = []
        lock = WorkspaceLock(self.lock_path)
        try:
            with lock:
                self.workspace.mkdir(parents=True, exist_ok=True)
                cleanup_orphan_temps(self.workspace)
                for path in path_list:
                    snapshot, findings = self._parse_snapshot_file(
                        path, include_source=True
                    )
                    report["errors"].extend(findings)
                    if snapshot is None:
                        continue
                    try:
                        stored_name, status = self._store_snapshot(snapshot)
                    except OSError as exc:
                        report["errors"].append(
                            _finding(
                                "E_IO",
                                f"cannot store snapshot: {exc}",
                                version=snapshot.version,
                                fingerprint=snapshot.fingerprint,
                                source=path.name,
                            )
                        )
                        continue
                    report["snapshots"].append(
                        {
                            **snapshot.meta(),
                            "source": path.name,
                            "status": status,
                            "stored_as": stored_name,
                        }
                    )
                report["snapshots"].sort(
                    key=lambda item: (item["version"], item["fingerprint"])
                )
                report["summary"].update(
                    {
                        "input_count": len(path_list),
                        "snapshot_count": len(report["snapshots"]),
                    }
                )
        except LockConflict as exc:
            report["errors"].append(_finding("E_LOCK", str(exc)))
        except OSError as exc:
            report["errors"].append(
                _finding("E_IO", f"workspace operation failed: {exc}")
            )
        return self._finalize(report)

    def recover(
        self, paths: Iterable[str | os.PathLike[str]] | None = None
    ) -> dict[str, Any]:
        report = self._report("recover")
        report["snapshots"] = []
        report["effective_snapshots"] = []
        report["excluded_paths"] = []
        report["recovered_paths"] = []
        report["forks"] = []
        report["rollback_versions"] = []
        path_list = [Path(path).expanduser() for path in (paths or [])]
        lock = WorkspaceLock(self.lock_path)
        try:
            with lock:
                self.workspace.mkdir(parents=True, exist_ok=True)
                cleanup_orphan_temps(self.workspace)
                explicit_fingerprints: set[str] = set()
                for path in path_list:
                    snapshot, findings = self._parse_snapshot_file(
                        path, include_source=True
                    )
                    if snapshot is None:
                        report["errors"].extend(findings)
                        continue
                    explicit_fingerprints.add(snapshot.fingerprint)
                    try:
                        self._store_snapshot(snapshot)
                    except OSError as exc:
                        report["errors"].extend(findings)
                        report["errors"].append(
                            _finding(
                                "E_IO",
                                f"cannot store snapshot before recovery: {exc}",
                                version=snapshot.version,
                                fingerprint=snapshot.fingerprint,
                                source=path.name,
                            )
                        )

                snapshots, store_findings = self._load_snapshots_locked()
                report["errors"].extend(store_findings)
                baseline, checkpoint_findings, rebuilt, previous_checkpoint = (
                    self._load_checkpoint_for_recovery()
                )
                report["warnings"].extend(checkpoint_findings)
                structure = self._evaluate_structure(
                    snapshots, baseline, explicit_fingerprints
                )
                report["errors"].extend(structure["findings"])
                lineage = self._select_lineage(snapshots, structure, baseline)

                if baseline is None:
                    entries: dict[str, dict[str, Any]] = {}
                    exclusions: dict[str, dict[str, Any]] = {}
                    applied: list[dict[str, Any]] = []
                else:
                    entries = {path: dict(item) for path, item in baseline["entries"].items()}
                    exclusions = {
                        path: dict(item) for path, item in baseline["exclusions"].items()
                    }
                    applied = [dict(item) for item in baseline["applied_snapshots"]]

                self._apply_entries(entries, exclusions, lineage)
                applied.extend(item.meta() for item in lineage)
                should_commit = bool(lineage) or rebuilt
                committed_checkpoint = previous_checkpoint
                if should_commit:
                    try:
                        committed_checkpoint = self._commit_state(
                            entries=entries,
                            exclusions=exclusions,
                            applied_snapshots=applied,
                            previous_checkpoint=previous_checkpoint,
                        )
                        report["summary"]["committed"] = True
                    except (OSError, RuntimeError) as exc:
                        report["errors"].append(
                            _finding("E_IO", f"atomic checkpoint commit failed: {exc}")
                        )
                        report["summary"]["committed"] = False
                else:
                    report["summary"]["committed"] = False

                for snapshot in snapshots:
                    entry_failures = [
                        finding
                        for result in snapshot.entry_results
                        for finding in result.findings
                    ]
                    report["errors"].extend(entry_failures)
                    report["snapshots"].append(
                        {
                            **snapshot.meta(),
                            "effective": snapshot.fingerprint
                            in {item["fingerprint"] for item in applied},
                            "entry_failures": _sorted_findings(entry_failures),
                            "status": self._snapshot_status(
                                snapshot,
                                lineage=lineage,
                                baseline=baseline,
                                structure=structure,
                            ),
                        }
                    )
                report["snapshots"].sort(
                    key=lambda item: (item["version"], item["fingerprint"])
                )
                report["effective_snapshots"] = sorted(
                    applied, key=lambda item: (item["version"], item["fingerprint"])
                )
                report["excluded_paths"] = sorted(
                    (
                        {
                            "codes": sorted({str(code) for code in item.get("codes", [])}),
                            "path": path,
                            "versions": sorted(
                                {
                                    value
                                    for value in item.get("versions", [])
                                    if isinstance(value, int)
                                }
                            ),
                        }
                        for path, item in exclusions.items()
                    ),
                    key=lambda item: path_sort_key(item["path"]),
                )
                report["recovered_paths"] = [
                    entry["path"]
                    for entry in sorted(
                        entries.values(), key=lambda item: path_sort_key(item["path"])
                    )
                ]
                report["forks"] = [
                    finding
                    for finding in structure["findings"]
                    if finding.get("code") == "E_FORK"
                ]
                for finding in structure["findings"]:
                    if finding.get("code") == "E_ROLLBACK":
                        report["rollback_versions"] = sorted(
                            set(report["rollback_versions"])
                            | set(finding.get("versions", []))
                        )
                checkpoint_info: dict[str, Any] = {
                    "committed": report["summary"].get("committed", False),
                    "present": committed_checkpoint is not None,
                    "rebuilt": rebuilt,
                }
                if committed_checkpoint is not None:
                    checkpoint_info.update(
                        {
                            "generation": committed_checkpoint.get("generation"),
                            "head_fingerprint": committed_checkpoint.get(
                                "head_fingerprint"
                            ),
                            "watermark": committed_checkpoint.get("watermark"),
                        }
                    )
                report["checkpoint"] = checkpoint_info
                report["summary"].update(
                    {
                        "applied_snapshot_count": len(applied),
                        "excluded_path_count": len(report["excluded_paths"]),
                        "recovered_path_count": len(report["recovered_paths"]),
                        "snapshot_count": len(report["snapshots"]),
                    }
                )
        except LockConflict as exc:
            report["errors"].append(_finding("E_LOCK", str(exc)))
        except OSError as exc:
            report["errors"].append(
                _finding("E_IO", f"workspace operation failed: {exc}")
            )
        return self._finalize(report)

    def status(self) -> dict[str, Any]:
        report = self._report("status")
        checkpoint, present, read_error = self._read_checkpoint()
        report["checkpoint"] = {"present": present, "valid": False}
        report["state"] = None
        if checkpoint is None:
            if present:
                report["errors"].append(
                    _finding("E_CHECKPOINT", read_error or "checkpoint is unreadable")
                )
            return self._finalize(report)
        valid, findings, restore = self._validate_checkpoint_data(checkpoint)
        policy_matches = checkpoint.get("policy") == self.policy.to_json()
        needs_rebuild = (
            checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or not policy_matches
        )
        report["checkpoint"].update(
            {
                "generation": checkpoint.get("generation"),
                "head_fingerprint": checkpoint.get("head_fingerprint"),
                "needs_rebuild": needs_rebuild,
                "policy_matches": policy_matches,
                "valid": valid,
                "watermark": checkpoint.get("watermark"),
            }
        )
        if not valid:
            report["errors"].extend(findings)
            return self._finalize(report)
        report["warnings"].extend(findings)
        if needs_rebuild:
            report["warnings"].append(
                _finding(
                    "E_CHECKPOINT_REBUILD",
                    "the next recover operation will rebuild under the active policy",
                )
            )
        assert restore is not None
        report["state"] = {
            "applied_snapshots": restore.get("applied_snapshots", []),
            "entries": restore.get("entries", []),
            "excluded_paths": restore.get("excluded_paths", []),
            "head_fingerprint": restore.get("head_fingerprint"),
            "state_sha256": restore.get("state_sha256"),
            "watermark": restore.get("watermark"),
        }
        return self._finalize(report)

    def verify_checkpoint(self) -> dict[str, Any]:
        report = self._report("verify-checkpoint")
        checkpoint, present, read_error = self._read_checkpoint()
        report["checkpoint_valid"] = False
        if checkpoint is None:
            report["errors"].append(
                _finding(
                    "E_CHECKPOINT",
                    read_error
                    if present
                    else "no checkpoint exists in this workspace",
                )
            )
            return self._finalize(report)
        valid, findings, restore = self._validate_checkpoint_data(checkpoint)
        report["checkpoint_valid"] = valid
        if not valid:
            report["errors"].extend(findings)
            return self._finalize(report)
        report["checkpoint"] = {
            "generation": checkpoint.get("generation"),
            "head_fingerprint": checkpoint.get("head_fingerprint"),
            "policy_matches": checkpoint.get("policy") == self.policy.to_json(),
            "restore_sha256": checkpoint.get("restore_sha256"),
            "state_sha256": checkpoint.get("state_sha256"),
            "watermark": checkpoint.get("watermark"),
        }
        report["summary"].update(
            {
                "entry_count": len(restore.get("entries", []))
                if restore is not None
                else 0,
                "applied_snapshot_count": len(
                    checkpoint.get("applied_snapshots", [])
                ),
            }
        )
        return self._finalize(report)
